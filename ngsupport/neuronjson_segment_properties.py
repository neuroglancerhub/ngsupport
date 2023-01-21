import json
import logging

import requests
import numpy as np
import pandas as pd

from flask import Response, jsonify
from http import HTTPStatus

logger = logging.getLogger(__name__)


def neuronjson_segment_properties_info(server, uuid, instance, label, altlabel=None):
    """
    Respond to the segment properties /info endpoint:
    /neuronjson_segment_synapse_properties/<server>/<uuid>/<instance>/<int:n>/info

    If an 'altlabel' column is specified, it will be appended to the label in brackets,
    e.g. "Mi1 [Anchor]"

    - Fetches all annotation data from DVID
    - selects the column(s) named in 'label' and (optionally) 'altlabel',
    - discards segment IDs which are empty for the selected columns.
    - combines label+altlabel into a single string for neuroglancer display,
    - converts the results to the JSON format neuroglancer expects to see for segment properties.
    """
    if not server.startswith('http'):
        server = f'https://{server}'

    # Fetch from DVID
    params = {'app': 'ngsupport'}
    if '_user' in label or (altlabel and '_user' in altlabel):
        params = {'show': 'user'}
    if '_time' in label or (altlabel and '_time' in altlabel):
        params = {'show': 'all'}

    r = requests.get(f"{server}/api/node/{uuid}/{instance}/all", params=params)

    if r.status_code != 200:
        msg = r.content.decode('utf-8')
        return Response(msg, r.status_code)

    df = pd.DataFrame(r.json()).set_index('bodyid').rename_axis('body')

    # If using group column, convert to strings.
    if 'group' in (label, altlabel):
        valid_group = ~df['group'].isnull()
        df.loc[valid_group, 'group'] = df.loc[valid_group, 'group'].astype(int).astype(str)

    df.loc[df[label] == "", label] = np.nan

    if altlabel:
        # Discard rows we don't care about
        df.loc[df[altlabel] == "", altlabel] = np.nan
        valid_rows = ~df[label].isnull() | ~df[altlabel].isnull()
        df = df.loc[valid_rows, [label, altlabel]].fillna('').copy()

        # Concatenate main and secondary columns into
        # a single 'label' for neuroglancer to display.
        nonempty_alt = (df[altlabel] != "")
        df.loc[nonempty_alt, altlabel] = ' [' + df.loc[nonempty_alt, altlabel] + ']'
        df['label'] = df[label] + df[altlabel]
    else:
        # Discard rows we don't care about
        valid_rows = ~df[label].isnull()
        df = df.loc[valid_rows, [label]].copy()
        df['label'] = df[label]

    # Convert to neuroglancer JSON format
    info = serialize_segment_properties_info(df[['label']])
    return jsonify(info), HTTPStatus.OK


def neuronjson_segment_synapse_properties_info(server, uuid, instance, n):
    """
    Fetch the synapse counts (PreSyn and PostSyn) for the top N bodies and use them to
    return neuroglancer numerical segment properties in the appropriate JSON format.
    """
    if not server.startswith('http'):
        server = f'https://{server}'

    # Fetch from DVID
    top_tbar = fetch_top(server, uuid, instance, n, 'PreSyn')
    top_psd = fetch_top(server, uuid, instance, n, 'PostSyn')
    top_syn = top_tbar.to_frame().merge(top_psd, 'outer', on='body')
    missing_tbar = fetch_counts(server, uuid, instance, top_syn.loc[top_syn['PreSyn'].isnull()].index, 'PreSyn')
    missing_psd = fetch_counts(server, uuid, instance, top_syn.loc[top_syn['PostSyn'].isnull()].index, 'PostSyn')
    top_syn['PreSyn'].update(missing_tbar)
    top_syn['PostSyn'].update(missing_psd)
    top_syn = top_syn.astype(int)

    info = serialize_segment_properties_info(top_syn.astype(np.int32))
    return jsonify(info), HTTPStatus.OK


def serialize_segment_properties_info(df, output_path=None):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/segment_properties.md

    Note:
        This function doesn't yet support 'tags'.

    Args:
        df:
            DataFrame.  Index must be named 'body'.
            Every column will be interpreted as a segment property.
            Note that properties named 'label' and 'description' have special meaning in neuroglancer.
        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict)
    """
    assert df.index.name == 'body'
    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': []
        }
    }

    if 'tags' in df.columns:
        raise NotImplementedError("tags not yet supported")

    for col in df.columns:
        prop = {}
        prop['id'] = col
        if col in ('label', 'description'):
            prop['type'] = col
            prop['values'] = df[col].fillna("").astype(str).tolist()
        elif np.issubdtype(df[col].dtype, np.number):
            assert not df[col].dtype in (np.int64, np.uint64), \
                "Neuroglancer doesn't support 64-bit integer properties.  Use int32 or float64"
            prop['type'] = 'number'
            prop['data_type'] = df[col].dtype.name
            assert not df[col].isnull().any(), \
                (f"Column {col} contans NaN entries. "
                 "I'm not sure what to do with NaN values in numeric properties.")
            prop['values'] = df[col].tolist()
        else:
            prop['type'] = 'string'
            prop['values'] = df[col].fillna("").astype(str).tolist()

        info['inline']['properties'].append(prop)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
    return info


def fetch_top(server, uuid, instance, n, element_type, format='pandas'):
    assert format in ('json', 'pandas')

    url = f'{server}/api/node/{uuid}/{instance}/top/{n}/{element_type}'
    r = requests.get(url, params={'app': 'ngsupport'})
    r.raise_for_status()

    if format == 'json':
        return r.json()

    results = r.json()
    if len(results) == 0:
        return pd.Series([], name=element_type, dtype=int).rename_axis('body')

    df = pd.DataFrame(results)
    df = df.rename(columns={'Label': 'body', 'Size': element_type})
    df = df.sort_values([element_type, 'body'], ascending=[False, True])
    s = df.set_index('body')[element_type]
    return s


def fetch_counts(server, uuid, instance, bodies, element_type, format='pandas'):
    """
    Returns the count of the given annotation element type for the given labels.

    For synapse indexing, the labelsz data instance must be synced with an annotations instance.
    (future) For number-of-voxels indexing, the labelsz data instance must be synced with a labelvol instance.

    Args:
        server:
            dvid server, e.g. 'emdata4:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid labelsz instance name

        bodies:
            A list of body IDs

        element_type:
            An indexed element type supported by the labelsz instance,
            i.e. one of: "PostSyn", "PreSyn", "Gap", "Note",
            or the catch-all for synapses "AllSyn", or the number of voxels "Voxels".

    Returns:
        JSON or pd.Series, depending on requested format.
        JSON example:

            [{ "Label": 21847,  "PreSyn": 81 }, { "Label": 23, "PreSyn": 65 }, ...]

        If a Series is returned, it's indexed by body
    """
    bodies = np.array(bodies).tolist()
    url = f'{server}/api/node/{uuid}/{instance}/counts/{element_type}'
    r = requests.get(url, json=bodies, params={'app': 'ngsupport'})
    r.raise_for_status()
    counts = r.json()

    if format == 'json':
        return counts

    counts = pd.DataFrame(counts).set_index('Label').rename_axis('body')[element_type]
    return counts
