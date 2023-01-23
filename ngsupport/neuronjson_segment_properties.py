import json
import logging

import requests
import numpy as np
import pandas as pd

from flask import Response, jsonify
from http import HTTPStatus

logger = logging.getLogger(__name__)
VALID_PROP_TYPES = ['label', 'description', 'tags', 'string', 'number']


def neuronjson_segment_properties_info(server, uuid, instance, label, altlabel=None):
    try:
        info = _neuronjson_segment_properties_info(server, uuid, instance, label, altlabel)
        return jsonify(info), HTTPStatus.OK
    except requests.HTTPError as ex:
        return Response(ex.response.content, ex.response.status_code)


def _neuronjson_segment_properties_info(server, uuid, instance, label, altlabel):
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
        if ':' in server:
            server = f'http://{server}'
        else:
            server = f'https://{server}'

    if not label and not altlabel:
        return Response("No fields specified", HTTPStatus.BAD_REQUEST)

    if not label:
        label, altlabel = altlabel, None
    if label == altlabel:
        altlabel = None

    # Fetch from DVID
    need_user = bool(('_user' in label) or (altlabel and '_user' in altlabel))
    need_time = bool(('_time' in label) or (altlabel and '_time' in altlabel))
    show = {
        (False, False): None,
        (True, False): 'user',
        (False, True): 'time',
        (True, True): 'all',
    }[(need_user, need_time)]

    fields = [label]
    if altlabel:
        fields = [label, altlabel]

    df = fetch_all(server, uuid, instance, fields=fields, show=show, format='pandas')
    if len(df) == 0:
        return serialize_segment_properties_info(df)

    if label not in df.columns:
        df[label] = np.nan
    if altlabel not in df.columns:
        df[altlabel] = np.nan

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
        propname = f'{label} [{altlabel}]'
        df[propname] = df[label] + df[altlabel]
    else:
        # Discard rows we don't care about
        valid_rows = ~df[label].isnull()
        df = df.loc[valid_rows, [label]].copy()
        propname = label

    # Convert to neuroglancer JSON format
    return serialize_segment_properties_info(df[propname])


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


def serialize_segment_properties_info(df, prop_types={}, output_path=None):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/segment_properties.md

    Note:
        This function doesn't yet support 'tags'.

    Args:
        df:
            DataFrame or Series.  Index must be named 'body'.
            Every column will be interpreted as a segment property.

        prop_types:
            Dict to specify the neuroglancer property type of each column, e.g. {'instance': 'label'}.
            For columns not listed in the dict, the property type is inferred from the name of the column
            (if the name is 'label' or 'description') or the dtype of the column (string vs. number).

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict)
    """
    assert df.index.name == 'body'
    if isinstance(df, pd.Series):
        df = df.to_frame()
    invalid_prop_types = set(prop_types.values()) - set(VALID_PROP_TYPES)
    assert not invalid_prop_types, \
        f"Invalid property types: {invalid_prop_types}"

    assert 'tags' not in prop_types.values(), \
        "Sorry, 'tags' properties aren't yet supported by this function."

    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': []
        }
    }

    # If there's only one column, assume it's the 'label' property
    if not prop_types and len(df.columns) == 1:
        prop_types = {df.columns[0]: 'label'}

    default_prop_types = {
        'label': 'label',
        'description': 'description'
    }
    prop_types = default_prop_types | prop_types

    for col in df.columns:
        prop = {}
        prop['id'] = col

        if np.issubdtype(df[col].dtype, np.number):
            assert not df[col].dtype in (np.int64, np.uint64), \
                "Neuroglancer doesn't support 64-bit integer properties.  Use int32 or float64"
            prop['type'] = 'number'
            prop['data_type'] = df[col].dtype.name
            assert not df[col].isnull().any(), \
                (f"Column {col} contans NaN entries. "
                 "I'm not sure what to do with NaN values in numeric properties.")
            prop['values'] = df[col].tolist()
        else:
            prop['type'] = prop_types.get(col, 'string')
            prop['values'] = df[col].fillna("").astype(str).tolist()

        info['inline']['properties'].append(prop)

    _validate_property_type_counts(info)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
    return info


def _validate_property_type_counts(info):
    type_counts = (
        pd.Series([prop['type'] for prop in info['inline']['properties']])
        .value_counts()
        .reindex(VALID_PROP_TYPES)
        .fillna(0)
        .astype(int)
    )
    for t in ['label', 'description', 'tags']:
        assert type_counts.loc[t] <= 1, \
            f"Can't have more than one property with type '{t}'"

    if type_counts.loc['label'] == 0 and type_counts.loc['string'] > 0:
        logger.warning("None of your segment properties are of type 'label', "
                       "so none will be displayed in the neuroglancer UI.")


def fetch_all(server, uuid, instance='segmentation_annotations', *, show=None, fields=None, format='pandas', session=None):
    if session is None:
        session = requests.Session()
        session.params = {'app': 'ngsupport'}

    assert show in ('user', 'time', 'all', None)
    assert format in ('pandas', 'json')

    params = {}
    if show:
        params['show'] = show

    if fields:
        if isinstance(fields, str):
            fields = [fields]
        params['fields'] = ','.join(fields)

    url = f'{server}/api/node/{uuid}/{instance}/all'
    r = session.get(url, params=params)
    r.raise_for_status()
    values = r.json() or []

    if format == 'pandas':
        if not values:
            return pd.DataFrame([]).rename_axis('body')
        return pd.DataFrame(values).set_index('bodyid').rename_axis('body')
    else:
        return sorted(values, key=lambda d: d['bodyid'])


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
