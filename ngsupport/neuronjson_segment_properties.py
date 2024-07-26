import logging

import requests
import numpy as np
import pandas as pd

from flask import Response, jsonify
from http import HTTPStatus

from .neuroglancer import segment_properties_json

logger = logging.getLogger(__name__)


def neuronjson_segment_properties_info(server, uuid, instance, label, altlabel=None):
    try:
        info = _neuronjson_segment_properties_info(server, uuid, instance, label, altlabel)
        return jsonify(info), HTTPStatus.OK
    except requests.HTTPError as ex:
        return Response(ex.response.content, ex.response.status_code)


def _neuronjson_segment_properties_info(server, uuid, instance, label, altlabel):
    """
    Respond to the segment properties /info endpoint:
    /neuronjson_segment_properties/<server>/<uuid>/<instance>/<label>/<altlabel>/info

    If an 'altlabel' parameter is specified, the corresponding field will be appended
    to the label in brackets, e.g. "Mi1 [Anchor]"

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
        return segment_properties_json(df)

    if label not in df.columns:
        df[label] = np.nan
    if altlabel not in df.columns:
        df[altlabel] = np.nan

    df[label] = convert_to_string(df[label])
    df[altlabel] = convert_to_string(df[altlabel])

    if altlabel:
        # Discard rows we don't care about
        valid_rows = (df[label] != "") | (df[altlabel] != "")
        df = df.loc[valid_rows, [label, altlabel]]

        # Concatenate main and secondary columns into
        # a single 'label' for neuroglancer to display.
        nonempty_alt = (df[altlabel] != "")
        df.loc[nonempty_alt, altlabel] = ' [' + df.loc[nonempty_alt, altlabel] + ']'
        propname = f'{label} [{altlabel}]'
        df[propname] = df[label] + df[altlabel]
    else:
        # Discard rows we don't care about
        valid_rows = (df[label] != "")
        df = df.loc[valid_rows, [label]]
        propname = label

    # Convert to neuroglancer JSON format
    return segment_properties_json(df[propname])


def convert_to_string(s):
    """
    Convert a series to string, but if it's a float series, convert all values which
    happen to be integers into actual ints first (to avoid strings like "123.0").
    """
    if s.dtype == object:
        return s.fillna('').astype(str)
    if np.issubdtype(s.dtype, np.integer):
        return s.astype(str)
    if np.issubdtype(s.dtype, np.floating):
        s = s.astype(object)
        int_rows = s.notnull() & ~((s % 1).astype(bool))
        s.loc[int_rows] = s.loc[int_rows].astype(int)
        return s.fillna('').astype(str)


def neuronjson_segment_tags_properties_info(server, uuid, instance, tags):
    """
    Respond to the segment tags /info endpoint:
    /neuronjson_segment_tags_properties/<server>/<uuid>/<instance>/tags/info

    where the 'tags' parameter is expected to be a comma-delimited list of annotation field names.

    - Fetches all annotation data from DVID
    - selects the column(s) named in 'tags'
    - discards segment IDs which are empty for the selected columns.
    - Constructs a single 'tags' property (as JSON) for neuroglancer to display.
    """
    if not server.startswith('http'):
        server = f'https://{server}'

    tags = tags.split(',')
    if not tags:
        return Response("No fields specified", HTTPStatus.BAD_REQUEST)

    # Fetch from DVID
    show = None
    if any('_user' in t for t in tags):
        show = 'user'

    df = fetch_all(server, uuid, instance, fields=tags, show=show, format='pandas')
    if len(df) == 0:
        return segment_properties_json(df)

    info = segment_properties_json(df[tags], tag_cols=tags)
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
    top_syn = top_syn.astype(np.int32)

    info = segment_properties_json(top_syn)
    return jsonify(info), HTTPStatus.OK


def neuronjson_segment_note_properties_info(server, uuid, instance, propname, n):
    """
    Fetch the annotation 'Note' counts for the top N bodies and use them to
    return neuroglancer numerical segment properties in the appropriate JSON format.

    Except for synapses, most point annotation instances in dvid just store a
    generic 'Note' annotation type.  Examples of such instances in our CNS include:
    - neck-fibers-anterior
    - nuclei-centroids
    - segmentation_todo
    - bookmark_annotations

    Args:
        server:
            DVID server
        uuid:
            DVID uuid
        instance:
            DVID 'annotation' instance name
        propname:
            Arbitrary string.  Will become the name of the property in the neuroglancer display.
        n: How many segment IDs to fetch via the /top endpoint

    Returns:
        neuroglancer segment properties JSON data
    """
    if not server.startswith('http'):
        server = f'https://{server}'

    # Fetch from DVID
    top_note = (
        fetch_top(server, uuid, instance, n, 'Note')
        .rename(propname)
        .astype(np.int32)
        .to_frame()
    )
    info = segment_properties_json(top_note)
    return jsonify(info), HTTPStatus.OK


##
## DVID access
##

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
