import json
import logging
from itertools import chain

import requests
import numpy as np
import pandas as pd

from flask import Response, jsonify
from http import HTTPStatus

logger = logging.getLogger(__name__)
VALID_SEGMENT_PROPERTY_TYPES = ['label', 'description', 'tags', 'string', 'number']


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
        return serialize_segment_properties_info(df)

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
    return serialize_segment_properties_info(df[propname])


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
        return serialize_segment_properties_info(df)

    df = df[tags]
    valid_rows = (df.notnull() & df != "").any(axis=1)
    df = df.loc[valid_rows]

    info = serialize_segment_properties_info(df, tags_columns=tags, prefix_tags=True)
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

    info = serialize_segment_properties_info(top_syn)
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
    info = serialize_segment_properties_info(top_note)
    return jsonify(info), HTTPStatus.OK


def serialize_segment_properties_info(df, prop_types={}, tags_columns=[], prefix_tags=False, output_path=None):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/segment_properties.md

    Args:
        df:
            DataFrame or Series.  Index must be named 'body'.
            Every column will be interpreted as a segment property.

        prop_types:
            Dict to specify the neuroglancer property type of each column, e.g. {'instance': 'label'}.
            For columns not listed in the dict, the property type is inferred from the name of the column
            (if the name is 'label' or 'description') or the dtype of the column (string vs. number).

        tags_columns:
            The list of columns which should be used to generate the (combined) 'tags' property.
            (You can also specify tags columns directly in the prop_types argument.)

        prefix_tags:
            If True, all tags will be prefixed with the name of the column they came from,
            e.g. 'status:Anchor'

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict)
    """
    assert df.index.name == 'body'
    if isinstance(df, pd.Series):
        df = df.to_frame()

    prop_types, tags_columns = _reconcile_prop_types(
        df.columns, prop_types, tags_columns
    )

    json_props = []
    for col in {*df.columns} - {*tags_columns}:
        prop = _property_json(df[col], prop_types)
        json_props.append(prop)

    # Tags are a special case
    tags_prop_json = _tags_property_json(df, tags_columns, prefix_tags)
    if tags_prop_json:
        json_props.append(tags_prop_json)

    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': json_props
        }
    }

    _validate_property_type_counts(info)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
    return info


def _reconcile_prop_types(col_names, prop_types, tags_columns):
    """
    Helper for serialize_segment_properties_info().

    Validate the property types and tag columns,
    append additional tag_columns if some were listed in prop_types,
    and insert default prop_types if needed.
    """
    prop_types = dict(prop_types)
    tags_columns = list(tags_columns)

    invalid_prop_types = set(prop_types.values()) - set(VALID_SEGMENT_PROPERTY_TYPES)
    assert not invalid_prop_types, \
        f"Invalid property types: {invalid_prop_types}"

    tag_props = {k for k,v in prop_types.items() if v == 'tags'}
    non_tag_props = {k:v for k,v in prop_types.items() if v != 'tags'}

    assert not (ambiguous_cols := set(non_tag_props.keys()) & set(tags_columns)), \
        "Ambiguous property type for columns: " \
        f"{ {k: v for k, v in non_tag_props if k in ambiguous_cols} }"

    tags_columns = {*tags_columns, *tag_props}
    prop_types |= {c: 'tags' for c in tags_columns}

    # If there's only one column, assume it's the 'label' property
    if not prop_types and len(col_names) == 1:
        prop_types = {col_names[0]: 'label'}

    default_prop_types = {
        'label': 'label',
        'description': 'description'
    }
    prop_types = default_prop_types | prop_types

    return prop_types, tags_columns


def _property_json(s, prop_types):
    """
    Helper for serialize_segment_properties_info().

    Constructs the JSON for a segment property, other than the 'tags'
    property, which is implemented in _tags_property_json()
    """
    prop = {}
    prop['id'] = s.name

    if np.issubdtype(s.dtype, np.number):
        assert s.dtype not in (np.int64, np.uint64), \
            "Neuroglancer doesn't support 64-bit integer properties.  Use int32 or float64"
        assert not s.isnull().any(), \
            (f"Column {s.name} contans NaN entries. "
             "I'm not sure what to do with NaN values in numeric properties.")
        prop['type'] = 'number'
        prop['data_type'] = s.dtype.name
        prop['values'] = s.tolist()
    else:
        prop['type'] = prop_types.get(s.name, 'string')
        prop['values'] = s.fillna("").astype(str).tolist()

    return prop


def _tags_property_json(df, tags_columns, add_prefix):
    """
    Helper for serialize_segment_properties_info().
    Constructs the JSON for the 'tags' segment property.
    """
    if not tags_columns:
        return None

    df = df[[*tags_columns]].copy()

    for c in df.columns:
        # spaces are forbidden in tags
        df[c] = df[c].astype('string').str.replace(' ', '_')

        # treat empty string as null
        df[c] = df[c].replace('', None)

        # Convert each series to categorical before we combine categories below
        df[c] = df[c].astype('category')
        if add_prefix:
            prefix = c.replace(' ', '_')
            prefixed_categories = [f'{prefix}:{cat}' for cat in df[c].dtype.categories]
            df[c] = df[c].cat.rename_categories(prefixed_categories)

    # Convert to a single big categorical dtype
    all_tags = sorted({*chain(*(df[col].dtype.categories for col in df.columns))})
    df = df.astype(pd.CategoricalDtype(categories=all_tags))

    # Tags are written as a list-of-lists of sorted codes
    codes_df = pd.DataFrame({c: df[c].cat.codes for c in df.columns}).values
    codes_df.sort(axis=1)
    codes_lists = [
        [x for x in row if x != -1]  # Drop nulls
        for row in codes_df.tolist()
    ]

    prop = {
        'id': 'tags',
        'type': 'tags',
        'tags': all_tags,
        'values': codes_lists,
    }
    return prop


def _validate_property_type_counts(info):
    """
    Helper for serialize_segment_properties_info().
    Asserts that only one property has the 'labels' type.
    Also checks 'description' and 'tags'
    """
    type_counts = (
        pd.Series([prop['type'] for prop in info['inline']['properties']])
        .value_counts()
        .reindex(VALID_SEGMENT_PROPERTY_TYPES)
        .fillna(0)
        .astype(int)
    )
    for t in ['label', 'description', 'tags']:
        assert type_counts.loc[t] <= 1, \
            f"Can't have more than one property with type '{t}'"

    if type_counts.loc['label'] == 0 and type_counts.loc['string'] > 0:
        logger.warning("None of your segment properties are of type 'label', "
                       "so none will be displayed in the neuroglancer UI.")


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
