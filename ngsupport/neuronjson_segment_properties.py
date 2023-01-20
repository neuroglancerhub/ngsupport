import json
import logging

import requests
import numpy as np
import pandas as pd

from flask import Response, jsonify
from http import HTTPStatus

logger = logging.getLogger(__name__)


def neuronjson_segment_properties_info(server, uuid, instance, label, altlabel):
    if not server.startswith('http'):
        server = f'https://{server}'

    # Fetch from DVID
    params = {'app': 'ngsupport'}
    if '_user' in label or '_user' in altlabel:
        params = {'show': 'user'}
    if '_time' in label or '_time' in altlabel:
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
    df.loc[df[altlabel] == "", altlabel] = np.nan

    valid_rows = ~df[label].isnull() | ~df[altlabel].isnull()
    df = df.loc[valid_rows, [label, altlabel]].fillna('').copy()

    nonempty_alt = (df[altlabel] != "")
    df.loc[nonempty_alt, altlabel] = ' [' + df.loc[nonempty_alt, altlabel] + ']'
    df['label'] = df[label] + df[altlabel]

    info = serialize_segment_properties_info(df[['label']])
    return jsonify(info), HTTPStatus.OK


def serialize_segment_properties_info(df, output_path=None):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/segment_properties.md
    """
    assert df.index.name == 'body'
    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': []
        }
    }

    for col in df.columns:
        prop = {}
        prop['id'] = col
        if col in ('label', 'description'):
            prop['type'] = col
            prop['values'] = df[col].fillna("").astype(str).tolist()
        elif np.issubdtype(df[col].dtype, np.number):
            prop['type'] = 'number'
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
