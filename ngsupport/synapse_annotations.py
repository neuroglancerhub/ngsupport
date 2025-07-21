import json
import logging
import tempfile
from functools import cache

import requests
import numpy as np
import pandas as pd

from flask import Response, jsonify
from http import HTTPStatus

from .util import normalize_server

logger = logging.getLogger(__name__)

@normalize_server
def synapse_annotations_info(server, uuid, instance):
    info = _synapse_annotations_info(server, uuid, instance)
    return jsonify(info), HTTPStatus.OK

@normalize_server
def synapse_annotations_by_id(server, uuid, instance, syn_id):
    syn_id = int(syn_id)
    return _synapse_annotations_by_id(server, uuid, instance, syn_id), HTTPStatus.OK

@normalize_server
def synapse_annotations_by_related_id(server, uuid, instance, relationship, segment_id):
    segment_id = int(segment_id)
    return _synapse_annotations_by_related_id(server, uuid, instance, relationship, segment_id), HTTPStatus.OK

@cache
def _synapse_annotations_info(server, uuid, syn_instance):
    # Figure out which segmentation synapse instance is sync'd to
    # and determine the resolution.
    syn_info, seg_info = _get_instance_infos(server, uuid, syn_instance)

    properties = json.loads((syn_info["Base"].get("Tags") or {}).get("neuroglancer-properties", '[]'))
    relationships = json.loads((syn_info["Base"].get("Tags") or {}).get("neuroglancer-relationships", '[]'))

    seg_name = seg_info["Base"]["Name"]
    seg_units = seg_info["Extended"]["VoxelUnits"]
    if seg_units != ["nanometers"] * 3:
        raise RuntimeError(f"Segmentation instance {seg_name} has voxel units {seg_units}, not nanometers")

    nm_x, nm_y, nm_z = seg_info["Extended"]["VoxelSize"]

    box_xyz = np.array((
        seg_info["Extended"]["MinPoint"],
        seg_info["Extended"]["MaxPoint"]
    ))
    box_xyz[1] += 1

    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {
            "x": [nm_x * 1e-09, "m"],
            "y": [nm_y * 1e-09, "m"],
            "z": [nm_z * 1e-09, "m"],
        },
        "lower_bound": box_xyz.tolist(),
        "upper_bound": box_xyz.tolist(),
        "annotation_type": "line",
        "properties": properties,
        "by_id": {
            "key": "by_id"
        },
        "relationships": relationships,
        "spatial": []
    }

    return info


@cache
def _get_instance_infos(server, uuid, syn_instance):
    syn_info = fetch_instance_info(server, uuid, syn_instance)

    try:
        seg_instance = syn_info["Base"]["Syncs"][0]
    except Exception:
        # Return a flask error appropriate for invalid parameters.
        return Response(f"No sync'd segmentation instance found for {syn_instance}", HTTPStatus.BAD_REQUEST)

    seg_info = fetch_instance_info(server, uuid, seg_instance)
    return syn_info, seg_info


def _synapse_annotations_by_id(server, uuid, instance, syn_id):
    from neuroglancer.coordinate_space import CoordinateSpace
    from neuclease.util import decode_coords_from_uint64
    from neuclease.dvid.annotation import fetch_elements
    from neuclease.misc.neuroglancer.annotations.precomputed import write_precomputed_annotations

    info = _synapse_annotations_info(server, uuid, instance)

    syn_ids = np.array([syn_id], np.uint64)
    zyx = decode_coords_from_uint64(syn_ids)[0]
    syn_df, rel_df = fetch_elements(server, uuid, instance, [zyx, zyx+1], relationships=True, format='pandas')
    partner_df = _fetch_partner_properties(server, uuid, instance, syn_df, rel_df, info)
    assert len(partner_df) == 1, "Expected exactly one partner"

    with tempfile.TemporaryDirectory() as tmpdir:
        write_precomputed_annotations(
            partner_df,
            coord_space=CoordinateSpace(json=info['dimensions']),
            annotation_type='line',
            properties=info['properties'],
            relationships=[r['id'] for r in info['relationships']],
            output_dir=tmpdir,
            write_sharded=False,
            write_by_id=True,
            write_by_relationship=False,
            write_single_spatial_level=False,
        )

        return open(f'{tmpdir}/by_id/{syn_id}', 'rb').read()


def _synapse_annotations_by_related_id(server, uuid, instance, relationship, segment_id):
    from neuroglancer.coordinate_space import CoordinateSpace
    from neuclease.dvid.annotation import fetch_label
    from neuclease.misc.neuroglancer.annotations.precomputed import write_precomputed_annotations

    info = _synapse_annotations_info(server, uuid, instance)

    rel_id = info['relationships'][relationship]['id']
    syn_df, rel_df = fetch_label(server, uuid, instance, segment_id, relationships=True, format='pandas')
    partner_df = _fetch_partner_properties(server, uuid, instance, syn_df, rel_df, info)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_precomputed_annotations(
            partner_df,
            coord_space=CoordinateSpace(json=info['dimensions']),
            annotation_type='line',
            properties=info['properties'],
            relationships=[r['id'] for r in info['relationships']],
            output_dir=tmpdir,
            write_sharded=False,
            write_by_id=False,
            write_by_relationship=True,
            write_single_spatial_level=False,
        )

        return open(f'{tmpdir}/by_rel_{rel_id}/{segment_id}', 'rb').read()

def _fetch_partner_properties(server, uuid, instance, syn_df, rel_df, info):
    from neuclease.dvid.util import swap_df_cols, encode_coords_to_uint64
    from neuclease.dvid.annotation import fetch_label, fetch_point_elements_by_block, fetch_labels_batched

    syn_df = syn_df.drop(columns=['tags'])
    rel_df = rel_df.drop(columns=['rel'])

    partner_df = syn_df.merge(rel_df, 'left', on=[*'xyz']).rename(columns={c: f'from_{c}' for c in syn_df.columns})
    partner_df = fetch_point_elements_by_block(
        server, uuid, instance,
        partner_df.rename(columns={f'to_{k}': k for k in 'xyz'})
    )
    partner_df = partner_df.rename(columns={c: f'to_{c}' for c in partner_df.columns if not c.startswith('from_')})

    syn_info, seg_info = _get_instance_infos(server, uuid, instance)
    seg_name = seg_info["Base"]["Name"]
    if 'from_body' not in partner_df.columns:
        partner_df['from_body'] = fetch_labels_batched(
            server, uuid, seg_name,
            partner_df[['from_z', 'from_y', 'from_x']].values,
            batch_size=1000, threads=4
        )

    if 'to_body' not in partner_df.columns:
        partner_df['to_body'] = fetch_labels_batched(
            server, uuid, seg_name,
            partner_df[['to_z', 'to_y', 'to_x']].values,
            batch_size=1000, threads=4
        )

    # We'll convert from 'from_' and 'to_' to '_pre' and '_post' by renaming
    # columns and swapping entries when the 'from_' side isn't a presynapse.
    # Note: swap_df_cols() requires that the suffixes (i.e. '__pre' and '_post') have the same length.
    partner_df = partner_df.rename(columns={c: f'{c[len('from_'):]}__pre' for c in partner_df.columns if c.startswith('from_')})
    partner_df = partner_df.rename(columns={c: f'{c[len('to_'):]}_post' for c in partner_df.columns if c.startswith('to_')})

    swap_df_cols(partner_df, None, partner_df['kind__pre'] == "PostSyn", ['__pre', '_post'])

    # Now we can switch back to the standard suffix: '_pre'
    partner_df = partner_df.rename(columns={c: f'{c[:-len('__pre')]}_pre' for c in partner_df.columns if c.endswith('__pre')})

    # By convention, the annotation ID will be the encoded POST-synaptic point.
    partner_df.index = encode_coords_to_uint64(partner_df[['z_post', 'y_post', 'x_post']].values)

    partner_df = partner_df.rename(columns={f'{k}_pre': f'{k}a' for k in 'xyz'})
    partner_df = partner_df.rename(columns={f'{k}_post': f'{k}b' for k in 'xyz'})

    partner_df = partner_df.drop(columns=['kind_pre', 'kind_post'])

    # Properties from DVID are stored as strings.
    # We'll convert them to the appropriate type.
    dtypes = {}
    for p in info['properties']:
        # We only support enums with ordinary 0..N-1 integer codes.
        if 'enum_labels' in p and np.issubdtype(np.dtype(p['type']), np.integer):
            assert p['enum_values'] == list(range(len(p['enum_labels'])))
            dtypes[p['id']] = pd.CategoricalDtype(p['enum_labels'])
        elif np.issubdtype(np.dtype(p['type']), np.number):
            dtypes[p['id']] = np.dtype(p['type'])
        else:
            raise RuntimeError(f"Can't convert DVID annotation property to neuroglancer precomputed: {p}")

    for r in info['relationships']:
        # We only support relationships with single-valued IDs, not lists of IDs, and no nulls.
        dtypes[r['id']] = np.uint64

    partner_df = partner_df.astype(dtypes)

    return partner_df

