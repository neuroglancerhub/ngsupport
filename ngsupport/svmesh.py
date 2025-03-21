import functools
from flask import make_response, jsonify
from neuclease.dvid import fetch_info, fetch_tarfile, fetch_supervoxel
from vol2mesh import Mesh


def svmesh_info():
    return jsonify({"@type": "neuroglancer_legacy_mesh"})


def svmesh_manifest(segment_id):
    return jsonify({"fragments": [f"meshes/{segment_id}.ngmesh"]})


def svmesh_ngmesh(server, uuid, instance, segment_type, segment_id):
    """
    Generate a mesh (in ngmesh format) for a supervoxel or body.

    If a supervoxel mesh is requested, we fetch it directly from the
    given tarsupervoxels instance.

    If a body mesh is requested, we fetch the entire tarfile of
    supervoxel meshes from the given tarsupervoxels instance and
    construct a single ngmesh from their union.

    Parameters
    ----------
    dvid:
        dvid server. Required.

    uuid:
        Dvid uuid to read from and write to.

    instance:
        Name of the tarsupervoxels instance to read from.
        If the instance ends with the suffix '?supervoxels=true',
        then we'll fetch a supervoxel mesh instead of a body mesh.

    segment_type:
        Either 'body' or 'supervoxels' (supervoxel).

    segment_id:
        Either a body ID or supervoxel ID, depending on whether the instance ends with '?supervoxels=true'.

    Returns
    -------
    The contents of the generated ngmesh.
    """
    if segment_type in ('sv', 'supervoxel', 'supervoxels'):
        try:
            drc_bytes = fetch_supervoxel(server, uuid, instance, segment_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 404
        mesh = Mesh.from_buffer(drc_bytes, fmt='drc')
    else:
        try:
            tarfile = fetch_tarfile(server, uuid, instance, segment_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 404
        mesh = Mesh.from_tarfile(tarfile)

    if len(mesh.vertices_zyx) == 0:
        return jsonify({"error": "No vertices found in tarfile"}), 404

    mesh.vertices_zyx *= get_dataset_resolution_zyx(server, uuid, instance)
    ngmesh_bytes = mesh.serialize(fmt='ngmesh')
    r = make_response(ngmesh_bytes)
    r.headers.set('Content-Type', 'application/octet-stream')
    return r


@functools.cache
def get_dataset_resolution_zyx(server, uuid, tsv_instance):
    tsv_info = fetch_info(server, uuid, tsv_instance)
    seg_instance = tsv_info['Base']['Syncs'][0]
    seg_info = fetch_info(server, uuid, seg_instance)
    return seg_info['Extended']['VoxelSize'][::-1]
