import os
import copy
import logging

import numpy as np

from flask import Response, request, make_response
from requests import RequestException, HTTPError

from neuclease import configure_default_logging
from neuclease.util import Timer
from neuclease.dvid import (default_dvid_session, find_master, fetch_volume_box, fetch_labelmap_voxels,
                            fetch_sparsevol_coarse, fetch_sparsevol, post_key)
from neuclease.dvid.rle import rle_ranges_box

from vol2mesh import Mesh

logger = logging.getLogger(__name__)
configure_default_logging()

MB = 2**20
MAX_BOX_VOLUME = 128*MB

# FIXME: For now, this resolution is hard-coded.
VOXEL_NM = 8.0

# FIXME: This shouldn't be hard-coded, either
MAX_SCALE = 7

# TODO: Should this function check DVID to see if a mesh already exists
#       for the requested body, or should we assume the caller doesn't
#       want that one?

def generate_and_store_mesh():
    """
    Generate a mesh (in ngmesh format) for a single body,
    upload it to a dvid keyvalue instance, and also return it.

    Pass DVID Authorization token via the 'Authorization' header,
    which will be forwarded to dvid requests.

    All other parameters should be passed via the query string.

    Parameters
    ----------
    dvid:
        dvid server. Required.

    uuid:
        Dvid uuid to read from and write to.

    segmentation:
        Name of the labelmap segmentation instance to read from.
        Default: 'segmentation'

    body:
        Body ID for which a mesh will be generated. Required.

    mesh_kv:
        Which keyvalue instance to store the mesh into.
        Default: 'segmentation_meshes'

    scale:
        Which scale of segmentation data to generate the mesh from.
        If not provided, scale-1 will be used if it won't require too much RAM,
        otherwise a higher scale is automatically chosen.
        Note:
            If scale > 2, the mesh will NOT be stored to DVID, to avoid storing
            low-quality meshes permanently.

    smoothing:
        How many rounds of laplacian smoothing to apply to the marching cubes result.
        Default: 2

    decimation:
        The effective decimation fraction to use.
        For example, 0.1 means "decimate until only 10% of the vertices remain".
        Set this value assuming your mesh will be generated from scale-1 data.
        If scale > 1 is used, then this number will be automatically adjusted
        accordingly.
        Default: 0.1

    user:
        The user name associated with this request.
        Will be forwarded to dvid requests.

    Returns
    -------
    The contents of the generated ngmesh.

    Side Effects
    ------------
    Stores the generated ngmesh into the instance specified
    keyvalue instance before returning it.
    """
    try:
        body = request.args['body']
    except KeyError as ex:
        return Response(f"Missing required parameter: {ex.args[0]}", 400)

    with Timer(f"Body {body}: Handling request", logger):
        try:
            return _generate_and_store_mesh()
        except RequestException as ex:
            return Response(f"Error encountered while accessing dvid server:\n{ex}", 500, content_type='text/plain')


def _generate_and_store_mesh():
    try:
        dvid = request.args['dvid']
        body = request.args['body']
    except KeyError as ex:
        return Response(f"Missing required parameter: {ex.args[0]}", 400)

    segmentation = request.args.get('segmentation', 'segmentation')
    mesh_kv = request.args.get('mesh_kv', f'{segmentation}_meshes')

    uuid = request.args.get('uuid') or find_master(dvid)
    if not uuid:
        uuid = find_master(dvid)

    scale = request.args.get('scale')
    if scale is not None:
        scale = int(scale)

    smoothing = int(request.args.get('smoothing', 2))

    # Note: This is just the effective desired decimation assuming scale-1 data.
    # If we're forced to select a higher scale than scale-1, then we'll increase
    # this number to compensate.
    decimation = float(request.args.get('decimation', 0.1))

    user = request.args.get('u')
    user = user or request.args.get('user', "UNKNOWN")

    # TODO: The global cache of DVID sessions should store authentication info
    #       and use it as part of the key lookup, to avoid creating a new dvid
    #       session for every single cloud call!
    dvid_session = default_dvid_session('cloud-meshgen', user)
    auth = request.headers.get('Authorization')
    if auth:
        dvid_session = copy.deepcopy(dvid_session)
        dvid_session.headers['Authorization'] = auth

    with Timer(f"Body {body}: Fetching coarse sparsevol"):
        svc_ranges = fetch_sparsevol_coarse(dvid, uuid, segmentation, body, format='ranges', session=dvid_session)

    #svc_mask, _svc_box = fetch_sparsevol_coarse(dvid, uuid, segmentation, body, format='mask', session=dvid_session)
    #np.save(f'mask-{body}-svc.npy', svc_mask)

    box_s6 = rle_ranges_box(svc_ranges)
    box_s0 = box_s6*(2**6)
    logger.info(f"Body {body}: Bounding box: {box_s0[:, ::-1].tolist()}")

    if scale is None:
        # Use scale 1 if possible or a higher scale
        # if necessary due to bounding-box RAM usage.
        scale = max(1, select_scale(box_s0))

    if scale > 1:
        # If we chose a low-res scale, then we
        # can reduce the decimation as needed.
        decimation = min(1.0, decimation * 4**(scale-1))

    with Timer(f"Body {body}: Fetching scale-{scale} sparsevol"):
        mask, mask_box = fetch_sparsevol(dvid, uuid, segmentation, body, scale=scale, format='mask', session=dvid_session)
        # np.save(f'mask-{body}-s{scale}.npy', mask)

        # Pad with a thin halo of zeros to avoid holes in the mesh at the box boundary
        mask = np.pad(mask, 1)
        mask_box += [(-1, -1, -1), (1, 1, 1)]

    with Timer(f"Body {body}: Computing mesh"):
        # The 'ilastik' marching cubes implementation supports smoothing during mesh construction.
        mesh = Mesh.from_binary_vol(mask, mask_box * VOXEL_NM * (2**scale), smoothing_rounds=smoothing)

        logger.info(f"Body {body}: Decimating mesh at fraction {decimation}")
        mesh.simplify(decimation)

        logger.info(f"Body {body}: Preparing ngmesh")
        mesh_bytes = mesh.serialize(fmt='ngmesh')

    if scale > 2:
        logger.info(f"Body {body}: Not storing to dvid (scale > 2)")
    else:
        with Timer(f"Body {body}: Storing {body}.ngmesh in DVID ({len(mesh_bytes)/MB:.1f} MB)"):
            try:
                post_key(dvid, uuid, mesh_kv, f"{body}.ngmesh", mesh_bytes, session=dvid_session)
            except HTTPError as ex:
                err = ex.response.content.decode('utf-8')
                if 'locked node' in err:
                    logger.info("Body {body}: Not storing to dvid (uuid {uuid[:4]} is locked).")
                else:
                    logger.warning("Mesh could not be cached to dvid:\n{err}")

    r = make_response(mesh_bytes)
    r.headers.set('Content-Type', 'application/octet-stream')
    return r


def select_scale(box):
    scale = 0
    box = np.array(box)
    while np.prod(box[1] - box[0]) > MAX_BOX_VOLUME:
        scale += 1
        box //= 2

    if scale > MAX_SCALE:
        return Response("Can't generate mesh for body {body}: "
                        "The bounding box would be too large, even at scale {MAX_SCALE}",
                        500)

    return scale
