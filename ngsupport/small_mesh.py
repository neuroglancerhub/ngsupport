import os
import copy
import logging
from functools import partial
from operator import le
from turtle import down

import numpy as np

from flask import Response, request, make_response
from requests import RequestException, HTTPError

from neuclease import configure_default_logging
from neuclease.util import Timer, compute_parallel, downsample_mask
from neuclease.dvid import (default_dvid_session, find_master, fetch_sparsevol_coarse,
                            fetch_sparsevol, post_key, fetch_commit)
from neuclease.dvid.rle import blockwise_masks_from_ranges

from vol2mesh import Mesh

logger = logging.getLogger(__name__)
configure_default_logging()

MB = 2**20

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

    supervoxels:
        If 'true', interpret the 'body' as a supervoxel ID, and generate a supervoxel mesh.

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
        Default: 0.05

    max_vertices:
        Decimate the mesh further until it has this many vertices or fewer.

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
    supervoxels = request.args.get('supervoxels', 'false')
    if supervoxels.lower() not in ('false', 'true'):
        return Response(f"Invalid value for 'supervoxels' parameter: {supervoxels}", 400)
    supervoxels = (supervoxels.lower() == 'true')

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
    decimation = float(request.args.get('decimation', 0.05))

    max_vertices = float(request.args.get('max_vertices', 200e3))

    user = request.args.get('u')
    user = user or request.args.get('user', "UNKNOWN")

    auth = request.headers.get('Authorization')

    # TODO: The global cache of DVID sessions should store authentication info
    #       and use it as part of the key lookup, to avoid creating a new dvid
    #       session for every single cloud call!
    dvid_session = default_dvid_session('cloud-meshgen', user)
    if auth:
        dvid_session = copy.deepcopy(dvid_session)
        dvid_session.headers['Authorization'] = auth

    if supervoxels:
        log_prefix = f"SV {body}"
    else:
        log_prefix = f"Body {body}"

    processes = os.environ.get('NGSUPPORT_PROCESS_POOL', 4)

    mesh, scale = generate_small_mesh(dvid, uuid, segmentation, body, scale, supervoxels,
                                      smoothing, decimation, max_vertices, processes, dvid_session)

    logger.info(f"{log_prefix}: Preparing ngmesh")
    mesh_bytes = mesh.serialize(fmt='ngmesh')

    store_mesh(dvid, uuid, mesh_kv, mesh_bytes, body, scale, supervoxels, dvid_session)

    r = make_response(mesh_bytes)
    r.headers.set('Content-Type', 'application/octet-stream')
    return r


def generate_small_mesh(dvid, uuid, segmentation, body, scale=5, supervoxels=False,
                        smoothing=3, decimation=0.1, max_vertices=200e3, processes=4,
                        dvid_session=None):
    if supervoxels:
        log_prefix = f"SV {body}"
    else:
        log_prefix = f"Body {body}"

    with Timer(f"{log_prefix}: Fetching coarse sparsevol", logger):
        svc_ranges = fetch_sparsevol_coarse(dvid, uuid, segmentation, body, supervoxels=supervoxels, format='ranges', session=dvid_session)

    if scale is not None:
        try:
            rng = fetch_sparsevol(dvid, uuid, segmentation, body, scale=scale, supervoxels=supervoxels, format='ranges')
        except HTTPError as ex:
            if ex.response.status_code == 404:
                logger.error(f"{log_prefix}: Body doesn't exist at the requested scale (scale-{scale}).")
            raise
        block_boxes, masks = blockwise_masks_from_ranges(rng, (64, 64, 64), halo=4)
        block_boxes = block_boxes * VOXEL_NM * (2**scale)

    else:
        # Pick a scale -- aim for 100 blocks
        s0_blocks = (svc_ranges[:, 3] - svc_ranges[:, 2] + 1).sum()
        s6_boxes, _ = blockwise_masks_from_ranges(svc_ranges, (64,64,64))
        logger.info(f"{log_prefix}: Coarse sparsevol covers {s0_blocks} blocks at scale-0, {len(s6_boxes)} blocks at scale-6")
        if len(s6_boxes) > 100:
            logger.info(f"{log_prefix}: Using coarse sparsevol (scale-6)")
            rng = svc_ranges
            scale = 6
        else:
            # Try scales from 3 to scale 1
            for scale in range(3, 0, -1):
                with Timer(f"{log_prefix}: Fetching scale-{scale} sparsevol", logger):
                    try:
                        rng = fetch_sparsevol(dvid, uuid, segmentation, body, scale=scale, supervoxels=supervoxels, format='ranges')
                    except HTTPError as ex:
                        # If the body is too small to exist at this scale, DVID returns 404.
                        # See https://github.com/janelia-flyem/dvid/issues/369
                        if ex.response.status_code == 404:
                            continue
                if len(rng) == 0:
                    continue

                block_boxes, masks = blockwise_masks_from_ranges(rng, (64, 64, 64), halo=4)
                block_boxes = block_boxes * VOXEL_NM * (2**scale)
                if len(block_boxes) > 50:
                    break

    logger.info(f"{log_prefix}: Selected scale-{scale} ({len(block_boxes)} blocks)")

    downsample_scale = 0
    while len(block_boxes) > 200:
        downsample_scale += 1
        # Too many blocks. Reduce scale with continuity-preserving downsampling
        block_boxes, masks = blockwise_masks_from_ranges(rng, 64 * (2**downsample_scale), halo=4)
        block_boxes = block_boxes * VOXEL_NM * (2**scale)
        logger.info(f"{log_prefix}: Downsampling further to scale-{scale+downsample_scale} ({len(block_boxes)} blocks)")
        masks = map(partial(downsample_mask, factor=2), masks)

    if scale + downsample_scale > 1:
        # If we chose a low-res scale, then we
        # can reduce the decimation as needed.
        decimation = min(1.0, decimation * 4**(scale + downsample_scale - 1))

    logger.info(f"{log_prefix}: Constructing mesh")
    mesh = mesh_from_binary_blocks(masks, block_boxes, stitch=False,
                                   presmoothing=smoothing, predecimation=decimation, processes=processes)

    decimation = min(1.0, max_vertices / len(mesh.vertices_zyx))
    if decimation < 1.0:
        logger.info(f"{log_prefix}: Default mesh has too many vertices ({len(mesh.vertices_zyx):.2e} > {max_vertices:.2e})")
        logger.info(f"{log_prefix}: Applying additional decimation to mesh with fraction {decimation:.3f}")
        mesh.simplify(decimation)

    return mesh, scale + downsample_scale


def mesh_from_binary_blocks(downsampled_binary_blocks, fullres_boxes_zyx, stitch=True, presmoothing=0, predecimation=1.0, processes=4):
    """
    Mesh.from_binary_blocks(), but adapted to use multiple processes.
    """
    if stitch and (presmoothing != 0 or predecimation != 1.0):
        msg = ("You're using stitch=True, but you're applying presmoothing or "
               "predecimation, so the block meshes won't stitch properly.")
        logger.warn(msg)
    num_blocks = getattr(fullres_boxes_zyx, '__len__', lambda: None)()
    gen_mesh = partial(_gen_mesh, smoothing=presmoothing, decimation=predecimation)
    meshes = compute_parallel(gen_mesh, zip(downsampled_binary_blocks, fullres_boxes_zyx), starmap=True, total=num_blocks, show_progress=False, processes=4)
    mesh = Mesh.concatenate_meshes(meshes)
    if stitch:
        mesh.stitch_adjacent_faces()
    return mesh


def _gen_mesh(binary_block, fullres_box, smoothing, decimation):
    m = Mesh.from_binary_vol(binary_block, fullres_box, method='ilastik', smoothing_rounds=smoothing)
    m.simplify(decimation)
    return m


def store_mesh(dvid, uuid, mesh_kv, mesh_bytes, body, scale, supervoxels, dvid_session):
    if supervoxels:
        log_prefix = f"SV {body}"
    else:
        log_prefix = f"Body {body}"

    if supervoxels:
        logger.info(f"{log_prefix}: Not storing supervoxel mesh to dvid")
        return False

    if scale and scale > 3:
        logger.info(f"{log_prefix}: Not storing to dvid (scale > 3)")
        return False

    if fetch_commit(dvid, uuid):
        logger.info(f"{log_prefix}: Not storing to dvid (uuid {uuid[:4]} is locked).")
        return False

    try:
        with Timer(f"{log_prefix}: Storing {body}.ngmesh in DVID ({len(mesh_bytes)/MB:.1f} MB)", logger):
            post_key(dvid, uuid, mesh_kv, f"{body}.ngmesh", mesh_bytes, session=dvid_session)
    except HTTPError as ex:
        err = ex.response.content.decode('utf-8')
        logger.warning(f"Mesh could not be cached to dvid:\n{err}")
        return False
    else:
        return True
