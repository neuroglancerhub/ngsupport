import os
import copy
import logging
import subprocess
from functools import partial

import numpy as np

from flask import Response, request, make_response, jsonify
import requests
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

NGSUPPORT_SERVICE_BASE = os.environ.get('NGSUPPORT_SERVICE_BASE', 'https://ngsupport-bmcp5imp6q-uk.a.run.app')
NGSUPPORT_PARALLELIZE_WITH_SERVICE = int(os.environ.get('NGSUPPORT_PARALLELIZE_WITH_SERVICE', 0))
NGSUPPORT_PROCESS_POOL = int(os.environ.get('NGSUPPORT_PROCESS_POOL', 4))


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
        For example, 0.01 means "decimate until only 1% of the vertices remain".
        Set this value assuming your mesh will be generated from scale-1 data.
        If scale > 1 is used, then this number will be automatically adjusted
        accordingly.

        Default: 0.025 (low quality, but lighter weight for speed and lots of meshes)

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
    decimation = float(request.args.get('decimation', 0.025))

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

    mesh, scale = generate_small_mesh(dvid, uuid, segmentation, body, scale, supervoxels,
                                      smoothing, decimation, max_vertices, dvid_session)

    logger.info(f"{log_prefix}: Preparing ngmesh")
    mesh_bytes = mesh.serialize(fmt='ngmesh')

    store_mesh(dvid, uuid, mesh_kv, mesh_bytes, body, scale, supervoxels, dvid_session)

    r = make_response(mesh_bytes)
    r.headers.set('Content-Type', 'application/octet-stream')
    return r


def generate_small_mesh(dvid, uuid, segmentation, body, scale=5, supervoxels=False,
                        smoothing=3, decimation=0.01, max_vertices=200e3,
                        dvid_session=None):
    if supervoxels:
        log_prefix = f"SV {body}"
    else:
        log_prefix = f"Body {body}"

    with Timer(f"{log_prefix}: Fetching coarse sparsevol", logger, log_start=False):
        svc_ranges = fetch_sparsevol_coarse(dvid, uuid, segmentation, body, supervoxels=supervoxels, format='ranges', session=dvid_session)

    # Do a test mesh on the coarse sparsevol
    logger.info(f"{log_prefix}: Estimating vertex count via coarse scale-6 mesh blocks")
    block_boxes, masks = blockwise_masks_from_ranges(svc_ranges, (64, 64, 64), halo=2)
    block_boxes = block_boxes * VOXEL_NM * (2**6)
    vertices_s6 = mesh_from_binary_blocks(log_prefix, masks, block_boxes, stitch=False,
                                          presmoothing=0, predecimation=1.0,
                                          size_only=True)
    logger.info(f"{log_prefix}: Coarse scale-6 blocks contain {vertices_s6:.2e} vertices")

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
        block_boxes, masks = blockwise_masks_from_ranges(svc_ranges, (64,64,64))
        block_boxes = block_boxes * VOXEL_NM * (2**6)

        logger.info(f"{log_prefix}: Coarse sparsevol covers {s0_blocks} blocks at scale-0, {len(block_boxes)} blocks at scale-6")
        if len(block_boxes) > 100:
            logger.info(f"{log_prefix}: Using coarse sparsevol (scale-6)")
            rng = svc_ranges
            scale = 6
        else:
            # Try scales from 3 to scale 1
            for scale in range(3, 0, -1):
                with Timer(f"{log_prefix}: Fetching scale-{scale} sparsevol", logger, log_start=False):
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
    while len(block_boxes) > 200 and scale != 6:
        # Too many blocks.  Find a better scale.
        # Reduce scale with continuity-preserving downsampling.
        downsample_scale += 1
        factor = (2**downsample_scale)
        block_boxes, masks = blockwise_masks_from_ranges(rng, 64 * factor, halo=2*factor)
        block_boxes = block_boxes * VOXEL_NM * (2**scale)
        logger.info(f"{log_prefix}: Downsampling further to scale-{scale+downsample_scale} ({len(block_boxes)} blocks)")
        masks = map(partial(downsample_mask, factor=factor), masks)

    predecimation = decimation
    if scale + downsample_scale > 1:
        # If we chose a low-res scale, then we
        # can reduce the decimation as needed.
        predecimation = min(1.0, decimation * 4**(scale + downsample_scale - 1))
        logger.info(f"{log_prefix}: Requested decimation (after adjusting for scale) is {predecimation}")

    # Further enhance the decimation if we think the vertex count will be too high with the default decimation.
    # We estimate the full vertex count by multiplying the scale-6 vertex count.
    estimated_full_vertices = vertices_s6 * 4**(6 - (scale + downsample_scale))
    estimated_decimated_vertices = predecimation * estimated_full_vertices
    logger.info(f"{log_prefix}: Estimated final vertex count is {estimated_decimated_vertices:.2e}.")
    if estimated_decimated_vertices > max_vertices:
        predecimation = max_vertices/estimated_full_vertices
        logger.info(f"{log_prefix}: Enhancing decimation to {predecimation:.5f} to fit within vertex budget ({max_vertices:.2e})")

    mesh = mesh_from_binary_blocks(log_prefix, masks, block_boxes, stitch=False,
                                   presmoothing=smoothing, predecimation=predecimation)

    postdecimation = min(1.0, max_vertices / len(mesh.vertices_zyx))
    if postdecimation < 0.9:
        logger.info(f"{log_prefix}: Computed mesh has too many vertices ({len(mesh.vertices_zyx):.2e} > {max_vertices:.2e})")
        with Timer(f"{log_prefix}: Applying additional decimation to mesh with fraction {postdecimation:.3f}", log_start=False):
            try:
                mesh.simplify(postdecimation)
            except subprocess.CalledProcessError as ex:
                # Sometimes the simplification fails for harmless reasons
                # (e.g. it can't find a way to achieve 0.999 decimation)
                logger.warn(f"Could not simplify mesh ({len(mesh.vertices_zyx)}) with decimation {predecimation}")
                logger.warn(f"{ex}")

    logger.info(f"{log_prefix}: Final mesh has {len(mesh.vertices_zyx):.2e} vertices")
    return mesh, scale + downsample_scale


def mesh_from_binary_blocks(log_prefix, downsampled_binary_blocks, fullres_boxes_zyx, stitch=True, presmoothing=0, predecimation=1.0, size_only=False):
    """
    Mesh.from_binary_blocks(), but adapted to use multiple processes.
    """
    if stitch and (presmoothing != 0 or predecimation != 1.0):
        msg = ("You're using stitch=True, but you're applying presmoothing or "
               "predecimation, so the block meshes won't stitch properly.")
        logger.warn(msg)

    num_blocks = getattr(fullres_boxes_zyx, '__len__', lambda: None)()
    if NGSUPPORT_PARALLELIZE_WITH_SERVICE:
        threads = NGSUPPORT_PARALLELIZE_WITH_SERVICE
        processes = 0
        msg = f"{log_prefix}: Computing {num_blocks} block meshes via {threads} service workers"
        gen_mesh = partial(_request_block_mesh, presmoothing=presmoothing, predecimation=predecimation, size_only=size_only)
    else:
        threads = 0
        processes = NGSUPPORT_PROCESS_POOL
        msg = f"{log_prefix}: Computing {num_blocks} block meshes via {processes} local processes"
        gen_mesh = partial(_gen_block_mesh, presmoothing=presmoothing, predecimation=predecimation, size_only=size_only)

    with Timer(msg, log_start=False):
        results = compute_parallel(gen_mesh, zip(downsampled_binary_blocks, fullres_boxes_zyx), starmap=True,
                                   total=num_blocks, show_progress=False,
                                   threads=threads, processes=processes)

    if size_only:
        return sum(results)

    mesh = Mesh.concatenate_meshes(results)
    if stitch:
        mesh.stitch_adjacent_faces()
    return mesh


def _gen_block_mesh(binary_block, fullres_box, presmoothing, predecimation, size_only=False):
    m = Mesh.from_binary_vol(binary_block, fullres_box, method='ilastik', smoothing_rounds=presmoothing)
    try:
        m.simplify(predecimation)
    except subprocess.CalledProcessError as ex:
        # Sometimes the simplification fails for harmless reasons
        # (e.g. it can't find a way to achieve 0.999 decimation)
        logger.warn(f"Could not simplify mesh ({len(m.vertices_zyx)}) with decimation {predecimation}")
        logger.warn(f"{ex}")
    if size_only:
        return len(m.vertices_zyx)
    return m


def _request_block_mesh(binary_block, fullres_box_zyx, presmoothing, predecimation, size_only=False):
    """
    Instead of running _gen_block_mesh() directly, request it from the service.
    This allows us to parallelize the computation of the block meshes beyond
    a single service instance, possibly achieving parallelism of 1000x or better.

    HOWEVER, this comes with a special risk: Deadlock.
    Since we're now making the outer (overall) request dependent on smaller requests
    WHICH WILL RUN ON THIS SAME SERVICE, it is critical that the service hasn't
    saturated all worker threads it has been provisioned with.  For that reason,
    running this service on a local machine with limited threads is not recommended
    without careful thought to the configuration.  Running with a highly-scaled
    CloudRun configuration will work, but even there the particular configuration
    (number of instances, number of workers per instance, etc.) is very important.

    The above-mentioned concern could be largely alleviated if we switch to an async
    implementation of our route handlers and the request in this function.
    """
    if not NGSUPPORT_SERVICE_BASE:
        raise RuntimeError("NGSUPPORT_SERVICE_BASE is not defined")
    url = f'{NGSUPPORT_SERVICE_BASE}/block-mesh'
    params = {
        'shape': '_'.join(map(str, binary_block.shape)),
        'box0': '_'.join(map(str, fullres_box_zyx[0])),
        'box1': '_'.join(map(str, fullres_box_zyx[1])),
        'presmoothing': str(presmoothing),
        'predecimation': str(predecimation),
        'size_only': str(size_only)
    }
    payload = bytes(np.packbits(binary_block.view(bool)))
    r = requests.post(url, params=params, data=payload)
    r.raise_for_status()

    if size_only:
        return r.json()[0]

    return Mesh.from_buffer(r.content, fmt='ngmesh')


def gen_block_mesh():
    shape = [*map(int, request.args['shape'].split('_'))]
    box0 = [*map(float, request.args['box0'].split('_'))]
    box1 = [*map(float, request.args['box1'].split('_'))]
    presmoothing = int(request.args['presmoothing'])
    predecimation = float(request.args['predecimation'])
    size_only = request.args.get('size_only', 'False') == 'True'

    packed_bits = np.frombuffer(request.data, dtype=np.uint8)
    binary_block = np.unpackbits(packed_bits).reshape(shape).view(np.uint8)
    m = _gen_block_mesh(binary_block, [box0, box1], presmoothing, predecimation)

    if size_only:
        r = jsonify([len(m.vertices_zyx)])
        return r

    mesh_bytes = m.serialize(fmt='ngmesh')
    r = make_response(mesh_bytes)
    r.headers.set('Content-Type', 'application/octet-stream')
    return r


def store_mesh(dvid, uuid, mesh_kv, mesh_bytes, body, scale, supervoxels, dvid_session):
    if supervoxels:
        log_prefix = f"SV {body}"
    else:
        log_prefix = f"Body {body}"

    if supervoxels:
        logger.info(f"{log_prefix}: Not storing supervoxel mesh to dvid")
        return False

    if scale and scale > 5:
        logger.info(f"{log_prefix}: Not storing to dvid (scale too bad)")
        return False

    if fetch_commit(dvid, uuid):
        if os.environ.get("DVID_ADMIN_TOKEN"):
            logger.info(f"{log_prefix}: Using DVID_ADMIN_TOKEN to store mesh in LOCKED dvid uuid {uuid[:6]}")
        else:
            logger.info(f"{log_prefix}: Not storing to dvid (uuid {uuid[:6]} is locked).")
            return False

    try:
        with Timer(f"{log_prefix}: Storing {body}.ngmesh in DVID ({len(mesh_bytes)/MB:.1f} MB)", logger, log_start=False):
            post_key(dvid, uuid, mesh_kv, f"{body}.ngmesh", mesh_bytes, session=dvid_session)
    except HTTPError as ex:
        err = ex.response.content.decode('utf-8')
        logger.warning(f"Mesh could not be cached to dvid:\n{err}")
        return False
    else:
        return True
