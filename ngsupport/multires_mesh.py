"""
On-demand multi-resolution meshes for neuroglancer, backed by DVID's mesh store.

These endpoints present neuroglancer with an *unsharded* precomputed mesh source
(``"@type": "neuroglancer_multilod_draco"``, i.e. draco-compressed fragments).
Body meshes are produced and stored by ``neuclease.misc.bodymesh.update_body_mesh()``,
exactly as the flyemflows 'chunkedbodymeshes' workflow does:

- When a body's manifest is requested, the body mesh is brought up to date first.
  Chunk meshes which are missing or out-of-date (relative to the body's current
  voxels) are generated and stored in DVID's '<segmentation>_multires_chunk_meshes'
  instance, and the body mesh is reassembled from them and stored in
  '<segmentation>_multires_meshes'.  If the stored body mesh is already up-to-date,
  nothing is regenerated.
- The manifest and fragment data are then served from '<segmentation>_multires_meshes'.

The DVID mesh instances must already exist (see bodymesh.init_mesh_instances(..., multires=True)),
and the service must be able to write to them (see DVID_ADMIN_TOKEN, below).

To use these meshes in neuroglancer, add a source like this to a segmentation layer:

    precomputed://https://<ngsupport-host>/multires-mesh/<dvid-server>/<uuid>/<segmentation>/<config-name>

for example:

    precomputed://https://ngsupport-bmcp5imp6q-uk.a.run.app/multires-mesh/emdata-colnora.janelia.org/f3969:master/segmentation/512_ddeda

Relative to that source URL, the endpoints are:

    info            The mesh source metadata.
    <body>.index    The body's manifest.  Updates the body's stored mesh first, if necessary.
    <body>          The body's fragment data, fetched via HTTP Range requests.

The <uuid> may be any DVID ref accepted by neuclease's resolve_ref(), e.g. 'f3969:master'.

Configs
-------
<config-name> names a YAML (or JSON) file in the directory given by the
NGSUPPORT_MULTIRES_MESH_CONFIG_DIR environment variable (by default, the
'multires_mesh_configs' directory next to this module).  The file may be a complete
flyemflows 'chunkedbodymeshes' workflow config, or just its two relevant sections:

    body-meshes: {...}   # See neuclease.misc.bodymesh.BodyMeshParametersSchema
    chunk-meshes: {...}  # See neuclease.misc.bodymesh.MeshChunkConfigSchema

The body-meshes 'format' must be 'neuroglancer-multi-res', and its 'source-method'
must be chunk-based ('concatenated-chunks-...').

Fragment data
-------------
By default, fragment requests are proxied to DVID (forwarding the Range header).
Alternatively, set NGSUPPORT_MULTIRES_REDIRECT_FRAGMENTS=1 to redirect neuroglancer
to fetch fragments directly from DVID, which avoids sending the mesh data through this
service.  That requires DVID to be readable by the client without credentials, to
support Range requests on keyvalue GETs, and to send CORS headers.

Other settings
--------------
- Bringing a large body's mesh up to date can take a while; make sure the service's
  request timeout allows for it.  Chunk meshes are generated in parallel using a pool
  of NGSUPPORT_PROCESS_POOL processes (default: 4).
- DVID access uses neuclease's default sessions, which pass DVID_ADMIN_TOKEN (if set)
  to DVID.  That is needed to store meshes in a locked (committed) DVID node.
- If the body mesh can't be updated (e.g. the node is locked and no token is set),
  the stored mesh is served anyway, if there is one.
- Concurrent manifest requests for the same body are serialized on each host.
  Requests handled by different hosts may update the same body at the same time;
  to limit the redundant work, each one generates the body's missing chunks in random
  order, skips any chunk that another has stored in the meantime, and skips storing
  the body mesh if another has already stored an up-to-date one.
"""
import os
import re
import copy
import json
import time
import fcntl
import hashlib
import logging
import tempfile
import functools
import threading
from collections import namedtuple
from contextlib import contextmanager
from http import HTTPStatus

from flask import Response, jsonify, redirect, request
from requests import HTTPError, RequestException

from .util import normalize_server

logger = logging.getLogger(__name__)

CONFIG_DIR = os.environ.get(
    'NGSUPPORT_MULTIRES_MESH_CONFIG_DIR',
    os.path.join(os.path.dirname(__file__), 'multires_mesh_configs')
)
LOCK_DIR = os.environ.get(
    'NGSUPPORT_MULTIRES_LOCK_DIR',
    os.path.join(tempfile.gettempdir(), 'ngsupport-multires-mesh-locks')
)
PROCESSES = int(os.environ.get('NGSUPPORT_PROCESS_POOL', 4))
REDIRECT_FRAGMENTS = os.environ.get('NGSUPPORT_MULTIRES_REDIRECT_FRAGMENTS', '').lower() in ('1', 'true', 'yes')

# How long a resolved UUID can be reused by fragment requests (see _resolve_uuid()).
UUID_CACHE_SECONDS = 60

# Bodies are locked via a fixed set of lock files (one per stripe).
LOCK_STRIPES = 1024

OCTET_STREAM = 'application/octet-stream'

# Mesh payloads must not be cached by the browser, lest a stale manifest
# be combined with fragment data from a newer version of the mesh.
NO_STORE_HEADERS = {
    'Cache-Control': 'no-store',
    'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges',
}


class ConfigError(Exception):
    pass


class NotFound(Exception):
    pass


def _error_responses(f):
    """
    Convert the errors we expect into HTTP responses that neuroglancer handles
    sensibly: "not found" errors become 404 (neuroglancer then shows no mesh);
    other DVID problems become 502.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except (ConfigError, NotFound) as ex:
            return Response(str(ex), HTTPStatus.NOT_FOUND, mimetype='text/plain')
        except HTTPError as ex:
            # DVID returns 400 for an unknown instance and 404 for an unknown body.
            status = ex.response.status_code if ex.response is not None else None
            if status in (HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND):
                return Response(f"Not found in DVID: {ex}", HTTPStatus.NOT_FOUND, mimetype='text/plain')
            return Response(f"Error from DVID: {ex}", HTTPStatus.BAD_GATEWAY, mimetype='text/plain')
        except RequestException as ex:
            return Response(f"Error accessing DVID: {ex}", HTTPStatus.BAD_GATEWAY, mimetype='text/plain')
    return wrapper


@_error_responses
@normalize_server
def multires_mesh_info(server, uuid, seg_instance, config_name):
    """
    Return the 'info' metadata for the multires mesh source.

    (This matches the 'info' that update_body_mesh() stores in the mesh instance,
    but it's computed here so that it's available before any body mesh is stored.)
    """
    from vol2mesh.multires import build_info

    config = _load_config(config_name)
    vx, vy, vz = _voxel_size_xyz(server, uuid, seg_instance)
    info = build_info(
        vertex_quantization_bits=config.body['vertex-quantization-bits'],
        # Stored-model coordinates are full-resolution voxels; the model space is nanometers.
        transform=[vx, 0, 0, 0,
                   0, vy, 0, 0,
                   0, 0, vz, 0],
    )
    return jsonify(info)


@_error_responses
@normalize_server
def multires_mesh_index(server, uuid, seg_instance, config_name, body):
    """
    Return a body's multires manifest (the '<body>.index' file),
    after bringing the body's stored mesh up to date.
    """
    from neuclease.misc.bodymesh import update_body_mesh, mesh_instance_names

    config = _load_config(config_name)
    uuid = _resolve_uuid(server, uuid, fresh=True)
    mesh_instance = mesh_instance_names(seg_instance, multires=True).body

    update_error = None
    with _body_lock(server, uuid, seg_instance, body):
        try:
            # This is a no-op if the stored mesh is already up-to-date.
            # Other service instances may be updating the same body at the same time.
            update_body_mesh(server, uuid, seg_instance, body,
                             copy.deepcopy(config.body), copy.deepcopy(config.chunk),
                             processes=PROCESSES, expect_concurrent_updates=True)
        except RequestException as ex:
            # Fall back to whatever mesh is already stored (if any).
            logger.warning(f"Body {body}: Could not update the stored mesh: {ex}")
            update_error = ex

    index = _fetch_value(server, uuid, mesh_instance, f'{body}.index')
    if index is None and update_error is not None:
        raise update_error
    if not index:
        raise NotFound(f"Body {body} has no mesh geometry")
    return Response(index, HTTPStatus.OK, mimetype=OCTET_STREAM, headers=NO_STORE_HEADERS)


@_error_responses
@normalize_server
def multires_mesh_data(server, uuid, seg_instance, config_name, body):
    """
    Return (a byte range of) a body's multires fragment data (the '<body>' file),
    either by proxying the request to DVID or redirecting the client to DVID.
    """
    from neuclease.misc.bodymesh import mesh_instance_names

    _load_config(config_name)
    uuid = _resolve_uuid(server, uuid)
    mesh_instance = mesh_instance_names(seg_instance, multires=True).body

    if REDIRECT_FRAGMENTS:
        # Note: The redirect URL must not include DVID_ADMIN_TOKEN.
        url = f'{server}/api/node/{uuid}/{mesh_instance}/key/{body}'
        r = redirect(url, HTTPStatus.TEMPORARY_REDIRECT)
        r.headers.update(NO_STORE_HEADERS)
        return r

    range_header = request.headers.get('Range')
    result = _fetch_value_range(server, uuid, mesh_instance, f'{body}', range_header)
    if result is None or (result.status == HTTPStatus.OK and not result.content):
        raise NotFound(f"Body {body} has no stored mesh")

    headers = {**NO_STORE_HEADERS, 'Accept-Ranges': 'bytes'}
    if result.status == HTTPStatus.PARTIAL_CONTENT:
        headers['Content-Range'] = result.content_range
        return Response(result.content, HTTPStatus.PARTIAL_CONTENT, mimetype=OCTET_STREAM, headers=headers)
    if result.status == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
        if result.content_range:
            headers['Content-Range'] = result.content_range
        return Response(status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, headers=headers)

    # DVID sent the whole value.  (Older DVID versions ignore the Range header.)
    return _byte_range_response(result.content)


def _byte_range_response(data):
    """Respond with the byte range of data requested by the client's Range header, if any."""
    headers = {**NO_STORE_HEADERS, 'Accept-Ranges': 'bytes'}
    rng = request.range
    if rng is None or len(rng.ranges) != 1:
        # No (usable) Range header: send the whole thing.
        return Response(data, HTTPStatus.OK, mimetype=OCTET_STREAM, headers=headers)

    bounds = rng.range_for_length(len(data))
    if bounds is None:
        headers['Content-Range'] = f'bytes */{len(data)}'
        return Response(status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, headers=headers)

    start, stop = bounds
    headers['Content-Range'] = f'bytes {start}-{stop - 1}/{len(data)}'
    return Response(data[start:stop], HTTPStatus.PARTIAL_CONTENT, mimetype=OCTET_STREAM, headers=headers)


@contextmanager
def _body_lock(server, uuid, seg_instance, body):
    """
    Serialize mesh updates for a body among all threads and processes on this host,
    so that concurrent manifest requests for the same body don't duplicate the work.
    (Requests handled by different hosts may still update the same body concurrently;
    see update_body_mesh(..., expect_concurrent_updates=True).)
    """
    key = json.dumps([server, uuid, seg_instance, int(body)])
    stripe = int(hashlib.sha256(key.encode()).hexdigest(), 16) % LOCK_STRIPES
    os.makedirs(LOCK_DIR, exist_ok=True)
    with open(os.path.join(LOCK_DIR, f'{stripe}.lock'), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


#
# Configs
#

MeshConfig = namedtuple('MeshConfig', 'body chunk')

# Config names become file names, so don't permit anything path-like.
_CONFIG_NAME_PATTERN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')


def _load_config(config_name):
    if not _CONFIG_NAME_PATTERN.fullmatch(config_name) or '..' in config_name:
        raise ConfigError(f"Invalid multires mesh config name: {config_name!r}")

    for ext in ('.yaml', '.yml', '.json'):
        path = os.path.join(CONFIG_DIR, config_name + ext)
        if os.path.isfile(path):
            # Including the mtime in the cache key means edited configs are reloaded.
            return _load_config_file(path, os.path.getmtime(path))

    raise ConfigError(f"Unknown multires mesh config: {config_name!r}")


@functools.lru_cache(maxsize=32)
def _load_config_file(path, _mtime):
    from confiddler import validate
    from jsonschema import ValidationError
    from ruamel.yaml import YAML
    from neuclease.misc.bodymesh import (
        BodyMeshParametersSchema, MeshChunkConfigSchema, quality_from_source_method)

    # Parse with YAML 1.2 (via ruamel), as flyemflows does.
    # (With PyYAML's YAML 1.1, values like '10e6' would be parsed as strings.)
    with open(path, 'r') as f:
        cfg = YAML(typ='safe').load(f) or {}

    # Accept a full flyemflows 'chunkedbodymeshes' workflow config, too.
    cfg = cfg.get('chunkedbodymeshes', cfg)
    try:
        body_cfg = cfg['body-meshes']
        chunk_cfg = cfg['chunk-meshes']
    except KeyError as ex:
        raise ConfigError(f"Config {path} has no '{ex.args[0]}' section") from ex

    try:
        validate(body_cfg, BodyMeshParametersSchema, inject_defaults=True)
        validate(chunk_cfg, MeshChunkConfigSchema, inject_defaults=True)
        quality_from_source_method(body_cfg, chunk_cfg)
    except (ValidationError, RuntimeError) as ex:
        raise ConfigError(f"Invalid multires mesh config {path}: {ex}") from ex

    if body_cfg['format'] != 'neuroglancer-multi-res':
        raise ConfigError(f"Config {path} must specify format: neuroglancer-multi-res")

    return MeshConfig(body_cfg, chunk_cfg)


#
# DVID helpers
#

_resolved_uuids = {}
_resolved_uuids_lock = threading.Lock()


def _resolve_uuid(server, uuid, fresh=False):
    """
    Resolve a DVID ref (e.g. 'abc123:master') to a full UUID.

    Manifest requests resolve the ref afresh (fresh=True), since the stored mesh
    must be updated in the branch's current node.  Fragment requests (which are
    numerous) reuse a recent result, which also makes them likely to read from the
    same node that the manifest request used.
    """
    from neuclease.dvid import resolve_ref

    now = time.time()
    if not fresh:
        with _resolved_uuids_lock:
            cached = _resolved_uuids.get((server, uuid))
        if cached and now - cached[0] < UUID_CACHE_SECONDS:
            return cached[1]

    try:
        resolved = resolve_ref(server, uuid, expand=True)
    except RuntimeError as ex:
        raise NotFound(str(ex)) from ex

    with _resolved_uuids_lock:
        _resolved_uuids[(server, uuid)] = (now, resolved)
    return resolved


@functools.cache
def _voxel_size_xyz(server, uuid, seg_instance):
    # Voxel size never changes, so it's safe to cache this, even if uuid is a branch ref.
    from neuclease.dvid import fetch_instance_info
    info = fetch_instance_info(server, _resolve_uuid(server, uuid), seg_instance)
    return tuple(info['Extended']['VoxelSize'])


def _fetch_value(server, uuid, instance, key):
    """Fetch a keyvalue value from DVID, or return None if it doesn't exist."""
    from neuclease.dvid.keyvalue import fetch_key
    try:
        return fetch_key(server, uuid, instance, key)
    except HTTPError as ex:
        if ex.response is not None and ex.response.status_code == HTTPStatus.NOT_FOUND:
            return None
        raise


RangeResult = namedtuple('RangeResult', 'status content_range content')


def _fetch_value_range(server, uuid, instance, key, range_header=None):
    """
    Fetch (a byte range of) a keyvalue value from DVID, or return None if it doesn't exist.
    Returns a RangeResult, whose status is 200, 206, or 416.
    """
    from neuclease.dvid import default_dvid_session

    headers = {'Range': range_header} if range_header else {}
    r = default_dvid_session().get(f'{server}/api/node/{uuid}/{instance}/key/{key}', headers=headers)
    if r.status_code == HTTPStatus.NOT_FOUND:
        return None
    if r.status_code != HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
        r.raise_for_status()
    return RangeResult(r.status_code, r.headers.get('Content-Range'), r.content)
