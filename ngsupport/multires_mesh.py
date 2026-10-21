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
- The manifest and fragment data are then served from '<segmentation>_multires_meshes',
  or from an object store (e.g. a Google Cloud Storage bucket), if the config says so
  (see 'storage' in neuclease.misc.bodymesh.BodyMeshParametersSchema).  In that case,
  the chunk meshes may be stored there, too.

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
    <body>/chunk    (POST) Generate and store one chunk mesh.  Used internally (see below).

The <uuid> may be any DVID ref accepted by neuclease's resolve_ref(), e.g. 'f3969:master'.

Configs
-------
<config-name> names a YAML (or JSON) file in the directory given by the
NGSUPPORT_MULTIRES_MESH_CONFIG_DIR environment variable (by default, the
'multires_mesh_configs' directory next to this module).  The directory may also be
a bucket location, such as 'gs://my-bucket/configs', in which case the service needs
read access to it.  Edited configs are reloaded automatically (within
CONFIG_CACHE_SECONDS, for configs in a bucket).  The file may be a complete
flyemflows 'chunkedbodymeshes' workflow config, or just its two relevant sections:

    body-meshes: {...}   # See neuclease.misc.bodymesh.BodyMeshParametersSchema
    chunk-meshes: {...}  # See neuclease.misc.bodymesh.MeshChunkConfigSchema

The body-meshes 'format' must be 'neuroglancer-multi-res', and its 'source-method'
must be chunk-based ('concatenated-chunks-...').

Fragment data
-------------
If the body meshes are stored in a (public) bucket, fragment requests are redirected to it.
(The bucket must send CORS headers; see below.)

Otherwise, fragment requests are proxied to DVID (forwarding the Range header), by default.
Alternatively, set NGSUPPORT_MULTIRES_REDIRECT_FRAGMENTS=1 to redirect neuroglancer
to fetch fragments directly from DVID, which avoids sending the mesh data through this
service.  That requires DVID to be readable by the client without credentials, to
support Range requests on keyvalue GETs, and to send CORS headers.

Chunk workers
-------------
By default, a body's missing chunk meshes are generated in parallel using a pool of
NGSUPPORT_PROCESS_POOL processes (default: 4) within the instance handling the request.
Alternatively, set NGSUPPORT_MULTIRES_CHUNK_WORKERS to N to send them to a separate
deployment of this service (NGSUPPORT_SERVICE_WORKER_BASE, as for /small-mesh),
with up to N concurrent requests per body.  Each request is handled by the
'<body>/chunk' endpoint, which generates and stores one chunk mesh in DVID.

- The worker deployment must be separate from the one which handles neuroglancer's
  requests.  Otherwise, all of the service's request threads could end up waiting for
  chunk requests that can't be handled (a deadlock).
- The worker deployment needs the same configs (it rejects requests from a service whose
  config differs from its own), and a DVID_ADMIN_TOKEN, if one is needed to store meshes.
- Failed chunk requests are retried a few times (e.g. while Cloud Run scales up).

Buckets
-------
For body meshes stored in a Google Cloud Storage bucket, the bucket must be publicly
readable, and it must permit CORS requests from any origin, since neuroglancer is
redirected to it.  (After a cross-origin redirect, the browser sends 'Origin: null'.)
For example (with gcloud storage buckets update gs://<bucket> --cors-file=cors.json):

    [{"origin": ["*"], "method": ["GET", "HEAD"], "maxAgeSeconds": 3600,
      "responseHeader": ["Content-Type", "Content-Range", "Content-Length", "Accept-Ranges"]}]

The service (and its chunk workers) must be able to write to the bucket, via the
service account's credentials.  Consider disabling the bucket's "soft delete" policy,
since overwritten mesh files would otherwise be retained (and billed) for a week.

Other settings
--------------
- Bringing a large body's mesh up to date can take a while; make sure the service's
  request timeout allows for it.
- Assembling a large body's mesh (especially decimating it) requires a lot of memory,
  and the decimation library doesn't release all of it afterwards.  So by default,
  each body mesh is assembled in a separate, short-lived process, which releases all
  of its memory when it exits.  (Set NGSUPPORT_MULTIRES_ASSEMBLE_IN_SUBPROCESS=0 to
  assemble it in the request's process instead.)  The peak memory usage is the same,
  so make sure each instance has enough memory for the largest bodies it may assemble
  concurrently.
- The number of gunicorn worker processes and threads (per instance) can be set via
  GUNICORN_WORKERS and GUNICORN_THREADS (see the Dockerfile).  Since body meshes are
  assembled in subprocesses (and chunks may be generated by the worker service), the
  request threads mostly just wait for I/O, so many threads per process are fine.
  Set the Cloud Run concurrency to GUNICORN_WORKERS * GUNICORN_THREADS, so that
  requests are sent to other instances (rather than queued) when an instance is busy.
- DVID access uses neuclease's default sessions, which pass DVID_ADMIN_TOKEN (if set)
  to DVID.  That is needed to store meshes in a locked (committed) DVID node.
- If the body mesh can't be updated (e.g. the node is locked and no token is set),
  the stored mesh is served anyway, if there is one.
- DVID requests use long timeouts (NGSUPPORT_DVID_TIMEOUT seconds, default: 600),
  since DVID may be slow to respond while it serves many chunk requests at once.
- Progress is logged at INFO level.  Set NGSUPPORT_MULTIRES_DEBUG_LOGGING=1 to also log
  the timing of every chunk.
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

import requests
from flask import Response, jsonify, redirect, request
from requests import HTTPError, RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from neuclease import redact_admintoken

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
DEBUG_LOGGING = os.environ.get('NGSUPPORT_MULTIRES_DEBUG_LOGGING', '').lower() in ('1', 'true', 'yes')
DVID_TIMEOUT = float(os.environ.get('NGSUPPORT_DVID_TIMEOUT', 600.0))
CHUNK_WORKERS = int(os.environ.get('NGSUPPORT_MULTIRES_CHUNK_WORKERS', 0))
ASSEMBLE_IN_SUBPROCESS = os.environ.get('NGSUPPORT_MULTIRES_ASSEMBLE_IN_SUBPROCESS', '1').lower() in ('1', 'true', 'yes')

# A separate deployment of this service, to which chunk requests are sent (if CHUNK_WORKERS).
# (The same setting is used by small_mesh.py.)
WORKER_BASE = os.environ.get('NGSUPPORT_SERVICE_WORKER_BASE', 'https://ngsupport-worker-bmcp5imp6q-uk.a.run.app')

# Timeouts (connect, read) for chunk requests sent to the worker service.
CHUNK_REQUEST_TIMEOUT = (30.0, 900.0)

# How long a resolved UUID can be reused by fragment requests (see _resolve_uuid()).
UUID_CACHE_SECONDS = 60

# For configs in a bucket: how long before we check the bucket for an updated config.
# (Every request loads the config, and we don't want to query the bucket every time.)
CONFIG_CACHE_SECONDS = 30

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


class ChunkWorkerError(HTTPError):
    """A chunk request sent to the worker service failed."""


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
        except ChunkWorkerError as ex:
            return _error_response(str(ex), HTTPStatus.BAD_GATEWAY)
        except HTTPError as ex:
            # DVID returns 400 for an unknown instance and 404 for an unknown body.
            status = ex.response.status_code if ex.response is not None else None
            if status in (HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND):
                return _error_response(f"Not found in DVID: {ex}", HTTPStatus.NOT_FOUND)
            return _error_response(f"Error from DVID: {ex}", HTTPStatus.BAD_GATEWAY)
        except RequestException as ex:
            return _error_response(f"Error accessing DVID ({type(ex).__name__}): {ex}", HTTPStatus.BAD_GATEWAY)
    return wrapper


def _error_response(msg, status):
    # DVID URLs may contain the admin token, which must not be sent to the client.
    return Response(redact_admintoken(msg), status, mimetype='text/plain')


@functools.cache
def _configure_dvid_access():
    """
    Configure the DVID sessions (and logging) of this process.  (Called once.)
    """
    from neuclease.dvid import set_default_dvid_session_timeout
    from neuclease.util import tqdm_proxy_config

    # By default, neuclease gives up if DVID doesn't accept a connection
    # within a few seconds, which can happen while it's busy serving our chunk requests.
    set_default_dvid_session_timeout(DVID_TIMEOUT, DVID_TIMEOUT)

    # Always report the progress of chunk generation in the log
    # (even when running in a terminal).
    tqdm_proxy_config['output_file'] = 'logger'
    if DEBUG_LOGGING:
        logging.getLogger('neuclease.misc.bodymesh').setLevel(logging.DEBUG)


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
    start = time.time()
    logger.info(f"Body {body}: Manifest requested ({server} {uuid} {seg_instance} {config_name})")
    try:
        response = _multires_mesh_index(server, uuid, seg_instance, config_name, body)
    except BaseException as ex:
        logger.info(f"Body {body}: Manifest request failed after {time.time() - start:.1f}s ({type(ex).__name__})")
        raise
    logger.info(f"Body {body}: Manifest request finished after {time.time() - start:.1f}s")
    return response


def _multires_mesh_index(server, uuid, seg_instance, config_name, body):
    from neuclease.misc.bodymesh import update_body_mesh

    _configure_dvid_access()
    config = _load_config(config_name)
    uuid = _resolve_uuid(server, uuid, fresh=True)

    chunk_fn = None
    if CHUNK_WORKERS:
        chunk_fn = functools.partial(_request_chunk_mesh, server, uuid, seg_instance,
                                     config_name, _config_digest(config), body)

    update_error = None
    with _body_lock(server, uuid, seg_instance, body):
        try:
            # This is a no-op if the stored mesh is already up-to-date.
            # Other service instances may be updating the same body at the same time.
            update_body_mesh(server, uuid, seg_instance, body,
                             copy.deepcopy(config.body), copy.deepcopy(config.chunk),
                             processes=PROCESSES, expect_concurrent_updates=True,
                             chunk_fn=chunk_fn, chunk_fn_threads=CHUNK_WORKERS,
                             assemble_in_subprocess=ASSEMBLE_IN_SUBPROCESS)
        except RequestException as ex:
            # Fall back to whatever mesh is already stored (if any).
            # (The log formatter redacts the admin token, if any.)
            logger.warning(f"Body {body}: Could not update the stored mesh ({type(ex).__name__}): {ex}", exc_info=True)
            update_error = ex

    index = None
    if location := _locate_body_mesh(server, uuid, seg_instance, body, fresh=True):
        if location.store:
            index = location.store.get(location.index_name)
        else:
            index = _fetch_value(server, uuid, location.mesh_instance, location.index_name)

    if index is None and update_error is not None:
        raise update_error
    if not index:
        raise NotFound(f"Body {body} has no mesh geometry")
    if update_error is not None:
        logger.warning(f"Body {body}: Serving the previously stored mesh")
    return Response(index, HTTPStatus.OK, mimetype=OCTET_STREAM, headers=NO_STORE_HEADERS)


@_error_responses
@normalize_server
def multires_mesh_chunk(server, uuid, seg_instance, config_name, body):
    """
    Generate (and store in DVID) a single chunk mesh for a body, and return it.
    This is how a service instance which is updating a body mesh distributes
    the work of generating its chunks (see _request_chunk_mesh()).

    The request body is JSON:

        {"chunk": [z, y, x, surface_mutid, block_hash], "config-digest": "..."}

    The response is the chunk's (draco) mesh fragment, possibly empty.
    """
    from neuclease.misc.bodymesh import mesh_for_chunk, quality_from_source_method, DummyResourceMgr

    _configure_dvid_access()
    config = _load_config(config_name)
    uuid = _resolve_uuid(server, uuid)

    payload = request.get_json(silent=True) or {}
    try:
        z, y, x, surface_mutid, block_hash = payload['chunk']
        chunk_spec = (int(z), int(y), int(x), int(surface_mutid), str(block_hash))
    except (KeyError, TypeError, ValueError):
        return _error_response("Expected JSON with a 'chunk' list: [z, y, x, surface_mutid, block_hash]",
                               HTTPStatus.BAD_REQUEST)

    if payload.get('config-digest') != _config_digest(config):
        msg = (f"Config '{config_name}' differs between the requesting service "
               "and this one.  Deploy the same configs to both.")
        return _error_response(msg, HTTPStatus.CONFLICT)

    quality = quality_from_source_method(config.body, config.chunk)
    fragment = mesh_for_chunk(
        server, uuid, seg_instance, body, copy.deepcopy(config.chunk), quality,
        True, DummyResourceMgr(), True, config.body['vertex-quantization-bits'], True,
        chunk_spec
    )
    return Response(fragment, HTTPStatus.OK, mimetype=OCTET_STREAM)


def _request_chunk_mesh(server, uuid, seg_instance, config_name, config_digest, body, chunk_spec):
    """
    Generate (and store) a chunk mesh by sending a request to the worker service.
    Used as the chunk_fn for update_body_mesh().  Returns the chunk's mesh fragment.
    """
    # The worker URL must express the server the way our routes expect it (see normalize_server()).
    server_path = server.replace('://', '-', 1)
    url = f'{WORKER_BASE}/multires-mesh/{server_path}/{uuid}/{seg_instance}/{config_name}/{body}/chunk'
    payload = {'chunk': list(chunk_spec), 'config-digest': config_digest}
    status, content = _post_chunk_request(url, payload)
    if status != HTTPStatus.OK:
        msg = content[:1000].decode('utf-8', errors='replace')
        z, y, x, *_ = chunk_spec
        r = requests.Response()
        r.status_code = status
        raise ChunkWorkerError(f"Chunk worker failed for chunk {x},{y},{z} ({status}): {msg}", response=r)
    return content


def _post_chunk_request(url, payload):
    """Send a chunk request to the worker service.  Returns (status, content)."""
    r = _chunk_worker_session().post(url, json=payload, timeout=CHUNK_REQUEST_TIMEOUT)
    return r.status_code, r.content


_chunk_worker_sessions = threading.local()


def _chunk_worker_session():
    """
    A (per-thread) session for chunk requests, which retries failed requests,
    e.g. 429 (Too Many Requests) while Cloud Run scales up the worker service.
    (Retrying is harmless, since a chunk request just stores the same chunk again,
    or finds that it has been stored already.)
    """
    try:
        return _chunk_worker_sessions.session
    except AttributeError:
        retry = Retry(total=6, backoff_factor=1.0, allowed_methods=None, raise_on_status=False,
                      status_forcelist=(429, 500, 502, 503, 504))
        session = requests.Session()
        session.mount('http://', HTTPAdapter(max_retries=retry))
        session.mount('https://', HTTPAdapter(max_retries=retry))
        _chunk_worker_sessions.session = session
        return session


@_error_responses
@normalize_server
def multires_mesh_data(server, uuid, seg_instance, config_name, body):
    """
    Return (a byte range of) a body's multires fragment data (the '<body>' file),
    by redirecting the client to the object store containing it (if any),
    or else by proxying the request to DVID or redirecting the client to DVID.
    """
    _load_config(config_name)
    uuid = _resolve_uuid(server, uuid)

    location = _locate_body_mesh(server, uuid, seg_instance, body)
    if location is None:
        raise NotFound(f"Body {body} has no stored mesh")

    if location.store or REDIRECT_FRAGMENTS:
        if location.store:
            url = location.store.public_url(location.data_name)
        else:
            # Note: The redirect URL must not include DVID_ADMIN_TOKEN.
            url = f'{server}/api/node/{uuid}/{location.mesh_instance}/key/{location.data_name}'
        r = redirect(url, HTTPStatus.TEMPORARY_REDIRECT)
        r.headers.update(NO_STORE_HEADERS)
        return r

    range_header = request.headers.get('Range')
    result = _fetch_value_range(server, uuid, location.mesh_instance, location.data_name, range_header)
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
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(f"Body {body}: Waiting for another request on this host to finish updating "
                        f"this body (or another body that shares lock stripe {stripe})")
            start = time.time()
            fcntl.flock(f, fcntl.LOCK_EX)
            logger.info(f"Body {body}: Acquired the body lock after waiting {time.time() - start:.1f}s")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


#
# Configs
#

MeshConfig = namedtuple('MeshConfig', 'body chunk')


def _config_digest(config):
    """A digest of a (validated) config's settings, to check that two services agree on them."""
    settings = json.dumps([config.body, config.chunk], sort_keys=True)
    return hashlib.sha256(settings.encode()).hexdigest()[:16]

# Config names become file names, so don't permit anything path-like.
_CONFIG_NAME_PATTERN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')


CONFIG_EXTENSIONS = ('.yaml', '.yml', '.json')


def _load_config(config_name):
    if not _CONFIG_NAME_PATTERN.fullmatch(config_name) or '..' in config_name:
        raise ConfigError(f"Invalid multires mesh config name: {config_name!r}")

    if '://' in CONFIG_DIR:
        return _load_config_from_store(CONFIG_DIR, config_name)

    for ext in CONFIG_EXTENSIONS:
        path = os.path.join(CONFIG_DIR, config_name + ext)
        if os.path.isfile(path):
            # Including the mtime in the cache key means edited configs are reloaded.
            return _load_config_file(path, os.path.getmtime(path))

    raise ConfigError(f"Unknown multires mesh config: {config_name!r}")


_config_locations = {}
_config_locations_lock = threading.Lock()


def _load_config_from_store(config_dir, config_name):
    """
    Load a config from a bucket (or other object store; see neuclease.misc.meshstore).
    Which object holds the config (and its generation) is cached for CONFIG_CACHE_SECONDS.
    """
    from neuclease.misc.meshstore import open_store

    key = (config_dir, config_name)
    now = time.time()
    with _config_locations_lock:
        cached = _config_locations.get(key)
    if cached and now - cached[0] < CONFIG_CACHE_SECONDS:
        location = cached[1]
    else:
        location = None
        store = open_store(config_dir)
        for ext in CONFIG_EXTENSIONS:
            if (info := store.stat(config_name + ext)):
                location = (config_name + ext, info.generation)
                break
        with _config_locations_lock:
            _config_locations[key] = (now, location)

    if location is None:
        raise ConfigError(f"Unknown multires mesh config: {config_name!r}")

    # Including the generation in the cache key means edited configs are reloaded.
    name, generation = location
    return _load_config_object(config_dir, name, generation)


@functools.lru_cache(maxsize=32)
def _load_config_object(config_dir, name, _generation):
    from neuclease.misc.meshstore import open_store
    text = open_store(config_dir).get(name)
    if text is None:
        # It was deleted since we checked.
        raise ConfigError(f"Unknown multires mesh config: {name!r}")
    return _parse_config(text.decode('utf-8'), f'{config_dir.rstrip("/")}/{name}')


@functools.lru_cache(maxsize=32)
def _load_config_file(path, _mtime):
    with open(path, 'r') as f:
        return _parse_config(f.read(), path)


def _parse_config(text, path):
    """Parse and validate a config.  (The path is just for error messages.)"""
    from confiddler import validate
    from jsonschema import ValidationError
    from ruamel.yaml import YAML
    from neuclease.misc.bodymesh import (
        BodyMeshParametersSchema, MeshChunkConfigSchema, quality_from_source_method)

    # Parse with YAML 1.2 (via ruamel), as flyemflows does.
    # (With PyYAML's YAML 1.1, values like '10e6' would be parsed as strings.)
    cfg = YAML(typ='safe').load(text) or {}

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


_body_mesh_locations = {}
_body_mesh_locations_lock = threading.Lock()


def _locate_body_mesh(server, uuid, seg_instance, body, fresh=False):
    """
    Find a body's stored multires mesh (see bodymesh.locate_multires_body_mesh()).
    Returns None if there is none, or a SimpleNamespace with attributes
    store, index_name, data_name, mesh_info, and mesh_instance.

    Manifest requests locate the mesh afresh (fresh=True).  Fragment requests
    (which are numerous) reuse a recent result.
    """
    from neuclease.misc.bodymesh import locate_multires_body_mesh, mesh_instance_names

    key = (server, uuid, seg_instance, int(body))
    now = time.time()
    if not fresh:
        with _body_mesh_locations_lock:
            cached = _body_mesh_locations.get(key)
        if cached and now - cached[0] < UUID_CACHE_SECONDS:
            return cached[1]

    location = locate_multires_body_mesh(server, uuid, seg_instance, body)
    if location is not None:
        location.mesh_instance = mesh_instance_names(seg_instance, multires=True).body

    with _body_mesh_locations_lock:
        _body_mesh_locations[key] = (now, location)
    return location


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
