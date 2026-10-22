"""
Tests for the /multires-mesh endpoints (ngsupport/multires_mesh.py).

DVID is faked, including an in-memory keyvalue store, so the real
neuclease.misc.multiresmesh.update_multires_body_mesh() machinery (chunk reuse,
staleness checks, body assembly) runs against it.  The fake segmentation contains a single
body (label 1) shaped like a sphere, spanning a 3x3x3 grid of 64-voxel chunks.
"""
import json
import time
import threading
from functools import partial
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests
from requests import HTTPError

pytest.importorskip("DracoPy")
pytest.importorskip("trimesh")

import DracoPy  # noqa: E402

SOURCE = '/multires-mesh/fake-dvid/abc123:master/segmentation/test'
RESOLVED_UUID = 'abc123def456abc123def456abc12345'
MESH_INSTANCES = {'segmentation_multires_meshes', 'segmentation_multires_mesh_info'}

TEST_CONFIG = """\
# A complete flyemflows 'multiresbodymeshes' workflow config is accepted.
workflow-name: multiresbodymeshes
multiresbodymeshes:
  mesh-config:
    base-uuid: abc123
    chunk-shape-s0: [64, 64, 64]
    chunk-halo: 2
    source-scale: 1
    fill-holes: false
    smoothing: 1
    lod-error-thresholds: [0.5, 5.0]
"""

# The same settings, as a plain config.
PLAIN_CONFIG = """\
base-uuid: abc123
chunk-shape-s0: [64, 64, 64]
chunk-halo: 2
source-scale: 1
fill-holes: false
smoothing: 1
lod-error-thresholds: [0.5, 5.0]
"""

ALL_CELLS = {(x, y, z) for x in range(3) for y in range(3) for z in range(3)}


def _dvid_error(status):
    r = requests.Response()
    r.status_code = status
    return HTTPError(f"{status} Error", response=r)


def _crop(vol, box):
    """Extract box from vol, zero-padding wherever the box extends beyond the volume."""
    box = np.asarray(box)
    out = np.zeros(box[1] - box[0], dtype=bool)
    lo = np.maximum(box[0], 0)
    hi = np.minimum(box[1], vol.shape)
    if (hi > lo).all():
        out[tuple(slice(a - b, c - b) for a, c, b in zip(lo, hi, box[0]))] = \
            vol[tuple(slice(a, c) for a, c in zip(lo, hi))]
    return out


class FakeDVID:
    """Just enough of DVID for multiresmesh.update_multires_body_mesh() and the multires-mesh endpoints."""

    def __init__(self):
        import pandas as pd

        zz, yy, xx = np.ogrid[:192, :192, :192]
        vol_s0 = ((zz - 96)**2 + (yy - 96)**2 + (xx - 96)**2) <= 70**2
        self.vols = {0: vol_s0, 1: vol_s0[::2, ::2, ::2]}

        block_zyx = np.unique((np.argwhere(vol_s0) // 64) * 64, axis=0)
        self.blocks_df = pd.DataFrame(block_zyx, columns=[*'zyx'])
        self.blocks_df['sv'] = np.uint64(1)
        self.blocks_df['count'] = 1
        self.surface_mutids = pd.DataFrame(block_zyx, columns=[*'zyx'])
        self.surface_mutids['surface_mutid'] = 10

        self.mutid = 10
        self.instances = set(MESH_INSTANCES)
        self.kv = {}
        self.sparsevol_requests = 0
        self.sparsevol_delay = 0.0     # seconds per sparsevol request
        self.on_sparsevol = None       # optional hook, called on each sparsevol request
        self.posted_keys = []
        self.writable = True
        self.honor_range = True
        self._lock = threading.Lock()

    def edit_chunk(self, chunk_zyx):
        """Simulate an edit to the body within the given chunk."""
        self.mutid += 1
        sm = self.surface_mutids
        sm.loc[(sm[[*'zyx']].values == chunk_zyx).all(axis=1), 'surface_mutid'] = self.mutid

    # neuclease API

    def resolve_ref(self, server, ref, expand=False, **kwargs):
        if not ref.startswith('abc123'):
            raise RuntimeError(f"Could not resolve reference '{ref}'")
        return RESOLVED_UUID

    def fetch_instance_info(self, server, uuid, instance, **kwargs):
        if instance != 'segmentation':
            raise _dvid_error(400)
        return {'Extended': {'VoxelSize': [8.0, 8.0, 8.0]}}

    def fetch_lastmod(self, server, uuid, instance, label, **kwargs):
        if label != 1:
            raise _dvid_error(404)
        return {'mutation id': self.mutid}

    def fetch_labelindex(self, server, uuid, instance, label, format='protobuf', **kwargs):
        from neuclease.dvid.labelmap import PandasLabelIndex
        assert uuid == RESOLVED_UUID
        if label != 1:
            raise _dvid_error(404)
        return PandasLabelIndex(self.blocks_df.copy(), 1, self.mutid, '', '', self.surface_mutids.copy())

    def fetch_sparsevol(self, server, uuid, instance, label, scale=0, supervoxels=False, *,
                        format='coords', mask_box=None, **kwargs):
        with self._lock:
            self.sparsevol_requests += 1
        if self.on_sparsevol:
            self.on_sparsevol()
        time.sleep(self.sparsevol_delay)
        mask = _crop(self.vols[scale], mask_box)
        if not mask.any():
            raise _dvid_error(404)
        return mask, np.asarray(mask_box)

    def _check_instance(self, instance):
        if instance not in self.instances:
            raise _dvid_error(400)

    def fetch_key(self, server, uuid, instance, key, *, as_json=False, check_head=False, **kwargs):
        self._check_instance(instance)
        value = self.kv.get((instance, str(key)))
        if check_head:
            return value is not None
        if value is None:
            raise _dvid_error(404)
        return json.loads(value) if as_json else value

    def post_key(self, server, uuid, instance, key, data=None, json=None, **kwargs):
        import json as _json
        self._check_instance(instance)
        if not self.writable:
            raise _dvid_error(400)  # e.g. "node is locked"
        self.kv[(instance, str(key))] = _json.dumps(json).encode() if json is not None else data
        self.posted_keys.append((instance, str(key)))

    def delete_key(self, server, uuid, instance, key, **kwargs):
        self._check_instance(instance)
        self.kv.pop((instance, str(key)), None)

    # The raw keyvalue GET used by the multires-mesh fragment proxy
    def fetch_value_range(self, server, uuid, instance, key, range_header=None):
        from werkzeug.http import parse_range_header
        from ngsupport.multires_mesh import RangeResult
        self._check_instance(instance)
        value = self.kv.get((instance, key))
        if value is None:
            return None
        rng = parse_range_header(range_header) if (range_header and self.honor_range) else None
        if rng is None:
            return RangeResult(200, None, value)
        bounds = rng.range_for_length(len(value))
        if bounds is None:
            return RangeResult(416, f'bytes */{len(value)}', b'')
        start, stop = bounds
        return RangeResult(206, f'bytes {start}-{stop - 1}/{len(value)}', value[start:stop])


@pytest.fixture
def service(monkeypatch, tmp_path):
    """
    A Flask test client for ngsupport, with DVID faked out, a temporary config
    directory, and serial chunk mesh generation (so the fakes apply).
    Returns (client, dvid).
    """
    import neuclease.dvid
    import neuclease.dvid.keyvalue
    from neuclease.misc import bodymesh, multiresmesh
    from neuclease.misc.meshstore import MemoryStore
    from ngsupport import multires_mesh
    from ngsupport.app import app

    dvid = FakeDVID()
    for mod in (bodymesh, multiresmesh):
        for name in ('resolve_ref', 'fetch_lastmod', 'fetch_labelindex', 'fetch_sparsevol',
                     'fetch_key', 'post_key', 'delete_key'):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, getattr(dvid, name))
    monkeypatch.setattr(multiresmesh, 'fetch_resolution_zyx', lambda *args: (8.0, 8.0, 8.0))
    monkeypatch.setattr(neuclease.dvid, 'resolve_ref', dvid.resolve_ref)
    monkeypatch.setattr(neuclease.dvid, 'fetch_instance_info', dvid.fetch_instance_info)
    monkeypatch.setattr(neuclease.dvid.keyvalue, 'fetch_key', dvid.fetch_key)
    monkeypatch.setattr(multires_mesh, '_fetch_value_range', dvid.fetch_value_range)

    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    (config_dir / 'test.yaml').write_text(TEST_CONFIG)
    monkeypatch.setattr(multires_mesh, 'CONFIG_DIR', str(config_dir))
    monkeypatch.setattr(multires_mesh, 'LOCK_DIR', str(tmp_path / 'locks'))
    monkeypatch.setattr(multires_mesh, 'PROCESSES', 0)
    monkeypatch.setattr(multires_mesh, 'REDIRECT_FRAGMENTS', False)
    monkeypatch.setattr(multires_mesh, '_resolved_uuids', {})
    monkeypatch.setattr(multires_mesh, '_body_mesh_locations', {})
    multires_mesh._voxel_size_xyz.cache_clear()
    multires_mesh._load_config_file.cache_clear()
    multires_mesh._load_config_object.cache_clear()
    monkeypatch.setattr(multires_mesh, '_config_locations', {})
    MemoryStore.clear_all()
    multiresmesh._verified_store_infos.clear()

    return app.test_client(), dvid


def _parse_manifest(index_bytes):
    """Return (num_lods, [(lod, (x, y, z), byte_offset, byte_size), ...]) from a manifest."""
    pos = 0

    def take(dtype, count):
        nonlocal pos
        a = np.frombuffer(index_bytes, dtype=dtype, count=count, offset=pos)
        pos += a.nbytes
        return a

    take('<f4', 3)
    take('<f4', 3)
    num_lods = int(take('<u4', 1)[0])
    take('<f4', num_lods)
    take('<f4', num_lods * 3)
    num_frags = take('<u4', num_lods)
    fragments = []
    offset = 0
    for lod in range(num_lods):
        n = int(num_frags[lod])
        positions = take('<u4', 3 * n).reshape(3, n).T
        sizes = take('<u4', n)
        for p, size in zip(positions, sizes):
            fragments.append((lod, tuple(int(c) for c in p), offset, int(size)))
            offset += int(size)
    assert pos == len(index_bytes)
    return num_lods, fragments


def _fetch_all_fragments(client, fragments):
    """Fetch each fragment via a Range request, as neuroglancer does, and check it."""
    total = sum(size for *_, size in fragments)
    for _lod, _p, offset, size in fragments:
        if size == 0:
            continue
        r = client.get(f'{SOURCE}/1', headers={'Range': f'bytes={offset}-{offset + size - 1}'})
        assert r.status_code == 206, r.data
        assert r.headers['Content-Range'] == f'bytes {offset}-{offset + size - 1}/{total}'
        assert r.headers['Cache-Control'] == 'no-store'
        assert len(DracoPy.decode(r.data).faces) > 0


def _positions(fragments, lod):
    return {p for (frag_lod, p, _, size) in fragments if size and frag_lod == lod}


def test_info(service):
    client, _ = service
    r = client.get(f'{SOURCE}/info')
    assert r.status_code == 200, r.data
    info = r.get_json()
    assert info['@type'] == 'neuroglancer_multilod_draco'
    assert info['vertex_quantization_bits'] == 16
    assert info['lod_scale_multiplier'] == 1.0
    assert info['transform'] == [8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0]


def test_manifest_generates_and_stores_mesh(service):
    client, dvid = service

    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200, r.data
    assert r.headers['Content-Type'] == 'application/octet-stream'
    assert r.headers['Cache-Control'] == 'no-store'

    # The chunks were generated, and the body mesh was stored in DVID.
    assert dvid.sparsevol_requests == 27
    assert dvid.kv[('segmentation_multires_meshes', '1.index')] == r.data
    assert ('segmentation_multires_meshes', '1.chunks') in dvid.kv
    stored_info = json.loads(dvid.kv[('segmentation_multires_meshes', 'info')])
    assert stored_info == client.get(f'{SOURCE}/info').get_json()
    mesh_info = json.loads(dvid.kv[('segmentation_multires_mesh_info', '1')])
    assert mesh_info['lastmod'] == 10

    num_lods, fragments = _parse_manifest(r.data)
    assert num_lods == 2
    # Every chunk of the sphere's 3x3x3 grid except the (fully interior) center chunk.
    assert _positions(fragments, 0) == ALL_CELLS - {(1, 1, 1)}
    assert _positions(fragments, 1) == {(x, y, z) for x in range(2) for y in range(2) for z in range(2)}
    _fetch_all_fragments(client, fragments)

    # A second manifest request finds the stored mesh up-to-date.
    assert client.get(f'{SOURCE}/1.index').data == r.data
    assert dvid.sparsevol_requests == 27


def test_plain_config(service, tmp_path):
    """A config file may contain just the mesh settings (rather than a workflow config)."""
    client, _dvid = service
    (tmp_path / 'configs' / 'test.yaml').write_text(PLAIN_CONFIG)
    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200, r.data
    assert _parse_manifest(r.data)[0] == 2


def test_edit_regenerates_only_affected_chunks(service):
    client, dvid = service
    first = client.get(f'{SOURCE}/1.index').data
    assert dvid.sparsevol_requests == 27

    dvid.edit_chunk([0, 0, 0])
    second = client.get(f'{SOURCE}/1.index')
    assert second.status_code == 200

    # Only the edited chunk was regenerated, and the body mesh was reassembled.
    assert dvid.sparsevol_requests == 28
    mesh_info = json.loads(dvid.kv[('segmentation_multires_mesh_info', '1')])
    assert mesh_info['lastmod'] == 11
    assert (mesh_info['generated-chunk-count'], mesh_info['reused-chunk-count']) == (1, 26)
    _num_lods, fragments = _parse_manifest(second.data)
    _fetch_all_fragments(client, fragments)
    assert len(first) == len(second.data)  # (Same fragment count; contents may differ.)


def test_fragments_from_dvid_without_range_support(service):
    """If DVID ignores the Range header (older DVID), the service slices the value itself."""
    client, dvid = service
    dvid.honor_range = False
    _num_lods, fragments = _parse_manifest(client.get(f'{SOURCE}/1.index').data)
    _fetch_all_fragments(client, fragments)

    full = client.get(f'{SOURCE}/1')
    assert full.status_code == 200
    assert full.data == dvid.kv[('segmentation_multires_meshes', '1')]


def test_redirect_fragments(service, monkeypatch):
    from ngsupport import multires_mesh
    client, _ = service
    monkeypatch.setattr(multires_mesh, 'REDIRECT_FRAGMENTS', True)
    assert client.get(f'{SOURCE}/1.index').status_code == 200

    r = client.get(f'{SOURCE}/1', headers={'Range': 'bytes=0-99'})
    assert r.status_code == HTTPStatus.TEMPORARY_REDIRECT
    assert r.headers['Location'] == \
        f'https://fake-dvid/api/node/{RESOLVED_UUID}/segmentation_multires_meshes/key/1'


def test_stale_mesh_served_if_update_fails(service):
    client, dvid = service
    first = client.get(f'{SOURCE}/1.index').data

    # The body changes, but the mesh can't be updated (e.g. the node is locked).
    dvid.edit_chunk([0, 0, 0])
    dvid.writable = False
    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200
    assert r.data == first


def test_update_failure_with_no_stored_mesh(service):
    client, dvid = service
    dvid.writable = False
    assert client.get(f'{SOURCE}/1.index').status_code == 404


def test_not_found(service):
    client, dvid = service
    assert client.get(f'{SOURCE}/2.index').status_code == 404               # no such body
    assert client.get(f'{SOURCE}/2').status_code == 404
    assert client.get(f'{SOURCE}/1').status_code == 404                     # not generated yet
    assert client.get(SOURCE.replace('/test', '/nosuchconfig') + '/info').status_code == 404
    assert client.get(SOURCE.replace('/test', '/..test') + '/info').status_code == 404
    assert client.get(SOURCE.replace('abc123:master', 'zzz:master') + '/info').status_code == 404
    assert client.get(SOURCE.replace('/segmentation/', '/nosuchinstance/') + '/info').status_code == 404
    assert dvid.sparsevol_requests == 0


def test_mesh_instances_not_initialized(service):
    client, dvid = service
    dvid.instances = set()
    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 404
    assert b'Not found in DVID' in r.data


def test_invalid_configs(service, tmp_path):
    client, _ = service
    configs = {
        'decreasing': TEST_CONFIG.replace('[0.5, 5.0]', '[5.0, 0.5]'),
        'badshape': TEST_CONFIG.replace('[64, 64, 64]', '[64, 64, 100]'),
        'unknownsetting': TEST_CONFIG + '    num-lods: 2\n',
        'nomeshconfig': 'workflow-name: multiresbodymeshes\nmultiresbodymeshes: {}\n',
        'oldformat': 'body-meshes: {}\nchunk-meshes: {}\n',
    }
    for name, text in configs.items():
        (tmp_path / 'configs' / f'{name}.yaml').write_text(text)
        r = client.get(SOURCE.replace('/test', f'/{name}') + '/info')
        assert r.status_code == 404, name
        assert f'{name}.yaml'.encode() in r.data, name


def test_unsatisfiable_range(service):
    client, _ = service
    assert client.get(f'{SOURCE}/1.index').status_code == 200
    assert client.get(f'{SOURCE}/1', headers={'Range': 'bytes=999999999-'}).status_code == 416


def _update_body_1():
    """Call update_multires_body_mesh() for body 1 the way the service does."""
    from neuclease.misc.multiresmesh import update_multires_body_mesh
    from ngsupport import multires_mesh
    config = multires_mesh._load_config('test')
    return update_multires_body_mesh('https://fake-dvid', RESOLVED_UUID, 'segmentation', 1,
                                     json.loads(json.dumps(config)), processes=0, expect_concurrent_updates=True)


def test_concurrent_updates(service):
    """Two concurrent updates of the same body (as if on two different hosts) both succeed."""
    client, dvid = service
    dvid.sparsevol_delay = 0.01

    with ThreadPoolExecutor(2) as executor:
        for f in [executor.submit(_update_body_1) for _ in range(2)]:
            f.result()

    # The stored mesh is complete and usable.
    r = client.get(f'{SOURCE}/1.index')
    _num_lods, fragments = _parse_manifest(r.data)
    assert _positions(fragments, 0) == ALL_CELLS - {(1, 1, 1)}
    _fetch_all_fragments(client, fragments)
    assert dvid.sparsevol_requests <= 54


def test_body_mesh_not_stored_if_another_process_finished_it(service):
    """
    If another process stores an up-to-date body mesh while we're generating chunks,
    we don't assemble (or store) our own.
    """
    _client, dvid = service

    from neuclease.misc import multiresmesh
    from ngsupport import multires_mesh
    config = multires_mesh._load_config('test')

    def another_process_finishes():
        mesh_info = {'body': 1, 'lastmod': dvid.mutid, 'method': multiresmesh.MESH_METHOD,
                     'mesh-info-version': multiresmesh.MESH_INFO_VERSION, 'storage': 'dvid',
                     'mesh-settings': multiresmesh.mesh_settings(config)}
        dvid.kv[('segmentation_multires_mesh_info', '1')] = json.dumps(mesh_info).encode()

    dvid.on_sparsevol = another_process_finishes
    assert _update_body_1() is None

    assert dvid.sparsevol_requests == 27
    assert ('segmentation_multires_meshes', '1') not in dvid.posted_keys
    assert ('segmentation_multires_meshes', '1.index') not in dvid.posted_keys


def test_dvid_errors_are_reported_without_admintoken(service, monkeypatch):
    """
    If DVID can't be reached, the error response explains why (e.g. a timeout),
    but doesn't reveal the admin token from the DVID URL.
    """
    from neuclease.dvid._dvid import dvid_api_wrapper
    from neuclease.misc import multiresmesh
    client, _dvid = service

    @dvid_api_wrapper
    def fetch_lastmod(server, uuid, instance, body, *, session=None):
        url = f'{server}/api/node/{uuid}/{instance}/lastmod/{body}?u=me&admintoken=SECRET'
        raise requests.ConnectTimeout(
            f"Max retries exceeded with url: {url} (Caused by ConnectTimeoutError())",
            request=requests.Request('GET', url).prepare())

    monkeypatch.setattr(multiresmesh, 'fetch_lastmod', partial(fetch_lastmod, session=object()))
    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == HTTPStatus.BAD_GATEWAY
    assert b'ConnectTimeout' in r.data
    assert b'Max retries exceeded' in r.data
    assert b'admintoken=<redacted>' in r.data
    assert b'SECRET' not in r.data


@pytest.fixture
def chunk_workers(service, monkeypatch):
    """
    Send chunk requests to the 'worker service' (which, here, is just the same test app).
    Returns the list of chunk request URLs.
    """
    from ngsupport import multires_mesh
    client, _dvid = service
    urls = []
    lock = threading.Lock()

    def post_chunk_request(url, payload):
        with lock:
            urls.append(url)
        assert url.startswith('https://worker')
        r = client.post(url[len('https://worker'):], json=payload)
        return r.status_code, r.data

    monkeypatch.setattr(multires_mesh, 'CHUNK_WORKERS', 3)
    monkeypatch.setattr(multires_mesh, 'WORKER_BASE', 'https://worker')
    monkeypatch.setattr(multires_mesh, '_post_chunk_request', post_chunk_request)
    return urls


def test_chunks_generated_by_workers(service, chunk_workers):
    client, dvid = service
    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200, r.data

    # Each chunk was requested from the workers (with the resolved UUID).
    assert len(chunk_workers) == 27
    assert all(u.startswith(f'https://worker/multires-mesh/https-fake-dvid/{RESOLVED_UUID}/segmentation/test/1/chunk')
               for u in chunk_workers)
    assert dvid.sparsevol_requests == 27

    num_lods, fragments = _parse_manifest(r.data)
    assert num_lods == 2
    assert _positions(fragments, 0) == ALL_CELLS - {(1, 1, 1)}
    _fetch_all_fragments(client, fragments)

    # After an edit, only the affected chunk is requested again.
    dvid.edit_chunk([0, 0, 0])
    assert client.get(f'{SOURCE}/1.index').status_code == 200
    assert len(chunk_workers) == 28


def test_chunk_endpoint_returns_pieces(service):
    from neuclease.misc.multiresmesh import decode_chunk_pieces
    from ngsupport import multires_mesh
    client, dvid = service
    config = multires_mesh._load_config('test')
    url = f'/multires-mesh/fake-dvid/{RESOLVED_UUID}/segmentation/test/1/chunk'
    r = client.post(url, json={'chunk': [0, 0, 0], 'config-digest': multires_mesh._config_digest(config)})
    assert r.status_code == 200, r.data
    pieces = decode_chunk_pieces(r.data)
    assert len(pieces) == 2
    assert all(len(DracoPy.decode(p).faces) > 0 for p in pieces)

    # Nothing was stored.
    assert dvid.posted_keys == []


def test_chunk_worker_rejects_mismatched_config(service, chunk_workers, monkeypatch):
    """If the worker's config differs from the requesting service's config, it does nothing."""
    from ngsupport import multires_mesh
    client, dvid = service

    post = multires_mesh._post_chunk_request
    monkeypatch.setattr(multires_mesh, '_post_chunk_request',
                        lambda url, payload: post(url, {**payload, 'config-digest': 'something-else'}))

    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == HTTPStatus.BAD_GATEWAY
    assert b'differs between the requesting service' in r.data
    assert dvid.sparsevol_requests == 0


def test_chunk_endpoint_rejects_bad_requests(service):
    client, _ = service
    url = f'/multires-mesh/fake-dvid/{RESOLVED_UUID}/segmentation/test/1/chunk'
    assert client.post(url, json={'chunk': [0, 0]}).status_code == HTTPStatus.BAD_REQUEST
    assert client.post(url, json={'chunk': [0, 0, 0, 10, '0x0']}).status_code == HTTPStatus.BAD_REQUEST
    assert client.post(url, data=b'garbage').status_code == HTTPStatus.BAD_REQUEST


@pytest.fixture
def bucket_config(service, tmp_path):
    """Store the body meshes in an (in-memory) bucket instead of DVID."""
    from neuclease.misc.meshstore import open_store
    config = TEST_CONFIG + '    storage: mem://bucket/bodies\n'
    (tmp_path / 'configs' / 'test.yaml').write_text(config)
    return open_store('mem://bucket/bodies')


def test_meshes_stored_in_bucket(service, bucket_config):
    client, dvid = service
    body_store = bucket_config

    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200, r.data

    # The body mesh is in the bucket, not DVID (except the mesh_info).
    assert sorted(body_store.list('')) == ['1', '1.chunks', '1.index', 'info']
    assert body_store.get('1.index') == r.data
    assert json.loads(body_store.get('info')) == client.get(f'{SOURCE}/info').get_json()
    assert ('segmentation_multires_meshes', '1') not in dvid.kv
    mesh_info = json.loads(dvid.kv[('segmentation_multires_mesh_info', '1')])
    assert mesh_info['storage'] == 'mem://bucket/bodies'
    assert mesh_info['written-uuid'] == RESOLVED_UUID
    assert body_store.stat('1').metadata == {'uuid': RESOLVED_UUID, 'lastmod': '10'}

    # Fragment requests are redirected to the bucket.
    f = client.get(f'{SOURCE}/1', headers={'Range': 'bytes=0-99'})
    assert f.status_code == HTTPStatus.TEMPORARY_REDIRECT
    assert f.headers['Location'] == body_store.public_url('1')
    assert f.headers['Cache-Control'] == 'no-store'

    # The fragments are valid.
    data = body_store.get('1')
    _num_lods, fragments = _parse_manifest(r.data)
    for _lod, _pos, offset, size in fragments:
        if size:
            assert len(DracoPy.decode(data[offset:offset + size]).faces) > 0

    # After an edit, only the affected chunk is regenerated, and the
    # body mesh is replaced in place (it's the same UUID).
    dvid.edit_chunk([0, 0, 0])
    r2 = client.get(f'{SOURCE}/1.index')
    assert r2.status_code == 200
    assert dvid.sparsevol_requests == 28
    assert sorted(body_store.list('')) == ['1', '1.chunks', '1.index', 'info']
    assert body_store.stat('1').metadata['lastmod'] == str(dvid.mutid)


def test_chunk_workers_with_bucket(service, chunk_workers, bucket_config):
    client, _dvid = service
    body_store = bucket_config
    assert client.get(f'{SOURCE}/1.index').status_code == 200
    assert len(chunk_workers) == 27
    assert body_store.get('1.index') is not None


def test_changed_settings_regenerate_body_mesh(service, tmp_path):
    """If the config's mesh settings change, stored meshes are regenerated from scratch."""
    client, dvid = service
    first = client.get(f'{SOURCE}/1.index').data

    config = TEST_CONFIG.replace('[0.5, 5.0]', '[0.5, 5.0, 50.0]')
    (tmp_path / 'configs' / 'test.yaml').write_text(config)
    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200
    assert r.data != first
    assert _parse_manifest(r.data)[0] == 3
    assert dvid.sparsevol_requests == 54


@pytest.fixture
def bucket_config_dir(service, monkeypatch):
    """Load configs from an (in-memory) bucket instead of a local directory."""
    from neuclease.misc.meshstore import open_store
    from ngsupport import multires_mesh
    monkeypatch.setattr(multires_mesh, 'CONFIG_DIR', 'mem://config-bucket/configs')
    store = open_store('mem://config-bucket/configs')
    store.put('test.yaml', TEST_CONFIG.encode())
    return store


def test_config_from_bucket(service, bucket_config_dir, monkeypatch):
    from ngsupport import multires_mesh
    client, _dvid = service
    store = bucket_config_dir

    r = client.get(f'{SOURCE}/1.index')
    assert r.status_code == 200, r.data
    assert _parse_manifest(r.data)[0] == 2

    unknown = client.get(SOURCE.replace('/test', '/nosuchconfig') + '/info')
    assert unknown.status_code == 404
    assert b'Unknown multires mesh config' in unknown.data

    # Edits to the config in the bucket are noticed once the cached lookup expires.
    store.put('test.yaml', TEST_CONFIG.replace('[0.5, 5.0]', '[0.5, 5.0, 50.0]').encode())
    assert _parse_manifest(client.get(f'{SOURCE}/1.index').data)[0] == 2
    monkeypatch.setattr(multires_mesh, 'CONFIG_CACHE_SECONDS', 0)
    assert _parse_manifest(client.get(f'{SOURCE}/1.index').data)[0] == 3

    # Invalid configs in the bucket are reported, too.
    store.put('bad.yaml', b'lod-error-thresholds: [5, 1]\n')
    bad = client.get(SOURCE.replace('/test', '/bad') + '/info')
    assert bad.status_code == 404
    assert b'mem://config-bucket/configs/bad.yaml' in bad.data


def test_config_bucket_lookups_are_cached(service, bucket_config_dir, monkeypatch):
    """Fragment requests (which also load the config) don't each query the bucket."""
    from neuclease.misc.meshstore import MemoryStore
    client, _dvid = service

    stats = []
    real_stat = MemoryStore.stat
    monkeypatch.setattr(MemoryStore, 'stat', lambda self, name: stats.append(name) or real_stat(self, name))

    r = client.get(f'{SOURCE}/1.index')
    _num_lods, fragments = _parse_manifest(r.data)
    _fetch_all_fragments(client, fragments)
    assert len([f for f in fragments if f[-1]]) > 10
    assert stats == ['test.yaml']
