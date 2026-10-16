"""
Tests for the /multires-mesh endpoints (ngsupport/multires_mesh.py).

DVID is faked, including an in-memory keyvalue store, so the real
neuclease.misc.bodymesh.update_body_mesh() machinery (chunk cache, staleness
checks, body assembly) runs against it.  The fake segmentation contains a single
body (label 1) shaped like a sphere, spanning a 3x3x3 grid of 64-voxel chunks.
"""
import json
import time
import threading
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
MESH_INSTANCES = {'segmentation_multires_meshes', 'segmentation_multires_mesh_info',
                  'segmentation_multires_chunk_meshes'}

TEST_CONFIG = """\
# A complete flyemflows 'chunkedbodymeshes' workflow config is accepted.
workflow-name: chunkedbodymeshes
chunkedbodymeshes:
  body-meshes:
    format: neuroglancer-multi-res
    source-method: concatenated-chunks-test-q
    smoothing: 0
    small-body-overall-decimation-s0: 1.0
    large-body-overall-decimation-s0: 1.0
    # YAML 1.2 floats (these would be strings under PyYAML's YAML 1.1)
    small-body-cutoff-vertices-s0: 10e6
    large-body-cutoff-vertices-s0: 100e6
  chunk-meshes:
    config-name: test
    base-uuid: abc123
    chunk-shape-s0: [64, 64, 64]
    chunk-halo: 2
    quality-configs:
      - name: q
        source-scale: 1
        morphological-closing-s0: 0
        fill-holes: false
        smoothing: 1
        decimation-s0: 1.0
"""


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
    """Just enough of DVID for bodymesh.update_body_mesh() and the multires-mesh endpoints."""

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

    def fetch_keyrange(self, server, uuid, instance, key1, key2, **kwargs):
        self._check_instance(instance)
        return sorted(k for (i, k) in self.kv if i == instance and key1 <= k <= key2)

    def fetch_keyvalues(self, server, uuid, instance, keys, **kwargs):
        self._check_instance(instance)
        return {k: self.kv.get((instance, k)) for k in keys}

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
    from neuclease.misc import bodymesh
    from ngsupport import multires_mesh
    from ngsupport.app import app

    dvid = FakeDVID()
    for name in ('resolve_ref', 'fetch_lastmod', 'fetch_labelindex', 'fetch_sparsevol',
                 'fetch_key', 'post_key', 'delete_key', 'fetch_keyrange', 'fetch_keyvalues'):
        monkeypatch.setattr(bodymesh, name, getattr(dvid, name))
    monkeypatch.setattr(bodymesh, 'fetch_resolution_zyx', lambda *args: (8.0, 8.0, 8.0))
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
    multires_mesh._voxel_size_xyz.cache_clear()
    multires_mesh._load_config_file.cache_clear()

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

    # The chunks and the body mesh were generated and stored in DVID.
    assert dvid.sparsevol_requests == 27
    chunk_keys = [k for (i, k) in dvid.kv if i == 'segmentation_multires_chunk_meshes']
    assert len(chunk_keys) == 27
    assert dvid.kv[('segmentation_multires_meshes', '1.index')] == r.data
    stored_info = json.loads(dvid.kv[('segmentation_multires_meshes', 'info')])
    assert stored_info == client.get(f'{SOURCE}/info').get_json()
    mesh_info = json.loads(dvid.kv[('segmentation_multires_mesh_info', '1')])
    assert mesh_info['lastmod'] == 10

    num_lods, fragments = _parse_manifest(r.data)
    assert num_lods == 1
    # Every chunk of the sphere's 3x3x3 grid except the (fully interior) center chunk.
    positions = {p for (_, p, _, size) in fragments if size}
    assert positions == {(x, y, z) for x in range(3) for y in range(3) for z in range(3)} - {(1, 1, 1)}
    _fetch_all_fragments(client, fragments)

    # A second manifest request finds the stored mesh up-to-date.
    assert client.get(f'{SOURCE}/1.index').data == r.data
    assert dvid.sparsevol_requests == 27


def test_edit_regenerates_only_affected_chunks(service):
    client, dvid = service
    first = client.get(f'{SOURCE}/1.index').data
    assert dvid.sparsevol_requests == 27

    dvid.edit_chunk([0, 0, 0])
    second = client.get(f'{SOURCE}/1.index')
    assert second.status_code == 200

    # Only the edited chunk was regenerated, and the body mesh was reassembled.
    assert dvid.sparsevol_requests == 28
    assert json.loads(dvid.kv[('segmentation_multires_mesh_info', '1')])['lastmod'] == 11
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
        'badquality': TEST_CONFIG.replace('concatenated-chunks-test-q', 'concatenated-chunks-test-nosuchquality'),
        'singleres': TEST_CONFIG.replace('neuroglancer-multi-res', 'neuroglancer-single-res'),
    }
    for name, text in configs.items():
        (tmp_path / 'configs' / f'{name}.yaml').write_text(text)
        r = client.get(SOURCE.replace('/test', f'/{name}') + '/info')
        assert r.status_code == 404, name

    r = client.get(SOURCE.replace('/test', '/singleres') + '/info')
    assert b'neuroglancer-multi-res' in r.data


def test_unsatisfiable_range(service):
    client, _ = service
    assert client.get(f'{SOURCE}/1.index').status_code == 200
    assert client.get(f'{SOURCE}/1', headers={'Range': 'bytes=999999999-'}).status_code == 416


def _update_body_1(dvid):
    """Call update_body_mesh() for body 1 the way the service does."""
    from neuclease.misc import bodymesh
    from ngsupport import multires_mesh
    config = multires_mesh._load_config('test')
    bodymesh.update_body_mesh('https://fake-dvid', RESOLVED_UUID, 'segmentation', 1,
                              json.loads(json.dumps(config.body)), json.loads(json.dumps(config.chunk)),
                              processes=0, expect_concurrent_updates=True)


def test_concurrent_updates_share_chunks(service):
    """
    Two concurrent updates of the same body (as if on two different hosts) mostly
    divide the chunks between them, rather than each generating all of them.
    """
    client, dvid = service
    dvid.sparsevol_delay = 0.02

    with ThreadPoolExecutor(2) as executor:
        for f in [executor.submit(_update_body_1, dvid) for _ in range(2)]:
            f.result()

    # Without coordination, that would have taken 2 * 27 = 54 chunk generations.
    assert dvid.sparsevol_requests < 40, dvid.sparsevol_requests
    assert len([k for (i, k) in dvid.kv if i == 'segmentation_multires_chunk_meshes']) == 27

    # The stored mesh is complete and usable.
    r = client.get(f'{SOURCE}/1.index')
    _num_lods, fragments = _parse_manifest(r.data)
    _fetch_all_fragments(client, fragments)
    assert dvid.sparsevol_requests < 40


def test_body_mesh_not_stored_if_another_process_finished_it(service):
    """
    If another process stores an up-to-date body mesh while we're generating chunks,
    we don't assemble (or store) our own.
    """
    _client, dvid = service
    method = 'concatenated-chunks-test-q'

    def another_process_finishes():
        dvid.kv[('segmentation_multires_mesh_info', '1')] = \
            json.dumps({'body': 1, 'lastmod': dvid.mutid, 'method': method}).encode()

    dvid.on_sparsevol = another_process_finishes
    _update_body_1(dvid)

    assert dvid.sparsevol_requests == 27
    assert ('segmentation_multires_meshes', '1') not in dvid.posted_keys
    assert ('segmentation_multires_meshes', '1.index') not in dvid.posted_keys
