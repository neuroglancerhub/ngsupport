"""
Tests for the /neuronjson_segment_*properties endpoints (ngsupport/neuronjson_segment_properties.py),
with DVID faked out.
"""
import numpy as np
import pandas as pd
import pytest

ANNOTATIONS = [
    {'bodyid': 10, 'type': 'Mi1', 'instance': 'Mi1_R', 'group': 10, 'type_user': 'alice'},
    {'bodyid': 20, 'type': 'T4', 'group': 20.0},
    {'bodyid': 30, 'instance': 'unknown_R', 'group': 31.5},
    {'bodyid': 40, 'status': 'Anchor'},
]


@pytest.fixture
def client(monkeypatch):
    from ngsupport import neuronjson_segment_properties as nsp
    from ngsupport.app import app

    def fetch_all(server, uuid, instance, *, show=None, fields=None, format='pandas', session=None):
        df = pd.DataFrame(ANNOTATIONS).set_index('bodyid').rename_axis('body')
        return df[[f for f in fields if f in df.columns]]

    top = {
        'PreSyn': pd.Series([5, 4, 3], index=pd.Index([10, 20, 30], name='body'), name='PreSyn'),
        'PostSyn': pd.Series([9, 8, 7], index=pd.Index([20, 30, 40], name='body'), name='PostSyn'),
    }
    counts = {'PreSyn': {40: 2}, 'PostSyn': {10: 1}}

    def fetch_counts(server, uuid, instance, bodies, element_type, format='pandas'):
        bodies = list(bodies)
        return pd.Series([counts[element_type][b] for b in bodies],
                         index=pd.Index(bodies, name='body'), name=element_type, dtype=int)

    monkeypatch.setattr(nsp, 'fetch_all', fetch_all)
    monkeypatch.setattr(nsp, 'fetch_top', lambda server, uuid, instance, n, element_type: top[element_type])
    monkeypatch.setattr(nsp, 'fetch_counts', fetch_counts)
    return app.test_client()


def _props(r):
    assert r.status_code == 200, r.data
    info = r.get_json()
    assert info['@type'] == 'neuroglancer_segment_properties'
    return info['inline']['ids'], {p['id']: p for p in info['inline']['properties']}


def test_convert_to_string():
    from ngsupport.neuronjson_segment_properties import convert_to_string
    assert convert_to_string(pd.Series(['a', None, 'b'])).tolist() == ['a', '', 'b']
    assert convert_to_string(pd.Series([1.0, np.nan, 2.5])).tolist() == ['1', '', '2.5']
    assert convert_to_string(pd.Series([1, 2])).tolist() == ['1', '2']
    assert convert_to_string(pd.Series([1, None], dtype='Int64')).tolist() == ['1', '']
    assert convert_to_string(pd.Series([True, False])).tolist() == ['True', 'False']


def test_label_property(client):
    ids, props = _props(client.get('/neuronjson_segment_properties/fake-dvid/abc123/annotations/type/info'))
    assert props['type']['type'] == 'label'
    assert dict(zip(ids, props['type']['values'])) == {'10': 'Mi1', '20': 'T4'}


def test_label_with_altlabel(client):
    ids, props = _props(client.get('/neuronjson_segment_properties/fake-dvid/abc123/annotations/type/instance/info'))
    values = dict(zip(ids, props['type [instance]']['values']))
    assert values == {'10': 'Mi1 [Mi1_R]', '20': 'T4', '30': ' [unknown_R]'}


def test_numeric_label_property(client):
    """Float columns are shown without a spurious '.0'."""
    ids, props = _props(client.get('/neuronjson_segment_properties/fake-dvid/abc123/annotations/group/info'))
    assert dict(zip(ids, props['group']['values'])) == {'10': '10', '20': '20', '30': '31.5'}


def test_missing_label_field(client):
    r = client.get('/neuronjson_segment_properties/fake-dvid/abc123/annotations/nosuchfield/info')
    ids, _props_ = _props(r)
    assert ids == []


def test_tags_properties(client):
    r = client.get('/neuronjson_segment_tags_properties/fake-dvid/abc123/annotations/status,_has_type/info')
    assert r.status_code == 200, r.data
    tags = r.get_json()['inline']['properties'][0]
    assert tags['type'] == 'tags'
    assert set(tags['tags']) >= {'status:Anchor', '_has_type'}


def test_synapse_properties(client):
    ids, props = _props(client.get('/neuronjson_segment_synapse_properties/fake-dvid/abc123/synapses/3/info'))
    pre = dict(zip(ids, props['PreSyn']['values']))
    post = dict(zip(ids, props['PostSyn']['values']))
    # Counts missing from one 'top' list are filled in via fetch_counts().
    assert pre == {'10': 5, '20': 4, '30': 3, '40': 2}
    assert post == {'10': 1, '20': 9, '30': 8, '40': 7}
