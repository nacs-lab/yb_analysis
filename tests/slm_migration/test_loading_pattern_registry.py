"""Tests for the loading-pattern registry + SlmSyncClient POST wrappers.

Covers (no GPU / no real SLM — uses FakeSlmServer):
- SlmSyncClient._post gate-busy (503) retry loop.
- initialize_loading_pattern / write_loading_phase request bodies.
- pattern_registry round-trip (write/get/list/delete), name sanitization,
  $YB_PATTERNS_DIR override.
- fetch_or_refresh_pattern: first call hits server, cache-hit avoids
  network, force=True re-POSTs, SLM-offline falls back to last-known-good.
"""

import pytest

from yb_analysis.slm_sync.client import SlmSyncClient
from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer


@pytest.fixture
def fake():
    with FakeSlmServer() as f:
        yield f


@pytest.fixture
def client(fake):
    # Fast retry so the gate-busy test doesn't sleep 0.3s per attempt.
    return SlmSyncClient(slm_url=fake.url, retry_backoff_s=0.01)


@pytest.fixture
def patterns_dir(tmp_path, monkeypatch):
    d = tmp_path / 'patterns'
    monkeypatch.setenv('YB_PATTERNS_DIR', str(d))
    # Import after env is set; the module reads the env at call time anyway.
    import yb_analysis.analysis.pattern_registry as reg
    return reg


# ---- client _post + wrappers --------------------------------------------

def test_post_gate_busy_retries(fake, client):
    fake.gate_busy('initialize_loading_pattern', 2)  # two 503s then success
    resp = client.initialize_loading_pattern(
        phase_path='phase/base/33x33_uniform.pt', order='col')
    assert resp is not None and resp['ok'] is True
    # 2 gate-busy + 1 success = 3 hits.
    assert fake.hits('initialize_loading_pattern') == 3


def test_initialize_loading_pattern_body(fake, client):
    client.initialize_loading_pattern(
        phase_path='phase/base/33x33_uniform.pt', order='col',
        fft_shape=(4096, 4096), threshold=0.30, name='33x33_uniform')
    body = fake.captured_bodies('initialize_loading_pattern')[-1]
    assert body['phase_filepath'] == 'phase/base/33x33_uniform.pt'
    assert body['order'] == 'col'
    assert body['fft_shape'] == [4096, 4096]
    assert body['threshold'] == 0.30
    assert body['write_to_slm'] is False     # registry never writes
    assert body['name'] == '33x33_uniform'


def test_write_loading_phase_body(fake, client):
    client.write_loading_phase(phase_path='phase/base/x.pt',
                               loading_zernike=[0, 0, 0, 0, -4],
                               block_timeout=2.0)
    body = fake.captured_bodies('write_loading_phase')[-1]
    assert body['phase_filepath'] == 'phase/base/x.pt'
    assert body['loading_zernike'] == [0, 0, 0, 0, -4]


def test_unreachable_returns_none():
    # No server listening on this port → ConnectionError → None.
    c = SlmSyncClient(slm_url='http://127.0.0.1:9', retry_backoff_s=0.01)
    assert c.initialize_loading_pattern(phase_path='x.pt') is None


def test_initialize_phase_not_found_raises(fake, client):
    # A 404 from the loading-pattern endpoint = the phase file doesn't exist on
    # the SLM server (a misspelled name). Distinct from unreachable (-> None) so
    # the dashboard can warn "phase missing" rather than "SLM down".
    from yb_analysis.slm_sync.client import SlmPhaseNotFound
    fake.fail('initialize_loading_pattern', 404)
    with pytest.raises(SlmPhaseNotFound):
        client.initialize_loading_pattern(phase_path='phase/base/typo.pt')


def test_fetch_phase_not_found_propagates(fake, client, patterns_dir):
    # The user's case: a misspelled phase name with no cache -> the registry
    # surfaces SlmPhaseNotFound instead of silently returning None.
    from yb_analysis.slm_sync.client import SlmPhaseNotFound
    reg = patterns_dir
    fake.fail('initialize_loading_pattern', 404)
    with pytest.raises(SlmPhaseNotFound):
        reg.fetch_or_refresh_pattern(
            'typo', base_phase_path='phase/base/typo.pt', client=client)


def test_fetch_phase_not_found_not_masked_by_cache(fake, client, patterns_dir):
    # Even with a last-known-good cache, a now-missing phase is NOT masked --
    # the cached grid belongs to a phase that no longer exists on the server.
    from yb_analysis.slm_sync.client import SlmPhaseNotFound
    reg = patterns_dir
    reg.write_pattern(_record(name='p1'))
    fake.fail('initialize_loading_pattern', 404)
    with pytest.raises(SlmPhaseNotFound):
        reg.fetch_or_refresh_pattern(
            'p1', base_phase_path='phase/base/p1.pt', client=client, force=True)


# ---- registry persistence -----------------------------------------------

def _record(name='p1', n=4):
    return {
        'name': name, 'base_phase_path': f'phase/base/{name}.pt',
        'legacy_zerniked': False, 'baked_zernike': None,
        'base_sha256': 'ab' * 32, 'default_loading_zernike': None,
        'order': 'col', 'fft_shape': [4096, 4096], 'threshold': 0.30,
        'min_dist': None, 'n_sites': n,
        'knm': [[i, i] for i in range(n)], 'phases': [0.1 * i for i in range(n)],
        'lattice': {'rows': [0] * n, 'cols': list(range(n)), 'n_rows': 1,
                    'n_cols': n, 'pitch_x': 1.0, 'pitch_y': 1.0,
                    'row_basis': [1, 0], 'col_basis': [0, 1], 'tilt_deg': 0.0,
                    'n_missing': 0, 'x0': 0.0, 'y0': 0.0},
        'source_endpoint': '/slm/initialize_loading_pattern',
        'created_iso': '2026-06-01T00:00:00', 'updated_iso': '2026-06-01T00:00:00',
    }


def test_registry_roundtrip(patterns_dir):
    reg = patterns_dir
    reg.write_pattern(_record('p1'))
    got = reg.get_pattern('p1')
    assert got['n_sites'] == 4
    assert got['knm'] == [[i, i] for i in range(4)]
    # list is compact (no big arrays)
    lst = reg.list_patterns()
    assert 'p1' in lst
    assert 'knm' not in lst['p1'] and 'phases' not in lst['p1']
    assert lst['p1']['lattice']['n_cols'] == 4
    # delete
    assert reg.delete_pattern('p1') is True
    assert reg.get_pattern('p1') is None
    assert reg.delete_pattern('p1') is False


def test_name_sanitization(patterns_dir):
    reg = patterns_dir
    with pytest.raises(ValueError):
        reg._sanitize_name('')
    # path separators are scrubbed, not allowed to escape the dir
    rec = _record('weird/../name')
    reg.write_pattern(rec)
    # stored under a sanitized dir; retrievable by the SAME raw name
    assert reg.get_pattern('weird/../name') is not None


# ---- fetch_or_refresh ----------------------------------------------------

def test_fetch_or_refresh_cache_hit_avoids_network(patterns_dir, fake, client):
    reg = patterns_dir
    r1 = reg.fetch_or_refresh_pattern(
        '33x33_uniform', base_phase_path='phase/base/33x33_uniform.pt',
        order='col', client=client)
    assert r1 is not None and r1['n_sites'] == 4
    assert fake.hits('initialize_loading_pattern') == 1
    # Same params → served from disk, no second network call.
    r2 = reg.fetch_or_refresh_pattern(
        '33x33_uniform', base_phase_path='phase/base/33x33_uniform.pt',
        order='col', client=client)
    assert r2['n_sites'] == 4
    assert fake.hits('initialize_loading_pattern') == 1
    # force=True re-POSTs.
    reg.fetch_or_refresh_pattern(
        '33x33_uniform', base_phase_path='phase/base/33x33_uniform.pt',
        order='col', client=client, force=True)
    assert fake.hits('initialize_loading_pattern') == 2


def test_fetch_or_refresh_param_change_refetches(patterns_dir, fake, client):
    reg = patterns_dir
    reg.fetch_or_refresh_pattern('p', base_phase_path='phase/base/p.pt',
                                 order='col', client=client)
    assert fake.hits('initialize_loading_pattern') == 1
    # Different threshold → params differ → refetch.
    reg.fetch_or_refresh_pattern('p', base_phase_path='phase/base/p.pt',
                                 order='col', threshold=0.5, client=client)
    assert fake.hits('initialize_loading_pattern') == 2


def test_fetch_or_refresh_offline_fallback(patterns_dir, fake, client):
    reg = patterns_dir
    reg.fetch_or_refresh_pattern('p', base_phase_path='phase/base/p.pt',
                                 order='col', client=client)
    fake.stop()  # SLM now unreachable
    # force=True would normally re-POST, but the call returns None → we keep
    # the last-known-good record instead of crashing.
    got = reg.fetch_or_refresh_pattern(
        'p', base_phase_path='phase/base/p.pt', order='col',
        client=client, force=True)
    assert got is not None and got['n_sites'] == 4


# ---- 3-D loading-pattern support ----------------------------------------

def test_initialize_loading_pattern_body_3d(fake, client):
    """planes_z_rad rides in the request body only when supplied (and a 2-D
    call still omits it entirely)."""
    client.initialize_loading_pattern(
        phase_path='phase/base/two_layer.pt', order='col', name='two_layer',
        planes_z_rad=[-3.07, 3.07])
    body = fake.captured_bodies('initialize_loading_pattern')[-1]
    assert body['planes_z_rad'] == [-3.07, 3.07]
    # A 2-D call must not carry the key at all.
    client.initialize_loading_pattern(
        phase_path='phase/base/flat.pt', order='col', name='flat')
    body2 = fake.captured_bodies('initialize_loading_pattern')[-1]
    assert 'planes_z_rad' not in body2
    # Empty list is treated as "no planes" -> 2-D request.
    client.initialize_loading_pattern(
        phase_path='phase/base/flat.pt', order='col', planes_z_rad=[])
    assert 'planes_z_rad' not in fake.captured_bodies(
        'initialize_loading_pattern')[-1]


def test_fetch_or_refresh_3d_stores_layer_fields(patterns_dir, fake, client):
    """A 3-D fetch persists the layer-major 3-D fields; knm stays (N,2) so 2-D
    consumers are unaffected."""
    reg = patterns_dir
    rec = reg.fetch_or_refresh_pattern(
        'two_layer', base_phase_path='phase/base/two_layer.pt', order='col',
        planes_z_rad=[-3.07, 3.07], client=client)
    assert rec is not None
    assert rec['is_3d'] is True
    assert rec['planes_z_rad'] == [-3.07, 3.07]
    assert rec['n_per_plane'] == [2, 2]            # fake splits 4 sites / 2
    assert rec['plane_of_site'] == [0, 0, 1, 1]
    assert len(rec['z_rad']) == rec['n_sites']
    assert len(rec['positions_knm3d']) == rec['n_sites']
    # knm is still (N,2) — the 2-D detection grid is unchanged.
    assert all(len(p) == 2 for p in rec['knm'])
    # Compact view drops the big 3-D arrays.
    compact = reg.list_patterns()['two_layer']
    for big in ('positions_knm3d', 'z_rad', 'plane_of_site', 'knm', 'phases'):
        assert big not in compact


def test_fetch_or_refresh_planes_are_cache_key(patterns_dir, fake, client):
    """A 2-D cached record must NOT satisfy a 3-D request, and vice versa."""
    reg = patterns_dir
    # 2-D first.
    reg.fetch_or_refresh_pattern('q', base_phase_path='phase/base/q.pt',
                                 order='col', client=client)
    assert fake.hits('initialize_loading_pattern') == 1
    # Same params + planes -> different identity -> re-derive.
    reg.fetch_or_refresh_pattern('q', base_phase_path='phase/base/q.pt',
                                 order='col', planes_z_rad=[-3.07, 3.07],
                                 client=client)
    assert fake.hits('initialize_loading_pattern') == 2
    # Same 3-D params again -> cache hit, no network.
    r = reg.fetch_or_refresh_pattern('q', base_phase_path='phase/base/q.pt',
                                     order='col', planes_z_rad=[-3.07, 3.07],
                                     client=client)
    assert fake.hits('initialize_loading_pattern') == 2
    assert r['is_3d'] is True
    # Different depths -> re-derive.
    reg.fetch_or_refresh_pattern('q', base_phase_path='phase/base/q.pt',
                                 order='col', planes_z_rad=[-5.0, 5.0],
                                 client=client)
    assert fake.hits('initialize_loading_pattern') == 3
    # Back to 2-D -> the 3-D cache must not satisfy it.
    reg.fetch_or_refresh_pattern('q', base_phase_path='phase/base/q.pt',
                                 order='col', client=client)
    assert fake.hits('initialize_loading_pattern') == 4


def test_legacy_2d_record_still_cache_hits(patterns_dir, fake, client):
    """A pre-3-D record (no planes_z_rad key) must still satisfy a 2-D request
    without a needless re-derive after deploy."""
    reg = patterns_dir
    rec = _record('legacy')
    rec.pop('planes_z_rad', None)   # simulate an old on-disk record
    reg.write_pattern(rec)
    got = reg.fetch_or_refresh_pattern(
        'legacy', base_phase_path='phase/base/legacy.pt', order='col',
        client=client)
    assert got is not None
    assert fake.hits('initialize_loading_pattern') == 0   # served from disk
