"""Phase 4 tests — run_analysis module, /api/runs/<scan_id>/analysis,
grid-sidecar sync, and dashboard tabs / Submit-Scan / protocol-viewer
plumbing.

Five test tiers:

1. ``analyze_scan_dir`` against synthetic HDF5 + slm_diag.h5 fixtures
   (no live SLM, no live MATLAB).
2. ``SlmSyncClient.get_grid_sidecar`` against the FakeSlmServer (Phase 4
   endpoint added to the fake).
3. ``sync_scan`` writes ``slm_grid.json`` alongside ``slm_diag.h5``
   when the SLM PC exposes the grid sidecar.
4. Dashboard endpoint ``/api/runs/<scan_id>/analysis`` (success + 400 +
   404) via Flask test client.
5. Dashboard endpoint ``/api/runs/<scan_id>/grid`` (sidecar-first, then
   live SLM passthrough).

The 4-tabs layout itself is exercised by importing ``_build_app`` and
asserting the layout includes the expected component IDs — full UI
testing belongs to selenium / playwright which we don't carry here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Tier 1 — analyze_scan_dir against synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_scan(scan_dir: Path, *, n_sites=4, n_params=5, n_reps=20,
                        with_diag=True, with_code=True, with_grid=False):
    """Build a minimal HDF5 + .mat + (optional) slm_diag.h5 / slm_code.json
    / slm_grid.json under scan_dir so ``analyze_scan_dir`` has something
    to load. Returns the constructed scan_id (last 14 digits of the
    folder name)."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    name = scan_dir.name        # data_YYYYMMDD_HHMMSS

    # Synthesize logicals: rep i lights site (i % n_sites). Two images per
    # shot (NumImages=2) -> survival is exactly 50% for sites 0..n_sites-1.
    num_images = 2
    n_seq = n_params * n_reps
    n_frames = n_seq * num_images

    rng = np.random.default_rng(seed=42)
    intensities = rng.standard_normal((n_frames, n_sites)).astype(np.float64)
    logicals = (rng.random((n_frames, n_sites)) > 0.5)

    # Scan struct (.mat) - write a v7.3 (HDF5) .mat so the lab-PC
    # ``load_scan_config_from_mat`` path takes the h5py branch and
    # unpacks the Scan group as production files do.
    with h5py.File(scan_dir / f'{name}.mat', 'w') as f:
        scan_grp = f.create_group('Scan')
        scan_grp.create_dataset('NumImages',   data=np.array([[num_images]]))
        scan_grp.create_dataset('NumPerGroup', data=np.array([[n_seq]]))
        scan_grp.create_dataset('Params',
            data=np.tile(np.arange(1, n_params + 1), n_reps))
        scan_grp.create_dataset('roi',         data=np.array([[0, 0, 16, 16]]))
        scan_grp.create_dataset('frameSize',   data=np.array([[16, 16]]))
        scan_grp.create_dataset('isHC',        data=np.array([[0]]))
        scan_grp.create_dataset('isInit',      data=np.array([[0]]))

    with h5py.File(scan_dir / f'{name}.h5', 'w') as f:
        f.attrs['two_array'] = False
        f.create_dataset('logicals', data=logicals)
        f.create_dataset('intensities', data=intensities)
        f.create_dataset('seq_ids', data=np.arange(1, n_seq + 1, dtype=np.int64))
        # Pass scan_config attrs so load_scan_from_path picks them up.
        scan_grp = f.create_group('scan_config')
        scan_grp.attrs['NumImages'] = num_images
        scan_grp.attrs['NumPerGroup'] = n_seq

    scan_id = name.replace('data_', '').replace('_', '')

    if with_diag:
        # Phase 2 schema: one row per shot.
        with h5py.File(scan_dir / 'slm_diag.h5', 'w') as f:
            g = f.create_group('diag')
            g.create_dataset('seq_id',
                             data=np.arange(1, n_seq + 1, dtype=np.int64))
            g.create_dataset('total_ms',
                             data=np.full(n_seq, 25.0, dtype=np.float64))
            g.create_dataset('n_loaded',
                             data=np.full(n_seq, 100, dtype=np.float64))
            g.create_dataset('n_dropped',
                             data=np.zeros(n_seq, dtype=np.float64))
            g.create_dataset('aborted',
                             data=np.zeros(n_seq, dtype=np.uint8))
            # diag_json vlen-string column carrying the full row JSON.
            vlen = h5py.string_dtype()
            blob = np.array(
                [json.dumps({'seq_id': i + 1, 'total_ms': 25.0,
                             'n_loaded': 100, 'n_dropped': 0,
                             'aborted': False})
                 for i in range(n_seq)], dtype=object)
            g.create_dataset('diag_json', data=blob, dtype=vlen)

    if with_code:
        (scan_dir / 'slm_code.json').write_text(json.dumps({
            'scan_id': scan_id,
            'safe_run_id': f'fake_sid_{scan_id}',
            'manifest_path': f'fake/path/{scan_id}/manifest.json',
            'manifest': {'files': [
                {'src_rel': 'a.py', 'sha256': 'a' * 64},
                {'src_rel': 'b.py', 'sha256': 'b' * 64},
            ]},
        }), encoding='utf-8')

    if with_grid:
        (scan_dir / 'slm_grid.json').write_text(json.dumps({
            'schema':        'rearrange_grid_sidecar_v1',
            'init_grid':     [[0, 0]] * 12,
            'target_grid':   [[0, 0]] * 12,
            'grid_rotation': 0.0,
        }), encoding='utf-8')

    return scan_id


def test_analyze_scan_dir_basic(tmp_path):
    """Synthetic scan yields a JSON-safe result with the expected fields."""
    scan_dir = tmp_path / 'data_20260601_120000'
    _make_synthetic_scan(scan_dir)

    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    result = analyze_scan_dir(str(scan_dir))

    assert result['scan_id'] == '20260601120000'
    assert result['n_params'] == 5
    assert result['n_shots'] > 0
    assert 'summary' in result
    assert 'sweep' in result
    assert isinstance(result['summary']['survival_mean'], list)


def test_analyze_scan_dir_diag_aggregate(tmp_path):
    scan_dir = tmp_path / 'data_20260601_130000'
    _make_synthetic_scan(scan_dir, with_diag=True)

    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    result = analyze_scan_dir(str(scan_dir))

    diag = result['diag_aggregate']
    assert diag is not None
    assert diag['n_rows'] == 5 * 20
    assert diag['mean_total_ms'] == pytest.approx(25.0)
    assert diag['mean_n_loaded'] == pytest.approx(100.0)
    assert diag['aborted_count'] == 0


def test_analyze_scan_dir_no_diag(tmp_path):
    """Missing slm_diag.h5 should not break the analysis path."""
    scan_dir = tmp_path / 'data_20260601_140000'
    _make_synthetic_scan(scan_dir, with_diag=False)

    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    result = analyze_scan_dir(str(scan_dir))
    assert result['diag_aggregate'] is None
    assert result['summary'] is not None   # still computes survival/loading


def test_analyze_scan_dir_code_pointer(tmp_path):
    scan_dir = tmp_path / 'data_20260601_150000'
    _make_synthetic_scan(scan_dir, with_code=True)

    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    result = analyze_scan_dir(str(scan_dir))
    assert result['code']['present'] is True
    assert result['code']['n_files'] == 2


def test_analyze_scan_dir_grid_pointer(tmp_path):
    scan_dir = tmp_path / 'data_20260601_160000'
    _make_synthetic_scan(scan_dir, with_grid=True)

    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    result = analyze_scan_dir(str(scan_dir))
    assert result['grid']['present'] is True
    assert result['grid']['n_sites'] == 12
    assert result['grid']['grid_rotation'] == 0.0


def test_analyze_scan_dir_invalid_path(tmp_path):
    from yb_analysis.analysis.run_analysis import (
        analyze_scan_dir, RunAnalysisError)
    with pytest.raises(RunAnalysisError):
        analyze_scan_dir(tmp_path / 'does_not_exist')


def test_analyze_scan_invalid_scan_id():
    from yb_analysis.analysis.run_analysis import (
        analyze_scan, RunAnalysisError)
    with pytest.raises(RunAnalysisError):
        analyze_scan('not_14_digits')


def test_analyze_scan_unknown_scan_id():
    """A well-formed but unknown scan_id yields a clear error."""
    from yb_analysis.analysis.run_analysis import (
        analyze_scan, RunAnalysisError)
    with pytest.raises(RunAnalysisError, match='could not find'):
        analyze_scan('99991231235959')


def test_analyze_result_json_safe(tmp_path):
    """The result must be json.dumps-able without TypeError -- the
    dashboard endpoint will jsonify() it."""
    scan_dir = tmp_path / 'data_20260601_170000'
    _make_synthetic_scan(scan_dir)
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    result = analyze_scan_dir(str(scan_dir))
    json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# Tier 2 — SlmSyncClient.get_grid_sidecar against the fake server
# ---------------------------------------------------------------------------

def _add_grid_sidecar_to_fake(fake, scan_id, payload):
    """Inject a grid-sidecar entry into the FakeSlmServer."""
    if not hasattr(fake, '_grid_sidecars'):
        fake._grid_sidecars = {}
        # Hand-register the route since this method existed for prior phases.
        from flask import jsonify

        @fake._app.route('/slm/runs/<scan_id>/grid_sidecar',
                         methods=['GET'])
        def _grid(scan_id):       # noqa: F811
            with fake._lock:
                payload = fake._grid_sidecars.get(str(scan_id))
            if payload is None:
                return jsonify({'detail':
                                f'no grid sidecar for scan_id={scan_id}'}), 404
            return jsonify(payload)
    fake._grid_sidecars[str(scan_id)] = dict(payload)


def test_get_grid_sidecar_success():
    from yb_analysis.slm_sync import SlmSyncClient
    from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer
    with FakeSlmServer() as fake:
        _add_grid_sidecar_to_fake(fake, 'SCAN_X', {
            'schema': 'rearrange_grid_sidecar_v1',
            'init_grid': [[1, 2], [3, 4]],
            'grid_rotation': 1.5,
        })
        client = SlmSyncClient(slm_url=fake.url)
        r = client.get_grid_sidecar('SCAN_X')
        assert r is not None
        assert r['schema'] == 'rearrange_grid_sidecar_v1'
        assert r['grid_rotation'] == 1.5


def test_get_grid_sidecar_missing_returns_none():
    from yb_analysis.slm_sync import SlmSyncClient
    from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer
    with FakeSlmServer() as fake:
        _add_grid_sidecar_to_fake(fake, 'SCAN_OTHER', {'schema': 'x'})
        client = SlmSyncClient(slm_url=fake.url)
        assert client.get_grid_sidecar('UNKNOWN') is None


def test_get_grid_sidecar_endpoint_missing_returns_none():
    """When the SLM build predates the grid_sidecar endpoint, 404 ->
    None (no exception)."""
    from yb_analysis.slm_sync import SlmSyncClient
    from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer
    with FakeSlmServer() as fake:
        # No grid endpoint registered on this fake server (the helper
        # above is only called when test_get_grid_sidecar_success runs).
        client = SlmSyncClient(slm_url=fake.url)
        assert client.get_grid_sidecar('ANY') is None


# ---------------------------------------------------------------------------
# Tier 3 — sync_scan writes slm_grid.json when available
# ---------------------------------------------------------------------------

def test_sync_scan_writes_grid_json(tmp_path):
    from yb_analysis.slm_sync.sync import sync_scan
    from yb_analysis.slm_sync import SlmSyncClient
    from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer
    with FakeSlmServer() as fake:
        # diag (so sync has something to write) + grid sidecar
        fake.add_diag_row('SCAN_GRID', 1, {'total_ms': 12.0})
        _add_grid_sidecar_to_fake(fake, 'SCAN_GRID', {
            'schema': 'rearrange_grid_sidecar_v1',
            'init_grid': [[0, 0]],
            'grid_rotation': 0.0,
        })
        client = SlmSyncClient(slm_url=fake.url)
        scan_dir = tmp_path / 'data_20260601_180000'
        status = sync_scan('SCAN_GRID', str(scan_dir), client=client,
                           sync_code=False, sync_grid=True)

        assert status['synced']
        assert status['grid_path'] is not None
        grid_path = Path(status['grid_path'])
        assert grid_path.exists()
        payload = json.loads(grid_path.read_text(encoding='utf-8'))
        assert payload['schema'] == 'rearrange_grid_sidecar_v1'
        # synced_at_iso added by sync.py
        assert 'synced_at_iso' in payload


def test_sync_scan_grid_endpoint_missing_no_crash(tmp_path):
    """If the SLM build doesn't expose grid_sidecar, sync_scan still
    completes; status['grid_path'] is None."""
    from yb_analysis.slm_sync.sync import sync_scan
    from yb_analysis.slm_sync import SlmSyncClient
    from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer
    with FakeSlmServer() as fake:
        fake.add_diag_row('SCAN_NO_GRID', 1, {'total_ms': 10.0})
        # No grid endpoint registered.
        client = SlmSyncClient(slm_url=fake.url)
        scan_dir = tmp_path / 'data_20260601_190000'
        status = sync_scan('SCAN_NO_GRID', str(scan_dir), client=client,
                           sync_code=False, sync_grid=True)
        assert status['synced']     # diag wrote
        assert status['grid_path'] is None


# ---------------------------------------------------------------------------
# Tier 4 — dashboard endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def dash_app(tmp_path, monkeypatch):
    """Build the dashboard's Flask app standalone, pointing both
    ``DATA_DIR`` (used by run_analysis) and ``PATH_PREFIX`` (used by
    the dashboard's scan_dir resolver) at the tmp path so synthetic
    scans are findable by both code paths."""
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, 'DATA_DIR', str(tmp_path / 'Data'))
    monkeypatch.setattr(yb_cfg, 'PATH_PREFIX', str(tmp_path))
    # scan_directory caches PATH_PREFIX at import time; patch its
    # module-level reference too.
    from yb_analysis.io import scan_directory as sd_mod
    monkeypatch.setattr(sd_mod, 'PATH_PREFIX', str(tmp_path))

    from yb_analysis.plotting import dashboard as dash_mod
    from flask import Flask
    flask_app = Flask('phase4_dash_test')
    dash_mod._register_api_routes(flask_app)
    flask_app.testing = True
    return flask_app.test_client()


def test_runs_analysis_endpoint_success(dash_app, tmp_path):
    """A real synthetic scan flows through /api/runs/<scan_id>/analysis."""
    day = tmp_path / 'Data' / '20260601'
    scan_dir = day / 'data_20260601_200000'
    _make_synthetic_scan(scan_dir)
    r = dash_app.get('/api/runs/20260601200000/analysis')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['scan_id'] == '20260601200000'
    assert body['n_params'] == 5
    assert 'summary' in body


def test_runs_analysis_endpoint_bad_scan_id(dash_app):
    r = dash_app.get('/api/runs/abc/analysis')
    assert r.status_code == 400


def test_runs_analysis_endpoint_unknown_scan_id(dash_app):
    r = dash_app.get('/api/runs/99991231235959/analysis')
    assert r.status_code == 404


def test_runs_grid_endpoint_sidecar_first(dash_app, tmp_path):
    """The dashboard returns the local slm_grid.json without contacting SLM."""
    day = tmp_path / 'Data' / '20260601'
    scan_dir = day / 'data_20260601_210000'
    _make_synthetic_scan(scan_dir, with_grid=True)
    # No fake SLM running -- if dashboard tries to passthrough we'd see
    # a connect error / 503. Should not happen because sidecar is local.
    r = dash_app.get('/api/runs/20260601210000/grid')
    assert r.status_code == 200
    body = r.get_json()
    assert body['schema'] == 'rearrange_grid_sidecar_v1'


def test_runs_grid_endpoint_no_sidecar_no_slm(dash_app, monkeypatch, tmp_path):
    """Missing sidecar + SLM unreachable -> 503 or 404 (the passthrough
    surfaces 503/404 cleanly; either is acceptable here)."""
    day = tmp_path / 'Data' / '20260601'
    scan_dir = day / 'data_20260601_220000'
    _make_synthetic_scan(scan_dir, with_grid=False)
    # Point the SLM client at a never-listening port so the passthrough
    # fails fast. Wire via the SLM_URL config knob.
    import socket
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    bad_port = s.getsockname()[1]
    s.close()
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, 'SLM_URL', f'http://127.0.0.1:{bad_port}')

    r = dash_app.get('/api/runs/20260601220000/grid')
    assert r.status_code in (404, 503), r.get_data(as_text=True)


def test_endpoints_index_lists_phase4_routes(dash_app):
    r = dash_app.get('/api/endpoints')
    assert r.status_code == 200
    paths = {e['path'] for e in r.get_json()['endpoints']}
    assert '/api/runs/<scan_id>/analysis' in paths
    assert '/api/runs/<scan_id>/grid' in paths


# ---------------------------------------------------------------------------
# Tier 5 — dashboard layout / Submit-Scan form smoke check
# ---------------------------------------------------------------------------

def test_build_app_has_phase4_tabs():
    """_build_app produces a Dash app whose layout contains the new tab
    structure + Phase 4 component IDs. We don't render the page; we
    inspect the component tree.
    """
    from yb_analysis.plotting.dashboard import _build_app
    app = _build_app()
    # Walk the layout collecting component IDs.
    found_ids = set()

    def _walk(comp):
        if comp is None:
            return
        if hasattr(comp, 'id') and getattr(comp, 'id', None):
            found_ids.add(comp.id)
        children = getattr(comp, 'children', None)
        if children is None:
            return
        if isinstance(children, (list, tuple)):
            for c in children:
                _walk(c)
        else:
            _walk(children)

    _walk(app.layout)

    expected = {
        # Phase 4 additions:
        'main-tabs', 'tab-url',
        'analysis-scan-id', 'analysis-load-btn',
        'analysis-summary', 'analysis-survival',
        'analysis-loading', 'analysis-protocol-src',
        'analysis-result-store',
        'submit-scan-json', 'submit-scan-btn', 'submit-scan-result',
        # Existing IDs must still exist after the refactor:
        'array', 'array2', 'intens', 'scan', 'slm-panel', 'queue-panel',
        'site-dd', 'tick',
    }
    missing = expected - found_ids
    assert not missing, f'Missing component IDs after refactor: {missing}'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
