"""Phase 4b tests — new SLM-styled HTML dashboard, seq catalog, runs
list, run groups, group analysis, and SLM-side passthrough endpoints.

Six tiers:

1. seq_catalog: introspects YbSeqs/YbSteps and surfaces a useful
   catalog with paths and default expressions.
2. run_groups: create/list/add/remove/delete cycle persists across
   reads, supports concurrent access (single-threaded under lock).
3. runs_list: walks PATH_PREFIX/Data and returns scans with meta.
4. /api/seqs/list, /api/seqs/<name>, /api/seqs/refresh endpoints.
5. /api/runs/list, /api/runs/groups* endpoints.
6. The new HTML dashboard at / serves the template + CSS + JS, and
   Dash is reachable at /old/ (legacy live page).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

# The Tier-1 catalog tests need the MATLAB Seq/Step library (introspected by
# seq_catalog). Absent in a standalone yb_analysis checkout — skip rather than
# fail on an empty catalog.
from yb_analysis.scans.seq_catalog import _matlab_root  # noqa: E402

_skip_no_matlab = pytest.mark.skipif(
    _matlab_root() is None,
    reason="MATLAB Seq/Step library absent (matlab_new tree not alongside)")


# ---------------------------------------------------------------------------
# Tier 1 — seq_catalog
# ---------------------------------------------------------------------------

@_skip_no_matlab
def test_seq_catalog_lists_yb_seqs():
    from yb_analysis.scans.seq_catalog import list_seqs, invalidate_cache
    invalidate_cache()
    seqs = list_seqs()
    assert len(seqs) > 0, 'expected at least one Seq function discovered'
    names = {s['name'] for s in seqs}
    # Should pick up the canonical sequences shipped in the MATLAB YbSeqs library.
    assert 'CoolingOptimizationSeq' in names or 'BlueTweezerLoadingSeq' in names


@_skip_no_matlab
def test_seq_catalog_step_introspection():
    from yb_analysis.scans.seq_catalog import list_steps
    steps = list_steps()
    assert any(s['name'] == 'Cool556Step' for s in steps), \
        'Cool556Step should be in the steps catalog'


@_skip_no_matlab
def test_seq_catalog_get_seq_has_params():
    """Pick a known seq and assert its catalog entry has params with
    paths like 'Namespace.field' and non-empty defaults."""
    from yb_analysis.scans.seq_catalog import get_seq, list_seqs
    candidates = list_seqs()
    # Find any seq that has at least one step
    name = next((s['name'] for s in candidates if s['n_steps'] > 0), None)
    assert name is not None, 'expected at least one seq with steps'
    entry = get_seq(name)
    assert entry['name'] == name
    assert isinstance(entry['steps'], list)
    assert isinstance(entry['params'], list)
    assert isinstance(entry['runp'], list)
    assert entry['runp'][0]['field'] == 'NumPerGroup'


def test_seq_catalog_refresh_bumps_version():
    from yb_analysis.scans import seq_catalog as sc
    sc.invalidate_cache()
    v1 = sc.cache_version()
    sc.list_seqs()           # populate
    sc.invalidate_cache()
    v2 = sc.cache_version()
    assert v2 == v1 + 1


# ---------------------------------------------------------------------------
# Tier 2 — run_groups
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_groups_dir(tmp_path, monkeypatch):
    """Point run_groups at a tmp dir."""
    monkeypatch.setenv('YB_RUN_GROUPS_DIR', str(tmp_path))
    yield tmp_path


def test_run_groups_create_and_list(isolated_groups_dir):
    from yb_analysis.analysis import run_groups
    gid = run_groups.create_group('blue lac batch')
    assert isinstance(gid, str) and gid
    groups = run_groups.list_groups()
    assert gid in groups
    assert groups[gid]['name'] == 'blue lac batch'
    assert groups[gid]['members'] == []


def test_run_groups_create_requires_name(isolated_groups_dir):
    from yb_analysis.analysis import run_groups
    with pytest.raises(ValueError):
        run_groups.create_group('   ')


def test_run_groups_add_remove_member(isolated_groups_dir):
    from yb_analysis.analysis import run_groups
    gid = run_groups.create_group('t')
    assert run_groups.add_member(gid, '20260529025015')
    assert run_groups.add_member(gid, '20260529030000')
    # Re-add is idempotent
    assert run_groups.add_member(gid, '20260529025015')
    g = run_groups.get_group(gid)
    assert len(g['members']) == 2
    assert run_groups.remove_member(gid, '20260529025015')
    assert len(run_groups.get_group(gid)['members']) == 1
    assert not run_groups.remove_member(gid, 'not-present')


def test_run_groups_delete(isolated_groups_dir):
    from yb_analysis.analysis import run_groups
    gid = run_groups.create_group('to-delete')
    assert run_groups.delete_group(gid)
    assert run_groups.get_group(gid) is None
    assert not run_groups.delete_group(gid)


def test_run_groups_rename(isolated_groups_dir):
    from yb_analysis.analysis import run_groups
    gid = run_groups.create_group('old name')
    assert run_groups.rename_group(gid, 'new name')
    assert run_groups.get_group(gid)['name'] == 'new name'


def test_run_groups_persistence_across_reads(isolated_groups_dir):
    """Two get_group calls see the same data after a write."""
    from yb_analysis.analysis import run_groups
    gid = run_groups.create_group('p')
    run_groups.add_member(gid, 'X')
    # Simulate restart: force-reread from disk by clearing module state
    # is unnecessary -- the module always reads from disk under lock.
    g = run_groups.get_group(gid)
    assert g['members'][0]['scan_id'] == 'X'


# ---------------------------------------------------------------------------
# Tier 3 — runs_list
# ---------------------------------------------------------------------------

def _make_dummy_scan(base, day, time_):
    import h5py
    import numpy as np
    sd = base / 'Data' / day / f'data_{day}_{time_}'
    sd.mkdir(parents=True, exist_ok=True)
    name = sd.name
    # Empty HDF5 + .mat skeletons so runs_list._enrich_meta has something
    with h5py.File(sd / f'{name}.h5', 'w') as f:
        f.create_dataset('seq_ids', data=np.arange(1, 6, dtype=np.int64))
    with h5py.File(sd / f'{name}.mat', 'w') as f:
        g = f.create_group('Scan')
        g.create_dataset('NumImages',   data=np.array([[2]]))
        g.create_dataset('NumPerGroup', data=np.array([[5]]))
    return sd


def test_runs_list_walks_data_dir(tmp_path, monkeypatch):
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, 'PATH_PREFIX', str(tmp_path))
    _make_dummy_scan(tmp_path, '20260530', '120000')
    _make_dummy_scan(tmp_path, '20260531', '101010')

    from yb_analysis.analysis.runs_list import list_runs
    rows = list_runs()
    ids = sorted(r['scan_id'] for r in rows)
    assert ids == ['20260530120000', '20260531101010']
    # newest-first ordering
    assert rows[0]['scan_id'] == '20260531101010'


def test_runs_list_max_count(tmp_path, monkeypatch):
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, 'PATH_PREFIX', str(tmp_path))
    for i in range(5):
        _make_dummy_scan(tmp_path, '20260601', f'12000{i}')
    from yb_analysis.analysis.runs_list import list_runs
    rows = list_runs(max_count=3)
    assert len(rows) == 3


def test_runs_list_sidecar_flags(tmp_path, monkeypatch):
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, 'PATH_PREFIX', str(tmp_path))
    sd = _make_dummy_scan(tmp_path, '20260601', '121212')
    (sd / 'slm_diag.h5').write_text('')
    (sd / 'slm_code.json').write_text('{}')
    from yb_analysis.analysis.runs_list import list_runs
    rows = list_runs()
    assert rows[0]['has_diag'] is True
    assert rows[0]['has_code'] is True
    assert rows[0]['has_grid'] is False


# ---------------------------------------------------------------------------
# Tier 4 — /api/seqs endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def dash_app(tmp_path, monkeypatch):
    """Standalone Flask test client mirroring the real dashboard."""
    monkeypatch.setenv('YB_RUN_GROUPS_DIR', str(tmp_path / 'groups'))
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, 'PATH_PREFIX', str(tmp_path))
    monkeypatch.setattr(yb_cfg, 'DATA_DIR', str(tmp_path / 'Data'))
    from yb_analysis.io import scan_directory as sd_mod
    monkeypatch.setattr(sd_mod, 'PATH_PREFIX', str(tmp_path))

    from yb_analysis.plotting import dashboard as dash_mod
    from flask import Flask
    flask_app = Flask('phase4b_test')
    dash_mod._register_api_routes(flask_app)
    dash_mod._register_main_html_routes(flask_app)
    flask_app.testing = True
    return flask_app.test_client()


@_skip_no_matlab
def test_api_seqs_list(dash_app):
    from yb_analysis.scans.seq_catalog import invalidate_cache
    invalidate_cache()
    r = dash_app.get('/api/seqs/list')
    assert r.status_code == 200
    body = r.get_json()
    assert 'seqs' in body
    assert isinstance(body['seqs'], list)
    assert len(body['seqs']) > 0


def test_api_seqs_get_unknown(dash_app):
    r = dash_app.get('/api/seqs/DefinitelyNotASeq_Phase4b')
    assert r.status_code == 404


def test_api_seqs_refresh(dash_app):
    r = dash_app.post('/api/seqs/refresh')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True


@_skip_no_matlab
def test_api_seqs_get_includes_runp(dash_app):
    """Any cataloged seq returns a runp block listing the standard
    NumPerGroup / NumImages / Scramble defaults."""
    from yb_analysis.scans.seq_catalog import list_seqs
    name = next((s['name'] for s in list_seqs() if s['n_steps'] > 0), None)
    assert name is not None
    r = dash_app.get(f'/api/seqs/{name}')
    assert r.status_code == 200
    body = r.get_json()
    fields = {p['field'] for p in (body.get('runp') or [])}
    assert {'NumPerGroup', 'NumImages', 'Scramble'} <= fields


# ---------------------------------------------------------------------------
# Tier 5 — /api/runs/list, /api/runs/groups*
# ---------------------------------------------------------------------------

def test_api_runs_list_empty(dash_app):
    """No Data/ dir -> empty result, not 500."""
    r = dash_app.get('/api/runs/list')
    assert r.status_code == 200
    assert r.get_json()['runs'] == []


def test_api_runs_groups_lifecycle(dash_app):
    # List initially empty
    r = dash_app.get('/api/runs/groups')
    assert r.status_code == 200
    assert r.get_json()['groups'] == {}

    # Create
    r = dash_app.post('/api/runs/groups',
                      json={'name': 'lifecycle test'})
    assert r.status_code == 200
    body = r.get_json()
    gid = body['group_id']
    assert isinstance(gid, str) and gid

    # List shows it
    r = dash_app.get('/api/runs/groups')
    assert gid in r.get_json()['groups']

    # Add a member
    r = dash_app.post(f'/api/runs/groups/{gid}/add/20260529025015')
    assert r.status_code == 200
    assert r.get_json()['ok'] is True

    # Get group detail
    r = dash_app.get(f'/api/runs/groups/{gid}')
    assert r.status_code == 200
    assert any(m['scan_id'] == '20260529025015'
               for m in r.get_json()['members'])

    # Remove
    r = dash_app.post(f'/api/runs/groups/{gid}/remove/20260529025015')
    assert r.get_json()['ok'] is True

    # Delete
    r = dash_app.delete(f'/api/runs/groups/{gid}')
    assert r.get_json()['ok'] is True
    r = dash_app.delete(f'/api/runs/groups/{gid}')
    assert r.status_code == 404


def test_api_runs_groups_create_empty_name(dash_app):
    r = dash_app.post('/api/runs/groups', json={'name': ''})
    assert r.status_code == 400


def test_api_runs_group_analysis_empty(dash_app):
    r = dash_app.post('/api/runs/groups', json={'name': 'empty'})
    gid = r.get_json()['group_id']
    r = dash_app.get(f'/api/runs/groups/{gid}/analysis')
    assert r.status_code == 400


def test_api_runs_group_analysis_unknown(dash_app):
    r = dash_app.get('/api/runs/groups/nonexistent/analysis')
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tier 6 — HTML dashboard serving + Dash at /old/
# ---------------------------------------------------------------------------

def test_main_html_served_at_root(dash_app):
    r = dash_app.get('/')
    assert r.status_code == 200
    text = r.get_data(as_text=True)
    # Sanity checks: title + tab buttons present.
    assert 'Yb Tweezer Dashboard' in text
    assert 'tab-btn-live' in text
    assert 'tab-btn-hardware' in text
    assert 'tab-btn-analysis' in text
    assert 'tab-btn-queue' in text
    # SLM-styled body classes present.
    assert 'class="card' in text
    # Submit-scan textarea exists.
    assert 'submit-scan-json' in text


def test_main_html_includes_seq_catalog_panel(dash_app):
    r = dash_app.get('/')
    text = r.get_data(as_text=True)
    # Seq catalog panel (the analog to handoff's "Custom protocols").
    assert 'queue-seq-catalog' in text
    assert 'seq-catalog-table' in text
    assert 'fill-template-btn' in text


def test_static_css_served(dash_app):
    r = dash_app.get('/static/dashboard/dashboard.css')
    assert r.status_code == 200
    assert b'--bg:' in r.get_data()    # CSS variable from the SLM palette


def test_static_js_served(dash_app):
    r = dash_app.get('/static/dashboard/dashboard.js')
    assert r.status_code == 200
    assert b'pollLive' in r.get_data()
    assert b'loadSeqCatalog' in r.get_data()


def test_api_live_figures_empty(dash_app):
    """No yb_dash_data.pkl yet -> endpoint still returns 200 with
    'waiting' placeholder figures (not 500)."""
    r = dash_app.get('/api/live/figures')
    assert r.status_code == 200
    body = r.get_json()
    assert 'figures' in body
    # Every panel returns a figure dict (waiting placeholder when no data).
    figures = body['figures']
    for name in ('array', 'array2', 'intens', 'loadlive',
                 'load', 'infid', 'shift', 'scan', 'avghist',
                 'rep0', 'rep1', 'rep2', 'rep3'):
        assert name in figures
        if figures[name] is not None:
            # If a figure was returned, it has the Plotly shape {data, layout}.
            assert isinstance(figures[name], dict)
            assert 'data' in figures[name] or 'layout' in figures[name]


def test_api_live_figures_single(dash_app):
    """?which=<name> returns a single figure (not the wrapper)."""
    r = dash_app.get('/api/live/figures?which=scan')
    assert r.status_code == 200
    body = r.get_json()
    assert 'data' in body or 'layout' in body


def test_api_live_figures_unknown(dash_app):
    r = dash_app.get('/api/live/figures?which=bogus')
    assert r.status_code == 400


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
