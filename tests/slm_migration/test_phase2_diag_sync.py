"""Phase 2 tests — lab-PC slm_sync package + dashboard runs endpoints.

Three test tiers:

1. **`SlmSyncClient` over a FakeSlmServer** — retry, JSON parsing, error
   surfacing. Runs anywhere.
2. **`sync_scan` end-to-end** — fake server preloaded with rows + a
   manifest; `sync_scan(scan_id, scan_dir)` writes `slm_diag.h5` and
   `slm_code.json`. Tests idempotence + resume + legacy-run marker.
3. **Dashboard `/api/runs/<scan_id>/{diag,code}` passthroughs** — Flask
   test client; sidecar-first then fake-server fallback.
4. **`ondemand.get_protocol_source`** — fetch a known SHA from the
   fake's blob store.
5. **`DataManager._schedule_slm_sync`** — mocks the sync function and
   asserts it's called from the eviction path with the right scan_id.

Run as:
    python -m yb_analysis.tests.slm_migration.test_phase2_diag_sync
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from yb_analysis.slm_sync import SlmSyncClient, sync_scan, get_protocol_source
from yb_analysis.slm_sync.client import _GATE_BUSY_PREFIX  # noqa: F401  (sanity import)
from yb_analysis.slm_sync.sync import (
    DIAG_H5, CODE_JSON, SYNC_STATE, mark_legacy_run,
    sync_scan_async,
)
from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer


# ---------------------------------------------------------------------------
# Tier 1 — SlmSyncClient against a fake server
# ---------------------------------------------------------------------------

def test_client_get_diag_basic():
    """GET /slm/runs/<scan_id>/diag returns the rows in order."""
    with FakeSlmServer() as fake:
        for i in (1, 2, 3):
            fake.add_diag_row('SCAN_A', i, {'total_ms': 10 + i})
        client = SlmSyncClient(slm_url=fake.url)
        r = client.get_diag('SCAN_A')
        assert r is not None
        assert r['scan_id'] == 'SCAN_A'
        assert r['count'] == 3
        assert r['overflow'] is False
        seq_ids = [e['seq_id'] for e in r['entries']]
        assert seq_ids == [1, 2, 3]


def test_client_get_diag_since_seq_id():
    """?since_seq_id=N returns only rows with seq_id > N."""
    with FakeSlmServer() as fake:
        for i in range(10):
            fake.add_diag_row('S', i, {'x': i})
        client = SlmSyncClient(slm_url=fake.url)
        r = client.get_diag('S', since_seq_id=4)
        assert r['count'] == 5
        assert [e['seq_id'] for e in r['entries']] == [5, 6, 7, 8, 9]


def test_client_get_diag_offline():
    """SLM unreachable → get_diag returns None (no exception)."""
    client = SlmSyncClient(slm_url='http://127.0.0.1:1',
                           timeout_s=(0.2, 0.2),
                           max_retries=1)
    assert client.get_diag('whatever') is None


def test_client_get_code_manifest_404():
    """Missing code snapshot returns None (not an exception)."""
    with FakeSlmServer() as fake:
        client = SlmSyncClient(slm_url=fake.url)
        assert client.get_code_manifest('NO_SUCH_SCAN') is None


def test_client_get_code_by_hash_404():
    """Missing blob returns None."""
    with FakeSlmServer() as fake:
        client = SlmSyncClient(slm_url=fake.url)
        # Valid-shape sha that we never registered
        sha = 'a' * 64
        assert client.get_code_by_hash(sha) is None


def test_client_get_code_by_hash_bad_format_raises():
    """Server rejects malformed sha256 with 400; client raises HTTPError."""
    import requests
    with FakeSlmServer() as fake:
        client = SlmSyncClient(slm_url=fake.url)
        with pytest.raises(requests.HTTPError):
            client.get_code_by_hash('not-a-sha')


# ---------------------------------------------------------------------------
# Tier 2 — sync_scan end-to-end
# ---------------------------------------------------------------------------

def test_sync_scan_writes_hdf5(tmp_path):
    """Fake server with 5 rows → `slm_diag.h5` exists with all 5 entries."""
    with FakeSlmServer() as fake:
        for i in range(1, 6):
            fake.add_diag_row('20260529000001', i, {
                'total_ms': 50.0 + i,
                'n_loaded': 100,
                'aborted': False,
            })
        client = SlmSyncClient(slm_url=fake.url)
        status = sync_scan('20260529000001', tmp_path, client=client)

    assert status['synced'] is True
    assert status['reason'] == 'ok'
    assert status['rows_written'] == 5
    assert status['total_rows'] == 5
    assert status['overflow'] is False
    h5_path = tmp_path / DIAG_H5
    assert h5_path.exists()
    with h5py.File(h5_path, 'r') as f:
        seq_ids = f['diag/seq_id'][:].tolist()
        total_ms = f['diag/total_ms'][:].tolist()
        n_loaded = f['diag/n_loaded'][:].tolist()
        aborted = f['diag/aborted'][:].tolist()
        assert seq_ids == [1, 2, 3, 4, 5]
        assert total_ms == [51.0, 52.0, 53.0, 54.0, 55.0]
        assert n_loaded == [100.0] * 5
        assert aborted == [0] * 5
        # diag_json preserves the full row
        first_row = json.loads(f['diag/diag_json'][0])
        assert first_row['seq_id'] == 1
        assert first_row['diag']['total_ms'] == 51.0


def test_sync_scan_idempotent(tmp_path):
    """Running sync twice produces no duplicate rows.

    After the first sync, the resume state records seq_id=5. The second
    sync passes `since_seq_id=5`; the fake returns 0 new rows; nothing
    appended to HDF5.
    """
    with FakeSlmServer() as fake:
        for i in range(1, 6):
            fake.add_diag_row('IDEM', i, {'total_ms': 50.0})
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('IDEM', tmp_path, client=client)
        sync_scan('IDEM', tmp_path, client=client)  # second run is a no-op

    with h5py.File(tmp_path / DIAG_H5, 'r') as f:
        assert f['diag/seq_id'].shape[0] == 5
        assert f['diag/seq_id'][:].tolist() == [1, 2, 3, 4, 5]


def test_sync_scan_resumes(tmp_path):
    """First sync writes 3 rows; server adds 4 more; resume writes 4 more
    (total 7) without duplicating."""
    with FakeSlmServer() as fake:
        for i in range(1, 4):
            fake.add_diag_row('RESUME', i, {'total_ms': 10.0 + i})
        client = SlmSyncClient(slm_url=fake.url)
        s1 = sync_scan('RESUME', tmp_path, client=client)
        assert s1['rows_written'] == 3
        # More shots arrive on the SLM PC.
        for i in range(4, 8):
            fake.add_diag_row('RESUME', i, {'total_ms': 10.0 + i})
        s2 = sync_scan('RESUME', tmp_path, client=client)
        assert s2['rows_written'] == 4
    with h5py.File(tmp_path / DIAG_H5, 'r') as f:
        seq_ids = f['diag/seq_id'][:].tolist()
        assert seq_ids == [1, 2, 3, 4, 5, 6, 7]


def test_sync_scan_no_data(tmp_path):
    """Scan_id the SLM has never seen → synced=False, reason='no_data'.

    No sidecar is written.
    """
    with FakeSlmServer() as fake:
        client = SlmSyncClient(slm_url=fake.url)
        status = sync_scan('NEVER_SEEN', tmp_path, client=client,
                           sync_code=False)
    assert status['synced'] is False
    assert status['reason'] == 'no_data'
    assert not (tmp_path / DIAG_H5).exists()
    assert not (tmp_path / SYNC_STATE).exists()


def test_sync_scan_offline(tmp_path):
    """SLM unreachable → synced=False, reason='slm_offline'."""
    client = SlmSyncClient(slm_url='http://127.0.0.1:1',
                           timeout_s=(0.2, 0.2), max_retries=1)
    status = sync_scan('20260529000099', tmp_path, client=client)
    assert status['synced'] is False
    assert status['reason'] == 'slm_offline'


def test_sync_scan_writes_code_json(tmp_path):
    """Code manifest fetched and written as slm_code.json."""
    manifest = {
        'run_id': '2026-05-29T00:00:00|sid=20260529001',
        'safe_run_id': '2026-05-29T00_00_00_sid_20260529001',
        'files': [
            {'src_rel': 'SLMnet/x.py', 'leaf': 'x.py',
             'sha256': 'a' * 64, 'materialise': 'hardlink'},
        ],
        'git_state': {'commit': 'abc123', 'branch': 'main', 'dirty': False},
    }
    with FakeSlmServer() as fake:
        fake.set_code_manifest('20260529001', manifest)
        # Add one diag row too so synced=True even though code is the
        # interesting bit here.
        fake.add_diag_row('20260529001', 1, {'total_ms': 10.0})
        client = SlmSyncClient(slm_url=fake.url)
        status = sync_scan('20260529001', tmp_path, client=client)

    assert status['code_path'] is not None
    code_file = tmp_path / CODE_JSON
    assert code_file.exists()
    payload = json.loads(code_file.read_text())
    assert payload['manifest']['files'][0]['sha256'] == 'a' * 64
    assert payload['synced_at_iso']


def test_legacy_run_marker(tmp_path):
    """mark_legacy_run drops a sentinel so subsequent syncs short-circuit."""
    # Sentinel is written.
    mark_legacy_run(tmp_path)
    state_path = tmp_path / SYNC_STATE
    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload['reason'] == 'legacy_run'


def test_sync_scan_overflow_flag(tmp_path):
    """Server-reported overflow is captured in the status dict."""
    with FakeSlmServer() as fake:
        for i in range(1, 4):
            fake.add_diag_row('OVR', i, {'total_ms': 1.0})
        fake.set_diag_overflow('OVR', True)
        client = SlmSyncClient(slm_url=fake.url)
        status = sync_scan('OVR', tmp_path, client=client)
    assert status['overflow'] is True


# ---------------------------------------------------------------------------
# Tier 3 — Dashboard /api/runs/<scan_id>/{diag,code} routes
# ---------------------------------------------------------------------------

def test_dashboard_runs_diag_passthrough_when_no_sidecar(monkeypatch):
    """No local slm_diag.h5 → dashboard hits the SLM PC via SlmSyncClient."""
    from yb_analysis.plotting.dashboard import _build_app
    with FakeSlmServer() as fake:
        for i in (1, 2):
            fake.add_diag_row('PASS_THRU', i, {'total_ms': 5.0})
        # Patch SLM_URL so the dashboard's SlmSyncClient hits our fake.
        monkeypatch.setattr('yb_analysis.config.SLM_URL', fake.url)
        client = _build_app().server.test_client()
        # Use a scan_id that won't parse to a real scan_dir
        # (scan_id_to_stamps requires 14-digit format).
        r = client.get('/api/runs/PASS_THRU/diag')
    assert r.status_code == 200
    body = r.get_json()
    assert body['count'] == 2
    assert body['entries'][0]['seq_id'] == 1


def test_dashboard_runs_code_passthrough(monkeypatch):
    """Live code-manifest passthrough."""
    from yb_analysis.plotting.dashboard import _build_app
    with FakeSlmServer() as fake:
        fake.set_code_manifest('PASS_CODE', {
            'run_id': 'fake', 'files': [],
            'git_state': {'commit': 'cafef00d'}})
        monkeypatch.setattr('yb_analysis.config.SLM_URL', fake.url)
        client = _build_app().server.test_client()
        r = client.get('/api/runs/PASS_CODE/code')
    assert r.status_code == 200
    body = r.get_json()
    assert body['scan_id'] == 'PASS_CODE'
    assert body['manifest']['git_state']['commit'] == 'cafef00d'


def test_dashboard_runs_code_404(monkeypatch):
    from yb_analysis.plotting.dashboard import _build_app
    with FakeSlmServer() as fake:
        monkeypatch.setattr('yb_analysis.config.SLM_URL', fake.url)
        client = _build_app().server.test_client()
        r = client.get('/api/runs/NEVER/code')
    assert r.status_code == 404


def test_endpoint_index_lists_runs_routes():
    """/api/endpoints reports the new Phase 2 routes."""
    from yb_analysis.plotting.dashboard import _build_app
    client = _build_app().server.test_client()
    r = client.get('/api/endpoints')
    paths = [e['path'] for e in r.get_json()['endpoints']]
    # Flask url_map renders <scan_id> with angle brackets.
    assert any('/api/runs/' in p and 'diag' in p for p in paths)
    assert any('/api/runs/' in p and 'code' in p for p in paths)


# ---------------------------------------------------------------------------
# Tier 4 — ondemand.get_protocol_source
# ---------------------------------------------------------------------------

def test_get_protocol_source_via_remote(monkeypatch):
    """get_protocol_source fetches manifest + blob purely from the SLM."""
    sha = 'b' * 64
    src = 'def hello(): return 42\n'
    with FakeSlmServer() as fake:
        fake.set_code_manifest('SRC_REMOTE', {
            'files': [{'src_rel': 'SLMnet/src/slmnet/experimental/'
                                  'tools/rearrange_protocols.py',
                       'leaf': 'rearrange_protocols.py',
                       'sha256': sha, 'materialise': 'hardlink'}],
        })
        fake.add_blob(sha, src)
        client = SlmSyncClient(slm_url=fake.url)
        result = get_protocol_source('SRC_REMOTE', client=client)
    assert result == src


def test_get_protocol_source_via_local_sidecar(tmp_path):
    """If `slm_code.json` exists locally, the manifest is read from there
    (no network call)."""
    sha = 'c' * 64
    src = '# local source\n'
    # Pre-stage the sidecar.
    code_sidecar = tmp_path / CODE_JSON
    code_sidecar.write_text(json.dumps({
        'manifest': {
            'files': [{'src_rel': 'SLMnet/src/slmnet/experimental/'
                                  'tools/rearrange_protocols.py',
                       'sha256': sha}],
        },
    }))
    # Fake server only knows about the blob, not the manifest — confirms
    # we used the local sidecar for the manifest step.
    with FakeSlmServer() as fake:
        fake.add_blob(sha, src)
        client = SlmSyncClient(slm_url=fake.url)
        result = get_protocol_source(
            'LOCAL_MANIFEST', scan_dir=tmp_path, client=client)
    assert result == src


def test_get_protocol_source_missing_returns_none():
    """No manifest, no blob → returns None instead of raising."""
    with FakeSlmServer() as fake:
        client = SlmSyncClient(slm_url=fake.url)
        result = get_protocol_source('GHOST', client=client)
    assert result is None


# ---------------------------------------------------------------------------
# Tier 5 — DataManager._schedule_slm_sync hook
# ---------------------------------------------------------------------------

def test_data_manager_sync_hook_calls_sync_scan_async(monkeypatch, tmp_path):
    """`get_data_manager(new_scan_id)` evicts the previous DataManager
    and triggers `_schedule_slm_sync` on it, which calls
    `sync_scan_async(old_scan_id, scan_dir)`."""
    from yb_analysis.acquisition import data_manager as dm_mod

    # Use a fake DataManager so we don't pull in the heavy MATLAB-side
    # config-load path. We monkeypatch the class to a stub for this test.
    calls = []

    class _StubDM:
        sync_after_finish = True
        def __init__(self, scan_id):
            self.scan_id = scan_id
            self.fname = str(tmp_path / f'data_{scan_id}.h5')
            self._file_created = True
        def save_data(self):
            pass
        def _schedule_slm_sync(self):
            # Real implementation: import slm_sync.sync and call sync_scan_async.
            # We intercept to record the (scan_id, scan_dir) pair.
            scan_dir = os.path.dirname(self.fname)
            calls.append((self.scan_id, scan_dir))

    # Force a clean cache.
    dm_mod.drop_all()
    monkeypatch.setattr(dm_mod, 'DataManager', _StubDM)

    dm_mod.get_data_manager(20260101000001)
    assert calls == []  # first DM created; nothing to evict
    # New scan_id triggers eviction of the previous one.
    dm_mod.get_data_manager(20260101000002)
    assert len(calls) == 1
    evicted_id, evicted_dir = calls[0]
    assert evicted_id == 20260101000001
    # The scan_dir is the directory of the evicted DM's fname.
    assert os.path.basename(evicted_dir) == os.path.basename(str(tmp_path))
    # Cleanup
    dm_mod.drop_all()


def test_sync_scan_async_runs_in_background(tmp_path):
    """sync_scan_async returns immediately + completes in the background."""
    with FakeSlmServer() as fake:
        fake.add_diag_row('BG', 1, {'total_ms': 1.0})
        from yb_analysis import config
        client = SlmSyncClient(slm_url=fake.url)
        # Patch sync_scan to use OUR client.
        from yb_analysis.slm_sync import sync as sync_mod
        original = sync_mod.sync_scan

        def _wrapped(*a, **kw):
            kw.setdefault('client', client)
            return original(*a, **kw)
        sync_mod.sync_scan = _wrapped
        try:
            t = sync_scan_async('BG', tmp_path)
            t.join(timeout=5)
            assert not t.is_alive()
        finally:
            sync_mod.sync_scan = original
    assert (tmp_path / DIAG_H5).exists()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
