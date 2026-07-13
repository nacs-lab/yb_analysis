"""Payload-cache fast path: finished vs running scan (run_analysis).

- FINISHED scan (shot count == cached): exact hit, no background refresh.
- RUNNING scan (shot count grew): serve the STALE cached payload immediately
  and schedule ONE background recompute (coalesced) so refreshes are instant.

Run: yb_analysis-env python -m pytest yb_analysis/tests/test_payload_cache_stale.py -v
"""
import json

import pytest

from yb_analysis.analysis import run_analysis as RA


@pytest.fixture
def scan_dir(tmp_path, monkeypatch):
    d = tmp_path / 'data_20260101_120000'
    d.mkdir()
    # The fast-path runs just after a (cheap) load_scan_from_path; stub it so the
    # test reaches the cache check without a real .h5/.mat on disk.
    monkeypatch.setattr(RA, 'load_scan_from_path', lambda p: {'Scan': {}})
    return d


def _write_cache(scan_dir, n_shots, marker='CACHED'):
    payload = {'scan_id': '20260101120000', 'n_shots': n_shots, '_marker': marker}
    RA._write_payload_cache(scan_dir, payload, n_shots)


def test_finished_scan_exact_hit_no_refresh(scan_dir, monkeypatch):
    _write_cache(scan_dir, 200)
    monkeypatch.setattr(RA, '_probe_actual_shots', lambda p: 200)   # unchanged
    scheduled = []
    monkeypatch.setattr(RA, '_schedule_payload_refresh',
                        lambda p: scheduled.append(str(p)))
    # A finished scan hits the exact fast path; nothing else runs.
    out = RA.analyze_scan_dir(scan_dir)
    assert out['_marker'] == 'CACHED'          # served from cache
    assert scheduled == []                     # no background refresh


def test_running_scan_serves_stale_and_schedules_refresh(scan_dir, monkeypatch):
    _write_cache(scan_dir, 100)                # cache is behind
    monkeypatch.setattr(RA, '_probe_actual_shots', lambda p: 137)  # grew
    scheduled = []
    monkeypatch.setattr(RA, '_schedule_payload_refresh',
                        lambda p: scheduled.append(str(p)))
    out = RA.analyze_scan_dir(scan_dir)
    assert out['_marker'] == 'CACHED'          # STALE payload served instantly
    assert out['n_shots'] == 100               # (stale count, refreshes next time)
    assert scheduled == [str(scan_dir)]        # exactly one refresh scheduled


def test_refresh_scheduler_coalesces(scan_dir, monkeypatch):
    """Rapid schedules for the same scan spawn at most one worker (cooldown +
    in-flight guard); the worker calls analyze_scan_dir with _skip_fastpath."""
    calls = []
    monkeypatch.setattr(
        RA, 'analyze_scan_dir',
        lambda p, **k: calls.append(k) or {})
    # First schedule fires; the rest are within cooldown / in-flight -> skipped.
    RA._PAYLOAD_REFRESHING.clear()
    RA._PAYLOAD_REFRESH_LAST.clear()
    for _ in range(5):
        RA._schedule_payload_refresh(scan_dir)
    # Let the (single) worker thread run.
    import time
    time.sleep(0.3)
    assert len(calls) <= 1
    if calls:
        assert calls[0].get('_skip_fastpath') is True   # forces a real compute


def test_skip_fastpath_bypasses_cache(scan_dir, monkeypatch):
    """_skip_fastpath=True must NOT short-circuit on the cache (else the
    background refresh would re-serve stale and never update)."""
    _write_cache(scan_dir, 100)
    monkeypatch.setattr(RA, '_probe_actual_shots', lambda p: 137)
    hit = []
    monkeypatch.setattr(RA, '_read_payload_cache',
                        lambda p: hit.append(1) or {'n_shots': 100,
                                                    'payload': {'_marker': 'X'}})
    # With _skip_fastpath the fast-path cache read must not run; the call
    # proceeds into the real analysis instead of short-circuiting on the cache.
    RA.analyze_scan_dir(scan_dir, _skip_fastpath=True)
    assert hit == []                            # fast-path cache read skipped
