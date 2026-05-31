"""Phase 0 tests — SLM proxy + dashboard passthrough routes + run_monitor flags.

Run as:
    python -m yb_analysis.tests.slm_migration.test_phase0_proxy
or via pytest:
    pytest yb_analysis/tests/slm_migration/test_phase0_proxy.py -v

Tests are self-contained: the FakeSlmServer fixture spins up a Flask server on
an OS-assigned port, the proxy is constructed with short poll intervals (50 ms),
and assertions run against the on-disk pickle / Flask test client.
"""

import math
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

from yb_analysis import config
from yb_analysis.slm_proxy import SlmProxy, SLM_DATA_FILE, _read_slm_data
from yb_analysis.plotting.dashboard import _to_jsonable, _build_app
from yb_analysis.tests.slm_migration.fake_slm_server import (
    FakeSlmServer, _MIN_PNG,
)


@pytest.fixture(autouse=True)
def _clean_slm_pickle():
    """Remove the SLM pickle before and after each test for isolation."""
    for p in (SLM_DATA_FILE, SLM_DATA_FILE + '.tmp'):
        try:
            os.remove(p)
        except OSError:
            pass
    yield
    for p in (SLM_DATA_FILE, SLM_DATA_FILE + '.tmp'):
        try:
            os.remove(p)
        except OSError:
            pass


# Fast poll intervals so tests don't have to wait — 50 ms across the board.
_FAST_INTERVALS = {name: 50 for name in (
    'health', 'devices', 'lock_status',
    'camera_png', 'phase_png', 'rearrange_diag')}


# ----------------------------------------------------------------------------
# Test 1 — config defaults
# ----------------------------------------------------------------------------

def test_config_defaults():
    """SLM_URL defaults to the lab Tailscale IP; env vars override.

    Also asserts SLM_POLL_INTERVALS_MS has every endpoint name the proxy
    expects and that timeouts are a (connect, read) tuple.
    """
    # Default is the documented SLM PC tailnet address.
    assert config.SLM_URL.startswith('http://100.114.207.118:'), (
        f'SLM_URL is {config.SLM_URL!r}; expected default to the SLM-PC '
        f'Tailscale IP 100.114.207.118 (override with YB_SLM_URL env var).')

    # Env override is respected. config is a module-level constant, so we
    # reload to pick up the env change. Restore env after.
    old = os.environ.get('YB_SLM_URL')
    try:
        os.environ['YB_SLM_URL'] = 'http://override.example:9999'
        import importlib
        importlib.reload(config)
        assert config.SLM_URL == 'http://override.example:9999'
    finally:
        if old is None:
            os.environ.pop('YB_SLM_URL', None)
        else:
            os.environ['YB_SLM_URL'] = old
        importlib.reload(config)

    # Every endpoint in the proxy's catalog has a configured interval.
    for ep_name, _path, _kind in SlmProxy._ENDPOINTS:
        assert ep_name in config.SLM_POLL_INTERVALS_MS, (
            f'SLM_POLL_INTERVALS_MS missing {ep_name!r}')

    # Timeout is a (connect, read) tuple of floats > 0.
    assert isinstance(config.SLM_HTTP_TIMEOUT_S, tuple)
    assert len(config.SLM_HTTP_TIMEOUT_S) == 2
    assert all(t > 0 for t in config.SLM_HTTP_TIMEOUT_S)


# ----------------------------------------------------------------------------
# Test 2 — _to_jsonable idempotent
# ----------------------------------------------------------------------------

def test_to_jsonable_idempotent():
    """JSON-safe coercer round-trips numpy, NaN, Inf, bytes, and nested dicts.

    The /api/* routes use this to coerce dashboard pickle contents into
    flask.jsonify-safe payloads. Idempotence matters because _to_jsonable runs
    once per request and a non-idempotent transform could mutate cached state.
    """
    payload = {
        'arr':     np.array([1, 2, 3], dtype=np.int64),
        'fl':      np.float64(1.5),
        'nan':     float('nan'),
        'inf':     float('inf'),
        'b':       b'hello',
        'nested':  {'k': np.int32(7), 'v': [np.float32(0.1), np.float32(0.2)]},
        'bool':    np.bool_(True),
    }
    coerced = _to_jsonable(payload)
    coerced2 = _to_jsonable(coerced)
    # Idempotent: running again is a no-op.
    assert coerced == coerced2, 'second pass should be a no-op'

    # Spot-check types: every leaf is plain Python now.
    assert coerced['arr'] == [1, 2, 3]
    assert coerced['fl'] == 1.5 and isinstance(coerced['fl'], float)
    assert coerced['nan'] is None and coerced['inf'] is None
    assert coerced['b'] == 'hello'
    assert coerced['nested']['k'] == 7
    assert coerced['bool'] is True


# ----------------------------------------------------------------------------
# Test 3 — proxy polls fake SLM and writes pickle
# ----------------------------------------------------------------------------

def test_proxy_polls_fake_slm():
    """Proxy fetches every endpoint and writes the result into yb_dash_slm.pkl.

    Asserts: after a brief poll window, every endpoint has been hit at least
    once and the pickle contains the corresponding payload (or PNG bytes).
    """
    with FakeSlmServer() as fake:
        proxy = SlmProxy(slm_url=fake.url, intervals_ms=_FAST_INTERVALS)
        proxy.start()
        try:
            # Wait until every endpoint has been polled (or 3s, whichever).
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if all(fake.hits(n) >= 1 for n in (
                        'health', 'devices', 'lock_status',
                        'camera_png', 'phase_png', 'rearrange_diag')):
                    break
                time.sleep(0.05)
        finally:
            proxy.stop()

    snap = _read_slm_data()
    assert snap is not None, 'pickle was never written'
    assert snap['slm_offline'] is False
    assert snap['slm_url'] == fake.url
    # JSON payloads round-tripped.
    assert snap['health']['version'] == 'fake-1'
    assert snap['devices']['slm'] == 'ok'
    assert snap['lock_status']['slm']['holder'] == 'matlab'
    assert snap['rearrange_diag']['count'] == 0
    # PNG bytes preserved as bytes (not decoded to numpy or anything).
    assert isinstance(snap['camera_png'], (bytes, bytearray))
    assert snap['camera_png'].startswith(b'\x89PNG')
    assert isinstance(snap['phase_png'], (bytes, bytearray))


# ----------------------------------------------------------------------------
# Test 4 — proxy handles offline gracefully
# ----------------------------------------------------------------------------

def test_proxy_handles_offline():
    """Proxy doesn't crash when the SLM PC is unreachable.

    Targets a port nobody is listening on; expects slm_offline=True in the
    pickle, with last_error_msg populated.
    """
    # Pick a port that's almost certainly closed.
    proxy = SlmProxy(
        slm_url='http://127.0.0.1:1',  # port 1: privileged + unused
        intervals_ms={n: 80 for n in _FAST_INTERVALS},
        timeout_s=(0.2, 0.2),  # fail fast
    )
    proxy.start()
    try:
        # Give it a few poll cycles to record errors.
        time.sleep(1.0)
    finally:
        proxy.stop()

    snap = _read_slm_data()
    assert snap is not None, 'proxy should still produce a pickle even when offline'
    assert snap['slm_offline'] is True
    assert snap['last_error_msg'], 'should record per-endpoint error messages'
    # Verify the proxy didn't crash a thread — the snapshot has the catalog.
    for ep_name, _p, _k in SlmProxy._ENDPOINTS:
        assert ep_name in snap, f'endpoint slot {ep_name!r} missing from snapshot'


# ----------------------------------------------------------------------------
# Test 5 — Flask passthrough routes return what the proxy cached
# ----------------------------------------------------------------------------

def test_passthrough_routes():
    """/api/slm/{health,devices,lock/status,rearrange/diag} return JSON from the pickle.

    /api/slm and /api/slm/ return the proxy index. Done via Flask's test
    client so this runs without standing up a real Dash subprocess.
    """
    with FakeSlmServer() as fake:
        fake.set_payload('health', {'uptime_s': 42, 'version': 'fake-2'})
        fake.set_payload('lock_status',
                         {'slm': {'holder': 'tester', 'age_s': 1}})
        fake.set_payload('rearrange_diag',
                         {'count': 1, 'entries': [{'kind': 'rearrange',
                                                    'diag': {'total_ms': 99}}]})
        proxy = SlmProxy(slm_url=fake.url, intervals_ms=_FAST_INTERVALS)
        proxy.start()
        try:
            # Wait for at least one full poll cycle so the pickle has data.
            deadline = time.time() + 3.0
            while time.time() < deadline:
                snap = _read_slm_data()
                if (snap and snap.get('health') and snap.get('lock_status')
                        and snap.get('rearrange_diag')):
                    break
                time.sleep(0.05)
        finally:
            proxy.stop()

    # Build the Dash app's Flask server and probe via test client.
    client = _build_app().server.test_client()

    r = client.get('/api/slm')
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body['service'] == 'yb-slm-proxy'
    assert body['slm_offline'] is False
    assert body['slm_url'] == fake.url
    # endpoints list reflects what's registered.
    assert '/api/slm/health' in body['endpoints']

    r = client.get('/api/slm/health')
    assert r.status_code == 200
    assert r.get_json()['version'] == 'fake-2'

    r = client.get('/api/slm/lock/status')
    assert r.status_code == 200
    assert r.get_json()['slm']['holder'] == 'tester'

    r = client.get('/api/slm/rearrange/diag')
    assert r.status_code == 200
    assert r.get_json()['count'] == 1


def test_passthrough_503_when_proxy_disabled():
    """With no pickle on disk, /api/slm/* returns 503 with an error body.

    Verifies the dashboard doesn't crash if --no-slm was passed.
    """
    # _clean_slm_pickle fixture already deleted the file.
    client = _build_app().server.test_client()

    r = client.get('/api/slm/health')
    assert r.status_code == 503
    body = r.get_json()
    assert 'error' in body
    assert 'disabled' in body['error'] or 'offline' in body['error']


# ----------------------------------------------------------------------------
# Test 6 — PNG passthrough returns raw bytes
# ----------------------------------------------------------------------------

def test_png_passthrough():
    """/api/slm/camera/png and /phase/png return the SLM PC's PNG bytes verbatim.

    Asserts the response is image/png mime type and the bytes match what the
    fake server returned (no double-encoding, no JSON wrapping).
    """
    custom_png = _MIN_PNG + b'\x00' * 8  # something distinctive
    with FakeSlmServer() as fake:
        fake.set_png('camera', custom_png)
        proxy = SlmProxy(slm_url=fake.url, intervals_ms=_FAST_INTERVALS)
        proxy.start()
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline:
                snap = _read_slm_data()
                if snap and snap.get('camera_png') == custom_png:
                    break
                time.sleep(0.05)
        finally:
            proxy.stop()

    client = _build_app().server.test_client()
    r = client.get('/api/slm/camera/png')
    assert r.status_code == 200
    assert r.mimetype == 'image/png'
    assert r.data == custom_png

    r = client.get('/api/slm/phase/png')
    assert r.status_code == 200
    assert r.mimetype == 'image/png'
    assert r.data == _MIN_PNG


# ----------------------------------------------------------------------------
# Test 7 — --no-slm flag skips proxy start in run_monitor
# ----------------------------------------------------------------------------

def test_run_monitor_no_slm_flag():
    """Invoking run_monitor with --no-slm + --no-runner exits cleanly without proxy.

    Spawn run_monitor as a subprocess with --help-equivalent invocation: pass
    --help so it parses args, prints help, exits 0, and never reaches the
    proxy startup. Verifies the new args are accepted by argparse.

    A full integration test would also start the dashboard, but that hangs in
    GUI mode; --help is the simplest argparse exercise.
    """
    env = dict(os.environ)
    # Don't actually start GUI / runner / hardware. argparse-only.
    cmd = [sys.executable, '-m', 'yb_analysis.scripts.run_monitor', '--help']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, f'--help failed: stderr={r.stderr!r}'
    # New flags should appear in the help output.
    out = r.stdout
    assert '--no-slm' in out, 'argparse missing --no-slm flag'
    assert '--slm-url' in out, 'argparse missing --slm-url flag'
    assert '--bind-tailscale' in out, 'argparse missing --bind-tailscale flag'

    # Also verify the proxy module module-loads OK with the same env (catches
    # import-time errors that --help wouldn't surface).
    cmd = [sys.executable, '-c', 'from yb_analysis.slm_proxy import SlmProxy; print("ok")']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=15)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr


# ----------------------------------------------------------------------------
# Test 8 — proxy thread clean shutdown
# ----------------------------------------------------------------------------

def test_proxy_thread_clean_shutdown():
    """All proxy threads exit within 2s after stop().

    Without this, restarting run_monitor would leak threads — the asyncio
    loop's daemon=True keeps them alive but they continue to spin until the
    process exits, which is wasteful and can confuse pytest's threading
    detection in later tests.
    """
    with FakeSlmServer() as fake:
        proxy = SlmProxy(slm_url=fake.url, intervals_ms=_FAST_INTERVALS)
        proxy.start()
        # Let threads spin up.
        time.sleep(0.3)
        threads_before = list(proxy._threads)
        assert len(threads_before) == 6
        assert all(t.is_alive() for t in threads_before)

        proxy.stop(timeout_s=2.0)
        # All threads gone from the proxy's bookkeeping.
        assert proxy._threads == []
        # And actually dead — let the OS scheduler catch up a beat.
        time.sleep(0.1)
        alive = [t.name for t in threads_before if t.is_alive()]
        assert alive == [], f'threads still alive after stop(): {alive}'

        # stop() is idempotent.
        proxy.stop(timeout_s=0.1)


# ----------------------------------------------------------------------------
# CLI entry point — run with: python -m yb_analysis.tests.slm_migration.test_phase0_proxy
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
