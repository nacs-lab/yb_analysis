"""Phase 5.5 Track A — dashboard control mirror.

Covers the shared MemoryMap layout module, the /api/control/* endpoints
(MemoryMap-first pause/abort, confirm-token gating, web-command spooling for
dummy_mode / init_dir), the remote-exposure gate, and the rendered control
DOM. Run in the yb_analysis conda env:

    python -m pytest yb_analysis/tests/test_phase5_5_controls.py -v
"""

import struct

import pytest

from yb_analysis.control import memmap_signal as mm
from yb_analysis.control import web_control as wc


# --- shared MemoryMap layout ----------------------------------------------

def test_memmap_signal_constants_match_matlab():
    # Re-derive offsets from MemoryMap.m's documented field order, fully
    # independently of memmap_signal._LAYOUT, then compare.
    matlab_fields = [
        ('ScanParamsSet', 8), ('AndorConfigured', 8), ('ScanComplete', 8),
        ('NumImages', 8), ('NumPerParamAvg', 8), ('NumPerGroup', 8),
        ('DateStamp', 8), ('TimeStamp', 8), ('AbortRunSeq', 8),
        ('PauseRunSeq', 8), ('IsPausedRunSeq', 8), ('CurrentSeqNum', 8),
        ('Email', 32), ('FreqEOM616Old', 8), ('AWGFreqs', 256),
        ('DummyRunning', 8),
    ]
    off, expected = 0, {}
    for name, size in matlab_fields:
        expected[name] = off
        off += size
    assert mm.OFFSETS == expected
    assert mm.OFF_ABORT == expected['AbortRunSeq'] == 64
    assert mm.OFF_PAUSE == expected['PauseRunSeq'] == 72
    assert mm.OFF_DUMMY_RUNNING == expected['DummyRunning'] == 392


# --- Flask app + temp wiring ----------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    # Point the MemoryMap + web-command spool at temp locations.
    mmap_file = tmp_path / 'mem_map.dat'
    mmap_file.write_bytes(b'\x00' * 512)
    monkeypatch.setattr(mm, 'MMAP_PATH', str(mmap_file))
    monkeypatch.setattr(wc, 'CMD_DIR', str(tmp_path / 'web_cmds'))
    monkeypatch.delenv('YB_DASH_REMOTE_CONTROLS', raising=False)
    # Default the dashboard to the MATLAB (memmap) path for the legacy tests;
    # the pyctrl-path tests set YB_BACKEND=pyctrl explicitly.
    monkeypatch.delenv('YB_BACKEND', raising=False)
    from yb_analysis.plotting.dashboard import _build_app
    flask_app = _build_app().server
    flask_app.config['TESTING'] = True
    return flask_app, mmap_file


def _read_double(path, offset):
    with open(path, 'rb') as f:
        f.seek(offset)
        return struct.unpack('d', f.read(8))[0]


def test_api_pause_writes_mmap(app):
    flask_app, mmap_file = app
    c = flask_app.test_client()
    r = c.post('/api/control/pause')
    assert r.status_code == 200
    assert r.get_json()['ok'] is True
    assert _read_double(mmap_file, mm.OFF_PAUSE) == 1.0


def test_api_start_clears_pause(app):
    flask_app, mmap_file = app
    c = flask_app.test_client()
    c.post('/api/control/pause')
    c.post('/api/control/start')
    assert _read_double(mmap_file, mm.OFF_PAUSE) == 0.0


def test_api_abort_requires_confirmation(app):
    flask_app, mmap_file = app
    c = flask_app.test_client()
    # No token -> rejected, abort flag untouched.
    r = c.post('/api/control/abort')
    assert r.status_code == 400
    assert _read_double(mmap_file, mm.OFF_ABORT) == 0.0
    # With a valid token -> abort flag set.
    tok = c.get('/api/control/confirm_token?action=abort').get_json()['token']
    r = c.post(f'/api/control/abort?confirm={tok}')
    assert r.status_code == 200
    assert _read_double(mmap_file, mm.OFF_ABORT) == 1.0
    # Token is single-use -> a replay is rejected.
    assert c.post(f'/api/control/abort?confirm={tok}').status_code == 400


def test_api_restart_all_requires_confirmation(app):
    flask_app, _ = app
    c = flask_app.test_client()
    assert c.post('/api/control/restart_all').status_code == 400
    # A token issued for 'abort' must not authorise restart_all.
    abort_tok = c.get(
        '/api/control/confirm_token?action=abort').get_json()['token']
    assert c.post(
        f'/api/control/restart_all?confirm={abort_tok}').status_code == 400
    tok = c.get(
        '/api/control/confirm_token?action=restart_all').get_json()['token']
    r = c.post(f'/api/control/restart_all?confirm={tok}')
    assert r.status_code == 200
    assert wc.drain()[0]['cmd'] == 'restart_all'


def test_dummy_mode_set(app):
    flask_app, _ = app
    c = flask_app.test_client()
    r = c.post('/api/control/dummy_mode', json={'mode': 'default'})
    assert r.status_code == 200
    cmds = wc.drain()
    assert cmds and cmds[0]['cmd'] == 'dummy_mode' and cmds[0]['mode'] == 'default'
    # Bad mode rejected, nothing spooled.
    assert c.post('/api/control/dummy_mode',
                  json={'mode': 'bogus'}).status_code == 400
    assert wc.drain() == []


def test_init_dir_load(app, tmp_path):
    flask_app, _ = app
    c = flask_app.test_client()
    good = tmp_path / 'calib'
    good.mkdir()
    r = c.post('/api/control/init_dir', json={'path': str(good)})
    assert r.status_code == 200
    cmds = wc.drain()
    assert cmds and cmds[0]['cmd'] == 'init_dir' and cmds[0]['path'] == str(good)
    # Non-existent path rejected.
    assert c.post('/api/control/init_dir',
                  json={'path': str(tmp_path / 'nope')}).status_code == 400


def test_downsample_toggle(app):
    flask_app, _ = app
    c = flask_app.test_client()
    assert c.post('/api/control/downsample',
                  json={'on': False}).get_json()['downsample'] is False
    assert c.post('/api/control/downsample',
                  json={'on': True}).get_json()['downsample'] is True


def test_remote_controls_default_disabled_on_lan(app):
    flask_app, _ = app
    c = flask_app.test_client()
    # A non-loopback / non-tailscale client is blocked by default ('auto').
    r = c.post('/api/control/pause',
               environ_base={'REMOTE_ADDR': '192.168.1.50'})
    assert r.status_code == 403
    # Loopback is always allowed.
    assert c.post('/api/control/pause',
                  environ_base={'REMOTE_ADDR': '127.0.0.1'}).status_code == 200


def test_remote_controls_enabled_when_on(app, monkeypatch):
    flask_app, _ = app
    monkeypatch.setenv('YB_DASH_REMOTE_CONTROLS', 'on')
    c = flask_app.test_client()
    assert c.post('/api/control/pause',
                  environ_base={'REMOTE_ADDR': '192.168.1.50'}).status_code == 200


def test_control_status_route(app):
    flask_app, _ = app
    c = flask_app.test_client()
    body = c.get('/api/control/status').get_json()
    assert 'controls_allowed' in body


def test_live_diag_pull_poll(app, monkeypatch):
    # Track D: /api/runs/<id>/diag_live proxies the SLM incremental diag.
    import yb_analysis.slm_sync as slm_sync

    class _FakeClient:
        def get_diag(self, scan_id, since_seq_id=None):
            assert since_seq_id == 4
            return {'entries': [{'seq_id': 5}, {'seq_id': 6}], 'count': 6}

    monkeypatch.setattr(slm_sync, 'SlmSyncClient', _FakeClient)
    flask_app, _ = app
    c = flask_app.test_client()
    r = c.get('/api/runs/20260101010101/diag_live?since_seq_id=4')
    assert r.status_code == 200
    body = r.get_json()
    assert body['count'] == 6
    assert [e['seq_id'] for e in body['entries']] == [5, 6]


def test_live_diag_offline_returns_503(app, monkeypatch):
    import yb_analysis.slm_sync as slm_sync

    class _OfflineClient:
        def get_diag(self, scan_id, since_seq_id=None):
            return None

    monkeypatch.setattr(slm_sync, 'SlmSyncClient', _OfflineClient)
    flask_app, _ = app
    c = flask_app.test_client()
    assert c.get('/api/runs/20260101010101/diag_live').status_code == 503


def test_dashboard_renders_controls(app):
    flask_app, _ = app
    c = flask_app.test_client()
    html = c.get('/').get_data(as_text=True)
    for needle in (
        'data-kind="start"', 'data-kind="pause"', 'data-kind="abort"',
        'data-kind="restart-dash"', 'data-kind="restart-all"',
        'name="dummy-mode-ph5"', 'id="ctrl-init-path"',
        'id="ctrl-scan-id"', 'id="ctrl-runner-state"',
        'id="downsample-live"', 'id="show-mid-live"',
        'id="analysis-svd-card"', 'id="shot-health-tile"',
        # Phase 5.5 sidebar additions: runner-status banner, camera card,
        # queue preview.
        'id="ctrl-status-banner"', 'id="cam-status"', 'data-cam="connect"',
        'data-cam="disconnect"', 'data-cam="apply"', 'id="cam-exposure"',
        'id="ctrl-queue-active"', 'id="ctrl-queue-list"',
    ):
        assert needle in html, f'missing control markup: {needle}'
    # The queue tiles moved out of the top status strip into the sidebar.
    assert 'id="mini-queue-active-tile"' not in html
    assert 'id="mini-queue-next-tile"' not in html


# --- Camera control mirror (Phase 5.5) ------------------------------------

def test_camera_status_route(app):
    flask_app, _ = app
    c = flask_app.test_client()
    # No file published yet -> just the controls_allowed flag.
    body = c.get('/api/control/camera/status').get_json()
    assert 'controls_allowed' in body
    # Publish a status and read it back through the route.
    wc.publish_camera_status({
        'connected': True, 'roi': [0, 0, 512, 512], 'exposure_time': 0.1,
        'error': '', 'busy': False, 'status_text': 'Connected'})
    body = c.get('/api/control/camera/status').get_json()
    assert body['connected'] is True
    assert body['roi'] == [0, 0, 512, 512]
    assert body['status_text'] == 'Connected'


def test_camera_connect_spools_command(app):
    flask_app, _ = app
    c = flask_app.test_client()
    r = c.post('/api/control/camera/connect',
               json={'roi': [1, 2, 3, 4], 'exposure': 0.05})
    assert r.status_code == 200
    cmds = wc.drain()
    assert cmds and cmds[0]['cmd'] == 'camera_connect'
    assert cmds[0]['roi'] == [1, 2, 3, 4] and cmds[0]['exposure'] == 0.05


def test_camera_apply_and_disconnect_spool(app):
    flask_app, _ = app
    c = flask_app.test_client()
    assert c.post('/api/control/camera/apply',
                  json={'roi': [0, 0, 64, 64], 'exposure': 0.2}).status_code == 200
    assert c.post('/api/control/camera/disconnect').status_code == 200
    cmds = wc.drain()
    kinds = [x['cmd'] for x in cmds]
    assert kinds == ['camera_apply', 'camera_disconnect']


def test_camera_connect_validation(app):
    flask_app, _ = app
    c = flask_app.test_client()
    # Bad ROI shape, missing exposure, non-positive exposure all 400.
    assert c.post('/api/control/camera/connect',
                  json={'roi': [1, 2, 3], 'exposure': 0.1}).status_code == 400
    assert c.post('/api/control/camera/connect',
                  json={'roi': [1, 2, 3, 4]}).status_code == 400
    assert c.post('/api/control/camera/connect',
                  json={'roi': [1, 2, 3, 4], 'exposure': 0}).status_code == 400
    # Nothing should have been spooled by the rejected requests.
    assert wc.drain() == []


def test_camera_controls_gated_on_lan(app):
    flask_app, _ = app
    c = flask_app.test_client()
    # Default 'auto' policy blocks a plain-LAN client on the write routes.
    r = c.post('/api/control/camera/connect',
               json={'roi': [1, 2, 3, 4], 'exposure': 0.1},
               environ_base={'REMOTE_ADDR': '192.168.1.50'})
    assert r.status_code == 403


# --- Backend-aware control: pyctrl uses NO local memmap -------------------

def test_mmap_open_ignores_stale_file_under_pyctrl(app):
    # The fixture wrote a 512-byte mem_map.dat (a *stale* file, as a prior
    # MATLAB session would leave). Under matlab it opens; under pyctrl it must
    # be ignored UNCONDITIONALLY (never read/written) — the core safety rule.
    _flask_app, mmap_file = app
    assert mmap_file.exists()
    m = mm.mmap_open('matlab')
    assert m is not None
    m.close()
    assert mm.mmap_open('pyctrl') is None
    assert mm.mmap_open('anything-else') is None


def test_signal_helpers_noop_under_pyctrl(app):
    _flask_app, mmap_file = app
    before = mmap_file.read_bytes()
    assert mm.signal_pause('pyctrl') is False
    assert mm.signal_start('pyctrl') is False
    assert mm.signal_abort('pyctrl') is False
    # The stale file must be byte-for-byte untouched.
    assert mmap_file.read_bytes() == before
    # Sanity: the matlab path still writes.
    assert mm.signal_pause('matlab') is True
    assert _read_double(mmap_file, mm.OFF_PAUSE) == 1.0


def test_api_pause_pyctrl_spools_not_memmap(app, monkeypatch):
    monkeypatch.setenv('YB_BACKEND', 'pyctrl')
    flask_app, mmap_file = app
    c = flask_app.test_client()
    r = c.post('/api/control/pause')
    assert r.status_code == 200
    assert r.get_json()['via'] == 'run_monitor'
    # Routed to ZMQ via the spool; the stale memmap is untouched.
    assert wc.drain()[0]['cmd'] == 'pause'
    assert _read_double(mmap_file, mm.OFF_PAUSE) == 0.0


def test_api_start_pyctrl_spools_not_memmap(app, monkeypatch):
    monkeypatch.setenv('YB_BACKEND', 'pyctrl')
    flask_app, mmap_file = app
    c = flask_app.test_client()
    r = c.post('/api/control/start')
    assert r.status_code == 200
    assert r.get_json()['via'] == 'run_monitor'
    assert wc.drain()[0]['cmd'] == 'start'
    assert _read_double(mmap_file, mm.OFF_PAUSE) == 0.0


def test_api_abort_pyctrl_spools_with_token_only(app, monkeypatch):
    monkeypatch.setenv('YB_BACKEND', 'pyctrl')
    flask_app, mmap_file = app
    c = flask_app.test_client()
    # No token -> rejected, nothing spooled, memmap untouched.
    assert c.post('/api/control/abort').status_code == 400
    assert wc.drain() == []
    assert _read_double(mmap_file, mm.OFF_ABORT) == 0.0
    # Valid token -> spools 'abort' (ZMQ path), memmap still untouched.
    tok = c.get('/api/control/confirm_token?action=abort').get_json()['token']
    r = c.post(f'/api/control/abort?confirm={tok}')
    assert r.status_code == 200
    assert r.get_json()['via'] == 'run_monitor'
    assert wc.drain()[0]['cmd'] == 'abort'
    assert _read_double(mmap_file, mm.OFF_ABORT) == 0.0


def test_api_pause_invalid_backend_fails_closed_to_zmq(app, monkeypatch):
    # Defense-in-depth: a misconfigured YB_BACKEND must NOT touch the memmap;
    # only exactly 'matlab' uses it, everything else routes via ZMQ (spool).
    monkeypatch.setenv('YB_BACKEND', 'matlabx')  # typo / bad value
    flask_app, mmap_file = app
    c = flask_app.test_client()
    r = c.post('/api/control/pause')
    assert r.status_code == 200
    assert r.get_json()['via'] == 'run_monitor'
    assert wc.drain()[0]['cmd'] == 'pause'
    assert _read_double(mmap_file, mm.OFF_PAUSE) == 0.0
