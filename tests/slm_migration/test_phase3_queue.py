"""Phase 3 tests — descriptor schema, scans client, ExptServer kind
discriminator, dashboard /api/queue/{submit,cancel,move}.

Five test tiers:

1. **Schema + helpers (pure unit)** — validate_descriptor + sweep_*
   builders. No ZMQ, no MATLAB.
2. **ExptServer descriptor methods (in-process)** — submit/pop/finish/
   link a descriptor by calling the Python methods directly. No ZMQ
   socket required.
3. **ExptServer + ExptClient over ZMQ** — bind a real ROUTER socket
   on a free port, exercise submit_scan_descriptor + descriptor_remove
   from ExptClient. No MATLAB.
4. **Dashboard /api/queue/submit** — Flask test client against the
   real dashboard route, hitting a live in-process ExptServer.
5. **Backward-compat / downgrade safety** — runner_queue.json with
   `kind: 'descriptor'` rows survives __load_queue without crashing
   older code.

MATLAB-side tests live at matlab_new/test/phase3/TestYbBuildScanPayload.m
and run via MATLAB's `runtests`.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest


# Make matlab_new/YbExptCtrl importable (ExptServer / ExptClient).
_REPO = Path(__file__).resolve().parents[3]
_EXPSERVER_DIR = _REPO / 'matlab_new' / 'YbExptCtrl'
if str(_EXPSERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPSERVER_DIR))

from yb_analysis.scans.convenience import (   # noqa: E402
    sweep_linspace, sweep_logspace, sweep_values, func_handle,
)
from yb_analysis.scans.descriptor import (    # noqa: E402
    SCHEMA_VERSION, DescriptorError, validate_descriptor,
)


# ---------------------------------------------------------------------------
# Tier 1 — Schema + helpers (pure unit)
# ---------------------------------------------------------------------------

def test_sweep_linspace_shape():
    sw = sweep_linspace(20e6, 30e6, 21)
    assert sw == {'scan': 1, 'linspace': [20e6, 30e6, 21]}


def test_sweep_linspace_axis_kwarg():
    sw = sweep_linspace(0, 1, 11, axis=2)
    assert sw['scan'] == 2


def test_sweep_linspace_bad_n():
    with pytest.raises(ValueError, match="n must be"):
        sweep_linspace(0, 1, 0)


def test_sweep_logspace_shape():
    sw = sweep_logspace(-3, 3, 7, axis=2)
    assert sw == {'scan': 2, 'logspace': [-3, 3, 7]}


def test_sweep_values_numeric():
    sw = sweep_values([1, 2, 3])
    assert sw['scan'] == 1
    assert sw['values'] == [1.0, 2.0, 3.0]


def test_sweep_values_string():
    sw = sweep_values(['AWG556', 'AWG308'])
    assert sw == {'scan': 1, 'values': ['AWG556', 'AWG308']}


def test_sweep_values_mixed_raises():
    with pytest.raises(ValueError, match="mixed types"):
        sweep_values([1, 'two', 3])


def test_func_handle_shape():
    assert func_handle('RearrangeCommSeq2') == {'@': 'RearrangeCommSeq2'}


def test_func_handle_invalid_name():
    with pytest.raises(ValueError):
        func_handle('not a valid name')


def test_validate_minimal():
    validate_descriptor({'seq': 'CoolingSeq'})


def test_validate_with_sweep():
    validate_descriptor({
        'seq': 'CoolingSeq',
        'params': {'Cooling.Detuning': sweep_linspace(20e6, 30e6, 11)},
    })


def test_validate_runp_cell_of_string():
    validate_descriptor({
        'seq': 'CoolingSeq',
        'runp': {'AWGs': ['AWG556']},
    })


def test_validate_function_handle_seq():
    validate_descriptor({'seq': func_handle('RearrangeCommSeq2')})


def test_validate_rejects_missing_seq():
    with pytest.raises(DescriptorError, match="seq"):
        validate_descriptor({'params': {'X': 1}})


def test_validate_rejects_bad_seq_name():
    with pytest.raises(DescriptorError, match="identifier"):
        validate_descriptor({'seq': 'invalid name'})


def test_validate_rejects_invalid_path_part():
    with pytest.raises(DescriptorError, match="identifier"):
        validate_descriptor({
            'seq': 'CoolingSeq',
            'params': {'Cooling.123Detuning': 5},
        })


def test_validate_rejects_bad_sweep_dim():
    with pytest.raises(DescriptorError, match="positive"):
        validate_descriptor({
            'seq': 'CoolingSeq',
            'params': {'X': {'scan': 0, 'linspace': [0, 1, 5]}},
        })


def test_validate_rejects_multi_sweep_kinds():
    with pytest.raises(DescriptorError, match="exactly one"):
        validate_descriptor({
            'seq': 'CoolingSeq',
            'params': {'X': {'scan': 1, 'linspace': [0, 1, 5],
                             'values': [1, 2, 3]}},
        })


def test_validate_opts_pair():
    validate_descriptor({
        'seq': 'CoolingSeq',
        'opts': [['scan_id', 12345], ['email', 'foo@bar']],
    })


def test_validate_opts_bad_pair():
    with pytest.raises(DescriptorError, match="\\[key, value\\] pair"):
        validate_descriptor({
            'seq': 'CoolingSeq',
            'opts': [['scan_id']],   # missing value
        })


def test_schema_version_constant():
    assert SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# Tier 2 — ExptServer descriptor methods (in-process, no ZMQ)
# ---------------------------------------------------------------------------

def _free_port():
    """Find a free TCP port on the loopback interface."""
    import socket
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def fresh_expt_server(tmp_path, monkeypatch):
    """Spawn a fresh ExptServer bound to a free port with an isolated
    runner_queue.json. The server's worker thread is started so ZMQ
    verbs work, but we mostly call methods directly."""
    import ExptServer as expt_server_mod
    # Route queue persistence into the tmp dir so the test doesn't
    # clobber the lab's real runner_queue.json.
    monkeypatch.setattr(expt_server_mod, 'QUEUE_PATH',
                        str(tmp_path / 'runner_queue.json'))
    port = _free_port()
    url = f'tcp://127.0.0.1:{port}'
    srv = expt_server_mod.ExptServer(url)
    try:
        yield srv
    finally:
        try:
            srv.stop_worker()
        except Exception:
            pass


def test_submit_scan_descriptor_assigns_id(fresh_expt_server):
    srv = fresh_expt_server
    desc = json.dumps({'seq': 'CoolingSeq',
                       'params': {'X': 1}})
    did = srv.submit_scan_descriptor(desc)
    assert isinstance(did, int) and did >= 1


def test_pop_next_descriptor_returns_body(fresh_expt_server):
    srv = fresh_expt_server
    desc1 = json.dumps({'seq': 'CoolingSeq', 'label': 'first'})
    desc2 = json.dumps({'seq': 'RamseySeq', 'label': 'second'})
    did1 = srv.submit_scan_descriptor(desc1)
    did2 = srv.submit_scan_descriptor(desc2)
    p1 = srv.pop_next_descriptor()
    assert p1['id'] == did1
    assert json.loads(p1['descriptor']).get('seq') == 'CoolingSeq'
    p2 = srv.pop_next_descriptor()
    assert p2['id'] == did2
    assert srv.pop_next_descriptor() is None


def test_pop_next_descriptor_skips_jobs(fresh_expt_server):
    """Mixed-kind queue: pop_next_descriptor returns only descriptors,
    pop_next_job returns only jobs."""
    srv = fresh_expt_server
    srv.submit_job(b'fake job payload')
    did = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    desc = srv.pop_next_descriptor()
    assert desc is not None and desc['id'] == did
    job = srv.pop_next_job()
    assert job is not None
    assert job['payload'] == b'fake job payload'


def test_link_descriptor_to_job_archives_to_history(fresh_expt_server):
    srv = fresh_expt_server
    did = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    srv.pop_next_descriptor()   # building
    fake_job_id = 999
    ok = srv.link_descriptor_to_job(did, fake_job_id)
    assert ok
    snap = srv.queue_list()
    # Descriptor is no longer in queued list
    assert not any(r['id'] == did for r in snap['queued'])
    # It moved to history with built_job_id stamped
    hist = [r for r in snap['history'] if r['id'] == did]
    assert len(hist) == 1
    assert hist[0]['kind'] == 'descriptor'
    assert hist[0]['built_job_id'] == fake_job_id
    assert hist[0]['state'] == 'built'


def test_finish_descriptor_error_path(fresh_expt_server):
    srv = fresh_expt_server
    did = srv.submit_scan_descriptor(json.dumps({'seq': 'Bad'}))
    srv.pop_next_descriptor()
    srv.finish_descriptor(did, 'error', 'bad seq')
    snap = srv.queue_list()
    hist = [r for r in snap['history'] if r['id'] == did]
    assert len(hist) == 1
    assert hist[0]['state'] == 'error'
    assert hist[0]['error_message'] == 'bad seq'


def test_descriptor_remove_only_when_queued(fresh_expt_server):
    srv = fresh_expt_server
    did = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    assert srv.descriptor_remove(did) is True
    # Re-submit and pop it -- now removal should fail (state='building')
    did2 = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    srv.pop_next_descriptor()
    assert srv.descriptor_remove(did2) is False


def test_queue_list_kind_discriminator(fresh_expt_server):
    srv = fresh_expt_server
    srv.submit_job(b'job1')
    srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    snap = srv.queue_list()
    kinds = sorted(r['kind'] for r in snap['queued'])
    assert kinds == ['descriptor', 'job']


def test_queue_move_kind_aware(fresh_expt_server):
    """queue_move moves an entry only relative to other same-kind
    queued entries -- jobs and descriptors don't compete for ordering."""
    srv = fresh_expt_server
    j1 = srv.submit_job(b'job1')
    j2 = srv.submit_job(b'job2')
    d1 = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    d2 = srv.submit_scan_descriptor(json.dumps({'seq': 'RamseySeq'}))
    # Move d2 up -> d2, d1
    assert srv.queue_move(d2, 'up')
    snap = srv.queue_list()
    desc_ids = [r['id'] for r in snap['queued'] if r['kind'] == 'descriptor']
    assert desc_ids == [d2, d1]
    # Jobs unaffected
    job_ids = [r['id'] for r in snap['queued'] if r['kind'] == 'job']
    assert job_ids == [j1, j2]


def test_pop_next_job_explicit_kind_for_existing_entries(fresh_expt_server):
    """A submit_job entry now stamps explicit kind='job' (Phase 3
    change). pop_next_job correctly filters by kind for both implicit
    (legacy) and explicit jobs."""
    srv = fresh_expt_server
    jid = srv.submit_job(b'job1')
    job = srv.pop_next_job()
    assert job is not None and job['id'] == jid


# ---------------------------------------------------------------------------
# Tier 3 — ZMQ-level: ExptServer + ExptClient end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def server_and_client(tmp_path, monkeypatch):
    """Real ZMQ in-process: ExptServer worker thread + ExptClient REQ
    socket on the same TCP loopback port."""
    import ExptServer as expt_server_mod
    import ExptClient as expt_client_mod
    monkeypatch.setattr(expt_server_mod, 'QUEUE_PATH',
                        str(tmp_path / 'runner_queue.json'))
    port = _free_port()
    url = f'tcp://127.0.0.1:{port}'
    srv = expt_server_mod.ExptServer(url)
    # Worker is started in __init__; give it a moment to bind.
    time.sleep(0.05)
    client = expt_client_mod.ExptClient(url)
    try:
        yield srv, client
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            srv.stop_worker()
        except Exception:
            pass


def test_zmq_submit_scan_descriptor(server_and_client):
    srv, client = server_and_client
    desc = json.dumps({'seq': 'CoolingSeq', 'params': {'X': 1}})
    did = client.submit_scan_descriptor(desc, label='from-zmq')
    assert did >= 1
    snap = client.queue_list()
    rows = [r for r in snap['queued'] if r['kind'] == 'descriptor']
    assert len(rows) == 1
    assert rows[0]['id'] == did
    assert rows[0]['label'] == 'from-zmq'


def test_zmq_descriptor_remove(server_and_client):
    srv, client = server_and_client
    did = client.submit_scan_descriptor(
        json.dumps({'seq': 'CoolingSeq'}))
    rep = client.descriptor_remove(did)
    assert rep == 'ok'
    rep2 = client.descriptor_remove(did)
    assert rep2.startswith('error')


# ---------------------------------------------------------------------------
# Tier 4 — Dashboard /api/queue/submit + /cancel + /move
# ---------------------------------------------------------------------------

@pytest.fixture
def dashboard_app(tmp_path, monkeypatch):
    """A dashboard server with /api/queue/* wired up against a fresh
    in-process ExptServer. Uses ZMQ via ZmqClient so we hit the full
    network stack the production dashboard uses."""
    import ExptServer as expt_server_mod
    monkeypatch.setattr(expt_server_mod, 'QUEUE_PATH',
                        str(tmp_path / 'runner_queue.json'))
    port = _free_port()
    url = f'tcp://127.0.0.1:{port}'
    srv = expt_server_mod.ExptServer(url)
    time.sleep(0.05)

    # Force the scans client to use our test URL by patching the default.
    from yb_analysis.scans import client as scans_client
    scans_client._CLIENT_CACHE.clear()
    monkeypatch.setattr(scans_client, '_default_matlab_url', lambda: url)

    from yb_analysis.plotting import dashboard as dash_mod
    from flask import Flask
    flask_app = Flask('phase3_dash_test')
    dash_mod._register_api_routes(flask_app)
    flask_app.testing = True
    try:
        yield srv, flask_app.test_client(), url
    finally:
        scans_client._CLIENT_CACHE.clear()
        try:
            srv.stop_worker()
        except Exception:
            pass


def test_dashboard_submit_descriptor(dashboard_app):
    srv, http, _url = dashboard_app
    payload = {
        'seq': 'CoolingSeq',
        'params': {'Cooling.Detuning': sweep_linspace(20e6, 30e6, 11)},
        'runp': {'NumPerGroup': 100},
        'label': 'dashboard-test',
    }
    r = http.post('/api/queue/submit', json=payload)
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['kind'] == 'descriptor'
    assert isinstance(body['descriptor_id'], int)

    # Confirm it actually landed on the queue
    snap = srv.queue_list()
    rows = [r for r in snap['queued'] if r['kind'] == 'descriptor']
    assert len(rows) == 1
    assert rows[0]['label'] == 'dashboard-test'


def test_dashboard_submit_invalid_descriptor(dashboard_app):
    _srv, http, _url = dashboard_app
    r = http.post('/api/queue/submit', json={'params': {'X': 1}})  # no seq
    assert r.status_code == 400
    assert 'seq' in r.get_json()['error']


def test_dashboard_submit_invalid_json(dashboard_app):
    _srv, http, _url = dashboard_app
    r = http.post('/api/queue/submit',
                  data='this is not JSON', content_type='application/json')
    assert r.status_code == 400


def test_dashboard_cancel_descriptor(dashboard_app):
    srv, http, _url = dashboard_app
    did = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    r = http.post(f'/api/queue/cancel/{did}')
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'id': did}
    # Gone from queue
    snap = srv.queue_list()
    assert not any(row['id'] == did for row in snap['queued'])


def test_dashboard_move_descriptor(dashboard_app):
    srv, http, _url = dashboard_app
    d1 = srv.submit_scan_descriptor(json.dumps({'seq': 'CoolingSeq'}))
    d2 = srv.submit_scan_descriptor(json.dumps({'seq': 'RamseySeq'}))
    r = http.post(f'/api/queue/move/{d2}/up')
    assert r.status_code == 200
    snap = srv.queue_list()
    desc_ids = [row['id'] for row in snap['queued']
                if row['kind'] == 'descriptor']
    assert desc_ids == [d2, d1]


def test_dashboard_move_bad_direction(dashboard_app):
    _srv, http, _url = dashboard_app
    r = http.post('/api/queue/move/1/sideways')
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Tier 5 — Backward-compat / downgrade safety
# ---------------------------------------------------------------------------

def test_unknown_kind_skipped_on_load(tmp_path, monkeypatch):
    """A future kind ('mlproject', etc.) must not crash the runner;
    those rows are skipped with a warning. This is the downgrade safety
    rule documented in the Phase 3 plan."""
    import ExptServer as expt_server_mod
    queue_path = tmp_path / 'runner_queue.json'
    # Hand-craft a runner_queue.json with a known-good job + an unknown
    # kind that shouldn't exist yet.
    import base64, datetime
    queue_path.write_text(json.dumps({
        'date': datetime.date.today().isoformat(),
        'next_job_id': 10,
        'queue': [
            {'id': 1, 'kind': 'job', 'state': 'queued',
             'payload': base64.b64encode(b'OK').decode('ascii'),
             'seqName': 'CoolingSeq', 'enqueued_ts': 0},
            {'id': 2, 'kind': 'mlproject_2030', 'state': 'queued',
             'description': 'a future schema'},
            {'id': 3, 'kind': 'descriptor', 'state': 'queued',
             'descriptor': '{"seq": "CoolingSeq"}',
             'label': 'kept', 'enqueued_ts': 0},
        ],
        'history': [],
    }))
    monkeypatch.setattr(expt_server_mod, 'QUEUE_PATH', str(queue_path))
    port = _free_port()
    srv = expt_server_mod.ExptServer(f'tcp://127.0.0.1:{port}')
    try:
        snap = srv.queue_list()
        # job + descriptor survive; unknown kind is gone
        kinds = sorted(r['kind'] for r in snap['queued'])
        assert kinds == ['descriptor', 'job']
    finally:
        try:
            srv.stop_worker()
        except Exception:
            pass


def test_building_descriptor_demoted_on_reload(tmp_path, monkeypatch):
    """A descriptor stuck in 'building' (mid-dispatch crash) is
    re-queued on next startup."""
    import ExptServer as expt_server_mod
    queue_path = tmp_path / 'runner_queue.json'
    import datetime
    queue_path.write_text(json.dumps({
        'date': datetime.date.today().isoformat(),
        'next_job_id': 5,
        'queue': [
            {'id': 1, 'kind': 'descriptor', 'state': 'building',
             'descriptor': '{"seq": "CoolingSeq"}',
             'enqueued_ts': 0, 'start_ts': 12345},
        ],
        'history': [],
    }))
    monkeypatch.setattr(expt_server_mod, 'QUEUE_PATH', str(queue_path))
    port = _free_port()
    srv = expt_server_mod.ExptServer(f'tcp://127.0.0.1:{port}')
    try:
        snap = srv.queue_list()
        rows = [r for r in snap['queued'] if r['kind'] == 'descriptor']
        assert len(rows) == 1
        assert rows[0]['state'] == 'queued'   # demoted
        assert rows[0]['start_ts'] is None
    finally:
        try:
            srv.stop_worker()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
