"""/api/nidaq/monitor: route orchestration (no hardware; faked subprocess).

The NI DAC monitor reads the PCIe-6738's internal AO monitors via the engine-venv driver
(yb_analysis/control/ni_monitor_driver.py), verified on hardware in a maintenance window.
Here we fake the subprocess to test the route's gate, the deferred-while-running path,
the result parsing, and the short-TTL cache -- all without touching the NI card.

    python -m pytest yb_analysis/tests/test_nidaq_monitor_route.py -v
"""

import types

import pytest

from yb_analysis.plotting import dashboard as dash_mod


# --------------------------------------------------------------------------- #
# _parse_nidaq_result (pure)
# --------------------------------------------------------------------------- #
def test_parse_result_finds_marker_amid_chatter():
    out = ("engine: importing nidaqmx...\n"
           "some warning\n"
           'NI_MONITOR_RESULT:{"ok": true, "device": "Dev1", "channels": []}\n')
    r = dash_mod._parse_nidaq_result(out)
    assert r == {"ok": True, "device": "Dev1", "channels": []}


def test_parse_result_absent_or_bad():
    assert dash_mod._parse_nidaq_result("no marker") is None
    assert dash_mod._parse_nidaq_result("NI_MONITOR_RESULT:not-json") is None
    assert dash_mod._parse_nidaq_result("") is None


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(tmp_path, monkeypatch):
    from flask import Flask
    app = Flask("nidaq_monitor_test")
    dash_mod._register_nidaq_routes(app)
    app.testing = True
    # Fresh cache per test (module-level state).
    monkeypatch.setattr(dash_mod, "_NIDAQ_CACHE", {"data": None, "ts": 0.0})
    # Idle by default; tests that need "running" override this.
    monkeypatch.setattr(dash_mod, "_read_queue_data", lambda: {"running": None})
    return app.test_client(), tmp_path, monkeypatch


def _allow_driver(monkeypatch, tmp_path):
    """Point the route at an existing py + pyctrl root so it proceeds to the (faked) spawn.
    The driver path is real (it ships in the repo), so only PYCTRL_PYTHON / PYCTRL_CWD
    need to exist."""
    from yb_analysis import config as yb_cfg
    py = tmp_path / "python.exe"; py.write_text("")
    pyctrl = tmp_path / "pyctrl"; pyctrl.mkdir()
    monkeypatch.setattr(yb_cfg, "PYCTRL_PYTHON", str(py))
    monkeypatch.setattr(yb_cfg, "PYCTRL_CWD", str(pyctrl))


def _fake_subprocess(monkeypatch, stdout, returncode=0):
    import subprocess
    calls = {"n": 0}
    def fake_run(cmd, **kw):
        calls["n"] += 1
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_disabled_gate_returns_403(client):
    cl, _tmp, monkeypatch = client
    monkeypatch.setenv("YB_NIDAQ_MONITOR", "0")
    r = cl.get("/api/nidaq/monitor")
    assert r.status_code == 403
    body = r.get_json()
    assert body["ok"] is False and body["disabled"] is True


def test_paused_while_scan_running(client):
    cl, _tmp, monkeypatch = client
    monkeypatch.setenv("YB_NIDAQ_MONITOR", "1")
    monkeypatch.setattr(dash_mod, "_read_queue_data", lambda: {"running": {"id": 7}})
    r = cl.get("/api/nidaq/monitor")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["paused"] is True
    assert body["channels"] == []        # no cache yet


def test_runs_driver_and_returns_channels(client):
    cl, tmp_path, monkeypatch = client
    monkeypatch.setenv("YB_NIDAQ_MONITOR", "1")
    _allow_driver(monkeypatch, tmp_path)
    _fake_subprocess(monkeypatch,
                     'NI_MONITOR_RESULT:{"ok": true, "device": "Dev1", "channels": '
                     '[{"alias": "VElectrode1", "chn": 12, "voltage": 0.5, '
                     '"default": 0.0, "error": null}]}\n')
    r = cl.get("/api/nidaq/monitor")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["device"] == "Dev1"
    assert body["channels"][0]["alias"] == "VElectrode1"


def test_second_call_served_from_cache(client):
    cl, tmp_path, monkeypatch = client
    monkeypatch.setenv("YB_NIDAQ_MONITOR", "1")
    _allow_driver(monkeypatch, tmp_path)
    calls = _fake_subprocess(monkeypatch,
                             'NI_MONITOR_RESULT:{"ok": true, "device": "Dev1", '
                             '"channels": []}\n')
    cl.get("/api/nidaq/monitor")                 # populates cache (1 spawn)
    r = cl.get("/api/nidaq/monitor")             # within TTL -> cache, no new spawn
    body = r.get_json()
    assert calls["n"] == 1
    assert body["ok"] is True and body.get("cached") is True


def test_driver_no_result_is_error(client):
    cl, tmp_path, monkeypatch = client
    monkeypatch.setenv("YB_NIDAQ_MONITOR", "1")
    _allow_driver(monkeypatch, tmp_path)
    _fake_subprocess(monkeypatch, "engine chatter, no marker\n")
    r = cl.get("/api/nidaq/monitor")
    assert r.status_code == 500
    assert r.get_json()["ok"] is False
