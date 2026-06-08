"""/api/sequence/reconstruct: route orchestration (engine-free, faked subprocess).

The actual reconstruction is the py3.8+libnacs driver (pyctrl/tools/reconstruct_scan.py),
verified in a maintenance window. Here we fake the subprocess to test the route's parsing,
the deferred-while-running path, and the bad-input guards.

    python -m pytest yb_analysis/tests/test_sequence_reconstruct_route.py -v
"""

import types

import pytest

from yb_analysis.plotting import dashboard as dash_mod


# --------------------------------------------------------------------------- #
# _parse_reconstruct_result (pure)
# --------------------------------------------------------------------------- #
def test_parse_result_finds_marker_amid_chatter():
    out = ("engine: loading bytecode...\n"
           "pulse dump 1 2 3\n"
           'RECONSTRUCT_RESULT:{"ok": true, "n_seq": 3, "approximate": false}\n')
    r = dash_mod._parse_reconstruct_result(out)
    assert r == {"ok": True, "n_seq": 3, "approximate": False}


def test_parse_result_absent_or_bad():
    assert dash_mod._parse_reconstruct_result("no marker here") is None
    assert dash_mod._parse_reconstruct_result("RECONSTRUCT_RESULT:not-json") is None
    assert dash_mod._parse_reconstruct_result("") is None


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(tmp_path, monkeypatch):
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, "PATH_PREFIX", str(tmp_path))
    monkeypatch.setattr(yb_cfg, "DATA_DIR", str(tmp_path / "Data"))
    from yb_analysis.io import scan_directory as sd_mod
    monkeypatch.setattr(sd_mod, "PATH_PREFIX", str(tmp_path))
    from flask import Flask
    app = Flask("seq_reconstruct_test")
    dash_mod._register_api_routes(app)
    app.testing = True
    return app.test_client(), tmp_path, monkeypatch


def _make_scan_dir(tmp_path):
    d = tmp_path / "Data" / "20250619" / "data_20250619_170317"
    d.mkdir(parents=True)
    return d


def _fake_subprocess(monkeypatch, stdout, returncode=0):
    import subprocess
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)
    monkeypatch.setattr(subprocess, "run", fake_run)


def _allow_driver(monkeypatch, tmp_path):
    """Point the route at an existing py + driver so it proceeds to the (faked) spawn."""
    from yb_analysis import config as yb_cfg
    py = tmp_path / "python.exe"; py.write_text("")
    pyctrl = tmp_path / "pyctrl"; (pyctrl / "tools").mkdir(parents=True)
    (pyctrl / "tools" / "reconstruct_scan.py").write_text("")
    monkeypatch.setattr(yb_cfg, "PYCTRL_PYTHON", str(py))
    monkeypatch.setattr(yb_cfg, "PYCTRL_CWD", str(pyctrl))


def test_reconstruct_missing_scan_id(client):
    cl, _, _ = client
    r = cl.post("/api/sequence/reconstruct")
    assert r.status_code == 400


def test_reconstruct_runs_driver_and_returns_result(client):
    cl, tmp_path, monkeypatch = client
    _make_scan_dir(tmp_path)
    _allow_driver(monkeypatch, tmp_path)
    monkeypatch.setattr(dash_mod, "_read_queue_data", lambda: {"running": None})
    _fake_subprocess(monkeypatch,
                     'RECONSTRUCT_RESULT:{"ok": true, "n_seq": 2, "approximate": true}\n')
    r = cl.post("/api/sequence/reconstruct?scan_id=20250619170317")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["n_seq"] == 2 and body["approximate"] is True


def test_reconstruct_deferred_while_running(client):
    cl, tmp_path, monkeypatch = client
    _make_scan_dir(tmp_path)
    _allow_driver(monkeypatch, tmp_path)
    monkeypatch.setattr(dash_mod, "_read_queue_data", lambda: {"running": {"id": 7}})
    # subprocess must NOT be called when deferred
    import subprocess
    def boom(*a, **k):
        raise AssertionError("driver spawned while a scan is running")
    monkeypatch.setattr(subprocess, "run", boom)
    r = cl.post("/api/sequence/reconstruct?scan_id=20250619170317")
    assert r.status_code == 200
    assert r.get_json()["deferred"] is True


def test_reconstruct_driver_no_result_is_500(client):
    cl, tmp_path, monkeypatch = client
    _make_scan_dir(tmp_path)
    _allow_driver(monkeypatch, tmp_path)
    monkeypatch.setattr(dash_mod, "_read_queue_data", lambda: {"running": None})
    _fake_subprocess(monkeypatch, "engine crashed, no marker\n")
    r = cl.post("/api/sequence/reconstruct?scan_id=20250619170317")
    assert r.status_code == 500
    assert r.get_json()["ok"] is False
