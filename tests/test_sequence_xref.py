"""xref: param<->channel PROVENANCE reader (gated behind the engine-built artifact).

Value-coincidence matching was removed as inaccurate (plan §8); the feature now reads a
``sequence/xref.json`` provenance artifact produced by the engine build and is dormant
(``available=False``) until that artifact exists.

Run in the yb_analysis env::

    python -m pytest yb_analysis/tests/test_sequence_xref.py -v
"""

import json
import os

import pytest

from yb_analysis.sequence import xref
from yb_analysis.tests.test_sequence_folder import _seq_bytes, _CHANS


_ENTRY = {
    "param_to_channels": {"Init.EOM616.Freq": ["FreqEOM616"]},
    "channel_to_params": {"FreqEOM616": ["Init.EOM616.Freq"]},
}


_DORMANT = {"available": False, "version": 0, "param_to_channels": {},
            "channel_to_params": {}, "pulses": {}, "param_to_pids": {}, "time_regions": {}}


def test_load_xref_exposes_version(tmp_path):
    import json as _json
    seq_dir = tmp_path / "sequence"
    seq_dir.mkdir()
    (seq_dir / "xref.json").write_text(_json.dumps(
        {"v": 3, "by_file": {"only.seq": {"param_to_channels": {}, "channel_to_params": {}}}}))
    assert xref.load_xref(str(seq_dir), "only.seq")["version"] == 3
    # a pre-versioning artifact reports version 0 (the viewer rebuilds it)
    xref.write_xref(str(seq_dir), {"only.seq": _ENTRY})
    assert xref.load_xref(str(seq_dir), "only.seq")["version"] == 0


def test_load_xref_absent_is_dormant(tmp_path):
    assert xref.load_xref(str(tmp_path), "point_00001__seqid_1.seq") == _DORMANT
    assert xref.load_xref(None) == _DORMANT


def test_write_then_load_by_file(tmp_path):
    seq_dir = str(tmp_path / "sequence")
    xref.write_xref(seq_dir, {"point_00001__seqid_1.seq": _ENTRY}, scan_id="20250619170317")
    out = xref.load_xref(seq_dir, "point_00001__seqid_1.seq")
    assert out["available"] is True
    assert out["param_to_channels"]["Init.EOM616.Freq"] == ["FreqEOM616"]
    assert out["channel_to_params"]["FreqEOM616"] == ["Init.EOM616.Freq"]
    # a file with no provenance entry stays dormant
    assert xref.load_xref(seq_dir, "other.seq")["available"] is False


def test_load_xref_single_entry_when_fname_none(tmp_path):
    seq_dir = str(tmp_path / "sequence")
    xref.write_xref(seq_dir, {"only.seq": _ENTRY})
    assert xref.load_xref(seq_dir)["available"] is True


def test_load_xref_exposes_per_pulse_region_maps(tmp_path):
    seq_dir = str(tmp_path / "sequence")
    entry = {
        "param_to_channels": {"A": ["CH1"]},
        "channel_to_params": {"CH1": ["A"]},
        "pulses": {"60": {"channel": "CH1", "params": ["A"]}},
        "param_to_pids": {"A": [60]},
    }
    xref.write_xref(seq_dir, {"only.seq": entry})
    out = xref.load_xref(seq_dir, "only.seq")
    assert out["pulses"]["60"] == {"channel": "CH1", "params": ["A"]}
    assert out["param_to_pids"]["A"] == [60]


def test_load_xref_pre_region_artifact_defaults_empty(tmp_path):
    # Older artifact (no pulses/param_to_pids) still loads; region maps default empty.
    seq_dir = str(tmp_path / "sequence")
    xref.write_xref(seq_dir, {"only.seq": {"param_to_channels": {"A": ["CH1"]},
                                           "channel_to_params": {"CH1": ["A"]}}})
    out = xref.load_xref(seq_dir, "only.seq")
    assert out["available"] is True
    assert out["pulses"] == {} and out["param_to_pids"] == {}


# --------------------------------------------------------------------------- #
# Route: /api/sequence/xref reflects the artifact (dormant without it).
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(tmp_path, monkeypatch):
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, "PATH_PREFIX", str(tmp_path))
    monkeypatch.setattr(yb_cfg, "DATA_DIR", str(tmp_path / "Data"))
    from yb_analysis.io import scan_directory as sd_mod
    monkeypatch.setattr(sd_mod, "PATH_PREFIX", str(tmp_path))
    from yb_analysis.plotting import dashboard as dash_mod
    from flask import Flask
    app = Flask("seq_xref_test")
    dash_mod._register_api_routes(app)
    app.testing = True
    return app.test_client()


def _make_scan_with_seq(tmp_path):
    scan = tmp_path / "data_20250619_170317"
    seqdir = scan / "sequence"
    seqdir.mkdir(parents=True)
    (seqdir / "point_00001__seqid_1.seq").write_bytes(_seq_bytes("RydDet", _CHANS))
    return scan, seqdir


def test_xref_route_dormant_without_artifact(client, tmp_path):
    scan, _ = _make_scan_with_seq(tmp_path)
    r = client.get("/api/sequence/xref?folder=" + str(scan) +
                   "&file=point_00001__seqid_1.seq")
    assert r.status_code == 200
    assert r.get_json()["available"] is False


def test_xref_route_returns_artifact(client, tmp_path):
    scan, seqdir = _make_scan_with_seq(tmp_path)
    xref.write_xref(str(seqdir), {"point_00001__seqid_1.seq": _ENTRY})
    r = client.get("/api/sequence/xref?folder=" + str(scan) +
                   "&file=point_00001__seqid_1.seq")
    assert r.status_code == 200
    body = r.get_json()
    assert body["available"] is True
    assert body["param_to_channels"]["Init.EOM616.Freq"] == ["FreqEOM616"]


# --------------------------------------------------------------------------- #
# Route: POST /api/sequence/build_xref spawns the background producer.
# --------------------------------------------------------------------------- #
def _write_sidecar(scan, *, descriptor):
    doc = {"scan_id": 20250619170317}
    if descriptor:
        doc["descriptor"] = {"seq": "X", "scangroup": {}}
    (scan / (scan.name + ".json")).write_text(json.dumps(doc))


def test_build_xref_already_built_short_circuits(client, tmp_path):
    scan, seqdir = _make_scan_with_seq(tmp_path)
    xref.write_xref(str(seqdir), {"point_00001__seqid_1.seq": _ENTRY})
    r = client.post("/api/sequence/build_xref?folder=" + str(scan))
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "available": True}


def test_build_xref_no_descriptor_is_inert(client, tmp_path):
    scan, _ = _make_scan_with_seq(tmp_path)
    _write_sidecar(scan, descriptor=False)
    r = client.post("/api/sequence/build_xref?folder=" + str(scan))
    body = r.get_json()
    assert body["ok"] is False and "descriptor" in body["error"]


def test_build_xref_spawns_with_descriptor(client, tmp_path, monkeypatch):
    import subprocess
    scan, _ = _make_scan_with_seq(tmp_path)
    _write_sidecar(scan, descriptor=True)
    # Fake pyctrl python + tool so the existence checks pass without the submodule.
    pyc = tmp_path / "pyctrl"
    (pyc / "tools").mkdir(parents=True)
    (pyc / "tools" / "provenance_scan.py").write_text("# stub")
    fake_py = tmp_path / "python.exe"
    fake_py.write_text("")
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, "PYCTRL_PYTHON", str(fake_py), raising=False)
    monkeypatch.setattr(yb_cfg, "PYCTRL_CWD", str(pyc), raising=False)

    calls = {}

    class _FakeProc:
        def poll(self):
            return None                          # still running -> dedupe sees it

    def _fake_popen(args, **kw):
        calls["args"] = args
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    r = client.post("/api/sequence/build_xref?folder=" + str(scan))
    body = r.get_json()
    assert body["ok"] is True and body["started"] is True
    assert calls["args"][0] == str(fake_py)
    assert "provenance_scan.py" in calls["args"][1]
    assert "--scan-dir" in calls["args"]


def test_build_xref_force_rebuilds_existing(client, tmp_path, monkeypatch):
    import subprocess
    scan, seqdir = _make_scan_with_seq(tmp_path)
    _write_sidecar(scan, descriptor=True)
    xref.write_xref(str(seqdir), {"point_00001__seqid_1.seq": _ENTRY})   # pre-region artifact
    # without force -> short-circuit (existing artifact wins)
    assert client.post("/api/sequence/build_xref?folder=" + str(scan)).get_json() == {
        "ok": True, "available": True}
    # with force -> spawns a rebuild even though an artifact exists (upgrade path)
    pyc = tmp_path / "pyctrl"
    (pyc / "tools").mkdir(parents=True)
    (pyc / "tools" / "provenance_scan.py").write_text("# stub")
    fake_py = tmp_path / "python.exe"
    fake_py.write_text("")
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, "PYCTRL_PYTHON", str(fake_py), raising=False)
    monkeypatch.setattr(yb_cfg, "PYCTRL_CWD", str(pyc), raising=False)
    calls = {}

    class _FakeProc:
        def poll(self):
            return None

    def _fake_popen(args, **kw):
        calls["args"] = args
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    r = client.post("/api/sequence/build_xref?folder=" + str(scan) + "&force=1")
    assert r.get_json().get("started") is True
    assert "--scan-dir" in calls["args"]
