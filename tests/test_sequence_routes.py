"""Flask route smoke tests for /api/sequence/* (the Sequence tab backend).

Run in the yb_analysis env::

    python -m pytest yb_analysis/tests/test_sequence_routes.py -v
"""

import json
import os

import pytest

from yb_analysis.tests.test_sequence_folder import _seq_bytes, _CHANS


def _write_scan(seqdir):
    os.makedirs(seqdir, exist_ok=True)
    with open(os.path.join(seqdir, "point_00001__seqid_0001.seq"), "wb") as f:
        f.write(_seq_bytes("RydDet", _CHANS))
    manifest = {
        "scan_id": "20250619170317",
        "seq": "RydDetSeq",
        "scanned_axes": [{"dim": 1, "path": "Pushout.Time", "values": [1e-3]}],
        "points": [{"n": 1, "seqid": 1,
                    "file": "point_00001__seqid_0001.seq",
                    "scanned": {"Pushout.Time": 1e-3}}],
    }
    with open(os.path.join(seqdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from yb_analysis import config as yb_cfg
    monkeypatch.setattr(yb_cfg, "PATH_PREFIX", str(tmp_path))
    monkeypatch.setattr(yb_cfg, "DATA_DIR", str(tmp_path / "Data"))
    from yb_analysis.io import scan_directory as sd_mod
    monkeypatch.setattr(sd_mod, "PATH_PREFIX", str(tmp_path))

    from yb_analysis.plotting import dashboard as dash_mod
    from flask import Flask
    app = Flask("seq_routes_test")
    dash_mod._register_api_routes(app)
    app.testing = True
    return app.test_client()


def test_list_by_folder(client, tmp_path):
    scan = tmp_path / "data_20250619_170317"
    _write_scan(str(scan / "sequence"))
    r = client.get(f"/api/sequence/list?folder={scan}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["has_manifest"] is True
    assert len(body["files"]) == 1
    assert body["files"][0]["sequences"][0]["channels"] == ["TTLx", "Freqy"]
    assert body["points"][0]["scanned"]["Pushout.Time"] == 1e-3
    assert body["scanned_axes"][0]["path"] == "Pushout.Time"


def test_list_by_scan_id(client, tmp_path):
    # scan_id -> DATA_DIR/YYYYMMDD/data_YYYYMMDD_HHMMSS/sequence
    seqdir = tmp_path / "Data" / "20250619" / "data_20250619_170317" / "sequence"
    _write_scan(str(seqdir))
    r = client.get("/api/sequence/list?scan_id=20250619170317")
    assert r.status_code == 200
    assert len(r.get_json()["files"]) == 1


def test_figure_route(client, tmp_path):
    scan = tmp_path / "data_20250619_170317"
    _write_scan(str(scan / "sequence"))
    r = client.get(
        f"/api/sequence/figure?folder={scan}"
        "&file=point_00001__seqid_0001.seq&chns=TTLx,Freqy")
    assert r.status_code == 200
    assert r.mimetype == "application/json"
    fig = json.loads(r.get_data(as_text=True))
    assert "data" in fig and "layout" in fig
    names = [t.get("name") for t in fig["data"]]
    assert "TTLx" in names and "Freqy" in names
    assert fig["layout"]["xaxis"]["title"]["text"] == "Time (ms)"


def test_params_route(client, tmp_path):
    scan = tmp_path / "data_20250619_170317"
    _write_scan(str(scan / "sequence"))
    r = client.get(f"/api/sequence/params?folder={scan}"
                   "&file=point_00001__seqid_0001.seq")
    assert r.status_code == 200
    body = r.get_json()
    assert body["seq_name"] == "RydDet"
    assert body["scanned_paths"] == ["Pushout.Time"]
    # the minimal fixture carries no params block
    assert body["has_params"] is False


def test_missing_args_400(client):
    r = client.get("/api/sequence/list")
    assert r.status_code == 400


def test_empty_folder_404(client, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    r = client.get(f"/api/sequence/list?folder={empty}")
    assert r.status_code == 404


def test_bad_scan_id_400(client):
    r = client.get("/api/sequence/list?scan_id=notanumber")
    assert r.status_code == 400


def test_dump_toggle_route(client, tmp_path, monkeypatch):
    from yb_analysis.sequence import dump_toggle
    monkeypatch.setattr(dump_toggle, "_PATH", str(tmp_path / "rs.dat"))
    assert client.get("/api/sequence/dump_toggle").get_json()["on"] is False
    assert client.post("/api/sequence/dump_toggle?on=1").get_json()["on"] is True
    assert client.get("/api/sequence/dump_toggle").get_json()["on"] is True
    assert client.post("/api/sequence/dump_toggle?on=0").get_json()["on"] is False
