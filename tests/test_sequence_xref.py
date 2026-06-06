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


def test_load_xref_absent_is_dormant(tmp_path):
    out = xref.load_xref(str(tmp_path), "point_00001__seqid_1.seq")
    assert out == {"available": False, "param_to_channels": {}, "channel_to_params": {}}
    assert xref.load_xref(None) == {"available": False, "param_to_channels": {},
                                    "channel_to_params": {}}


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
