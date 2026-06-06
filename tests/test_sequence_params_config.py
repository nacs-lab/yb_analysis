"""params_from_config: build the PARAMETERS tree from a scan's .json sidecar (§12.2).

Engine-free. Run in the yb_analysis env::

    python -m pytest yb_analysis/tests/test_sequence_params_config.py -v
"""

import json
import os

import pytest

from yb_analysis.sequence import params_from_config as pfc


# --------------------------------------------------------------------------- #
# A representative sidecar: full expConfig baseline + a few base.params
# overrides + one swept axis.
# --------------------------------------------------------------------------- #
def _config():
    return {
        "expConfig": {
            "Camera": {"ExposureTime": 0.1, "ROI": [0, 0, 100, 100]},
            "Pushout": {"Green": {"Amp": 0.5}},
            "Init": {"EOM616": {"Freq": 3.2e8}},
        },
        "ScanGroup": {
            "version": 1,
            "base": {
                "params": {
                    "Pushout": {"Green": {"Amp": 0.8}},   # config leaf changed -> type 3
                    "Init": {"EOM616": {"Freq": 3.2e8}},  # override == baseline -> type 1
                    "Extra": {"Knob": 7},                 # not in expConfig -> type 2
                },
                "vars": {
                    "params": [{"Pushout": {"Time": [1e-3, 2e-3, 3e-3]}}],
                    "size": [3],
                },
            },
        },
    }


def test_build_params_tree_status_codes():
    built = pfc.build_params_tree(_config())
    tree = built["params"]

    # config leaf overwritten with a DIFFERENT value -> type 3, config_value kept
    amp = tree["Pushout"]["Green"]["Amp"]
    assert amp == {"value": 0.8, "type": 3, "config_value": 0.5}

    # config leaf "overwritten" with the SAME value -> stays type 1 (not modified)
    freq = tree["Init"]["EOM616"]["Freq"]
    assert freq["type"] == 1 and freq["value"] == freq["config_value"] == 3.2e8

    # base.params leaf with no config baseline -> type 2 (overwritten ordinary)
    knob = tree["Extra"]["Knob"]
    assert knob == {"value": 7, "type": 2}

    # an untouched config leaf -> type 1 config
    assert tree["Camera"]["ExposureTime"] == {"value": 0.1, "type": 1, "config_value": 0.1}

    # swept axis injected as a type-0 leaf so the scanned highlight has a target
    assert tree["Pushout"]["Time"] == {"value": [1e-3, 2e-3, 3e-3], "type": 0}

    assert built["scanned_paths"] == ["Pushout.Time"]
    assert built["stats"] == {"n_leaves": 6, "n_modified": 2, "n_scanned": 1}
    assert built["has_params"] is True


def test_build_params_tree_no_expconfig_shows_params_plainly():
    """Older scans without an expConfig snapshot -> base.params shown as plain config
    (type 1), not all flagged modified."""
    cfg = {"ScanGroup": {"base": {"params": {"A": {"B": 5}}, "vars": {}}}}
    built = pfc.build_params_tree(cfg)
    assert built["params"]["A"]["B"]["type"] == 1
    assert built["stats"]["n_modified"] == 0


def test_build_params_tree_empty():
    built = pfc.build_params_tree({})
    assert built["has_params"] is False
    assert built["params"] == {}
    assert built["scanned_paths"] == []


def test_find_config_sidecar(tmp_path):
    d = tmp_path / "data_20250619_170317"
    d.mkdir()
    sidecar = d / "data_20250619_170317.json"
    sidecar.write_text("{}")
    assert pfc.find_config_sidecar(str(d)) == str(sidecar)
    # from the sequence/ subdir -> finds the parent's sidecar
    (d / "sequence").mkdir()
    assert pfc.find_config_sidecar(str(d / "sequence")) == str(sidecar)


def test_find_config_sidecar_ignores_other_json(tmp_path):
    d = tmp_path / "data_20250101_000000"
    d.mkdir()
    (d / "analysis_cache.json").write_text("{}")
    (d / "slm_grid.json").write_text("{}")
    assert pfc.find_config_sidecar(str(d)) is None


# --------------------------------------------------------------------------- #
# Route: /api/sequence/params prefers the .json sidecar.
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
    app = Flask("seq_params_cfg_test")
    dash_mod._register_api_routes(app)
    app.testing = True
    return app.test_client()


def test_params_route_prefers_config_sidecar(client, tmp_path):
    scandir = tmp_path / "Data" / "20250619" / "data_20250619_170317"
    scandir.mkdir(parents=True)
    (scandir / "data_20250619_170317.json").write_text(json.dumps(_config()))

    r = client.get("/api/sequence/params?scan_id=20250619170317")
    assert r.status_code == 200
    body = r.get_json()
    assert body["source"] == "config"
    assert body["scanned_paths"] == ["Pushout.Time"]
    assert body["params"]["Pushout"]["Green"]["Amp"]["type"] == 3
    assert body["stats"]["n_modified"] == 2
    # no .seq needed for the params card
    assert not os.path.isdir(scandir / "sequence")
