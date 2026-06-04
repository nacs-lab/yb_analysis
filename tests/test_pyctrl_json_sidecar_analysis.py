"""The analysis pane must discover + analyze pyctrl scans (.json sidecar).

The pyctrl backend writes a ``data_<stamp>.json`` scan-config sidecar instead
of a MATLAB ``data_<stamp>.mat`` (a Python backend has no reason to emit MATLAB
binary). Three places in the OFFLINE analysis path used to assume a ``.mat`` and
silently dropped / blanked pyctrl scans:

  * ``runs_list.list_runs`` gated ``complete`` on a ``.mat`` -> the scan never
    appeared in the dashboard's runs list ("cannot find the data saved by
    pyctrl").
  * ``runs_list._enrich_meta`` read name + dims only from the ``.mat``.
  * ``load_data._load_from_h5`` loaded the Scan config only from the ``.mat`` ->
    ``analyze_scan_dir`` got an EMPTY config and rendered blank curves.

These tests build a pyctrl-style scan dir on disk (``.json`` + ``.h5``, NO
``.mat``) and assert the run is listed, named, and analyzes into a real curve.

Run in the yb_analysis env:

    python -m pytest yb_analysis/tests/test_pyctrl_json_sidecar_analysis.py -v
"""

import json
import os

import numpy as np
import pytest

from yb_analysis.analysis import runs_list as RL
from yb_analysis.analysis import run_analysis as RA
from yb_analysis.analysis.run_analysis import analyze_scan_dir

h5py = pytest.importorskip("h5py")


SWEPT_PATH = "Pushout.Green.Freq"
SWEPT_VALUES = [10.0, 20.0, 30.0]
SCAN_NAME = "FakePyctrlScan"


def _write_pyctrl_scan(root, scan_id="20990101120000", n_reps=4):
    """Create ``<root>/Data/<day>/data_<stamp>/{.json,.h5}`` (no .mat).

    Mirrors what pyctrl's scan_prep.write_scan_config + the monitor's HDF5
    store produce for a 1-image loading scan: a JSON config carrying
    ScanGroup.base.vars (the swept axis), ScanName (char codes), Params
    (seq_id -> scan-point index), and an HDF5 with logicals + seq_ids.
    """
    day, hms = scan_id[:8], scan_id[8:]
    scan_dir = os.path.join(root, "Data", day, f"data_{day}_{hms}")
    os.makedirs(scan_dir, exist_ok=True)
    base = f"data_{day}_{hms}"

    n_params = len(SWEPT_VALUES)
    n_seqs = n_params * n_reps
    n_sites = 5
    # Round-robin scan-point order: shot i (1-based seq_id) ran point Params[i].
    params = [(i % n_params) + 1 for i in range(n_seqs)]
    seq_ids = np.arange(1, n_seqs + 1, dtype=np.int64)

    cfg = {
        "frameSize": [8, 8],          # MATLAB [W, H]
        "NumImages": 1,
        "NumPerGroup": n_reps,
        "isInit": 0,
        "isHC": 0,
        "isGrid2": 0,
        "scan_id": int(scan_id),
        "source": "pyctrl",
        "Params": params,
        # ScanName stored as uint16 char codes, exactly as pyctrl emits it.
        "ScanName": {"scanname": [ord(c) for c in SCAN_NAME]},
        # ScanGroup.base.vars in the {params:[dim-struct], size:[...]} shape
        # extract_scan_dims expects.
        "ScanGroup": {
            "version": 1,
            "base": {
                "vars": {
                    "params": [{"Pushout": {"Green": {"Freq": SWEPT_VALUES}}}],
                    "size": [n_params],
                },
                "params": {},
            },
        },
    }
    with open(os.path.join(scan_dir, f"{base}.json"), "w") as f:
        json.dump(cfg, f)

    rng = np.random.default_rng(0)
    logicals = rng.random((n_seqs, n_sites)) < 0.55
    with h5py.File(os.path.join(scan_dir, f"{base}.h5"), "w") as f:
        f.create_dataset("logicals", data=logicals)
        f.create_dataset("seq_ids", data=seq_ids)

    return scan_dir, scan_id


def test_list_runs_finds_json_sidecar_scan(tmp_path, monkeypatch):
    scan_dir, scan_id = _write_pyctrl_scan(str(tmp_path))
    monkeypatch.setattr(RL._yb_cfg, "PATH_PREFIX", str(tmp_path))

    rows = RL.list_runs(since_days=None, with_meta=True)
    row = next((r for r in rows if r["scan_id"] == scan_id), None)

    assert row is not None, "pyctrl (.json) scan was dropped from the runs list"
    assert row["complete"] is True
    assert row["name"] == SCAN_NAME            # decoded from char codes
    assert row["n_shots"] == 4                 # NumPerGroup
    assert row["n_params"] == 3
    assert SWEPT_PATH in (row.get("swept") or "")


def test_analyze_scan_dir_on_json_sidecar(tmp_path):
    scan_dir, _ = _write_pyctrl_scan(str(tmp_path))

    res = analyze_scan_dir(scan_dir)

    assert res.get("unpack_error") is None
    assert res["scan_name"] == SCAN_NAME
    assert res["n_params"] == 3
    assert res["n_sites"] == 5
    sweep = res.get("sweep") or {}
    assert sweep.get("cols") == [SWEPT_PATH]    # dotted path, not "axis0"
    assert sweep.get("dims") == [3]
    # A loading curve (one rate per scan point) is produced from the .h5.
    lr = (res.get("summary") or {}).get("loading_rate") or []
    assert len(lr) == 3
    assert all(0.0 <= x <= 1.0 for x in lr)


def test_str_or_none_decodes_float_char_codes():
    # _config_arrays floats numeric JSON lists, so a scanname stored as char
    # codes arrives as a float ndarray; _str_or_none must still decode it.
    codes = np.array([float(ord(c)) for c in "LACScan"])
    assert RA._str_or_none(codes) == "LACScan"


def test_str_or_none_leaves_real_float_arrays_alone():
    # A genuine numeric field (non-integral floats) must NOT be mis-decoded.
    assert RA._str_or_none(np.array([10.5, 20.25, 30.75])) is not None  # str repr, not chars
    decoded = RA._str_or_none(np.array([0.1, 0.2, 0.3]))
    # Non-integral -> not treated as char codes (no chr() of fractional values).
    assert decoded is None or not decoded.startswith("\x00")
