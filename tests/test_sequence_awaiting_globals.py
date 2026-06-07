"""Accuracy of the Sequence-tab 'awaiting globals' gate.

`globals.json` is captured at the END of a run, so global-dependent step/wait offsets (e.g.
the 616-EOM opening ramp) can't resolve until then. But MOST step boundaries + non-global
wait regions resolve WITHOUT globals -- so once the xref carries a timing map the ruler IS
drawn, and the old gate (globals.json absent + scan running) wrongly kept nagging
"waiting for globals -- only then can the ruler be placed". _xref_awaiting_globals now also
requires the timing map to still be empty.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_sequence_awaiting_globals.py -v
"""
import json
import os

from yb_analysis.plotting import dashboard as dsh


def _scan_with_xref(tmp_path, entry):
    base = tmp_path / "data_20260607_000000"
    seqd = base / "sequence"
    seqd.mkdir(parents=True)
    (seqd / "point_00001__seqid_1.seq").write_bytes(b"\x00")     # a (dummy) .seq must exist
    (seqd / "xref.json").write_text(
        json.dumps({"scan_id": "1", "v": 6, "by_file": {"point_00001__seqid_1.seq": entry}}),
        encoding="utf-8")
    return str(base)


def test_not_awaiting_when_timing_map_present(tmp_path, monkeypatch):
    # scan running, no globals.json, BUT the xref already has steps -> ruler placed, not awaiting
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": {"id": 7}})
    base = _scan_with_xref(tmp_path, {"steps": [{"label": "Init", "t0": 0.0, "t1": 20.0}]})
    assert dsh._xref_awaiting_globals(base) is False


def test_awaiting_when_timing_map_empty_and_running(tmp_path, monkeypatch):
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": {"id": 7}})
    base = _scan_with_xref(tmp_path, {"param_to_channels": {"A": ["x"]}})   # no steps/regions
    assert dsh._xref_awaiting_globals(base) is True


def test_not_awaiting_when_globals_present(tmp_path, monkeypatch):
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": {"id": 7}})
    base = _scan_with_xref(tmp_path, {"param_to_channels": {"A": ["x"]}})
    with open(os.path.join(base, "sequence", "globals.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    assert dsh._xref_awaiting_globals(base) is False


def test_not_awaiting_when_no_scan_running(tmp_path, monkeypatch):
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": None})
    base = _scan_with_xref(tmp_path, {"param_to_channels": {"A": ["x"]}})   # empty timing map
    assert dsh._xref_awaiting_globals(base) is False
