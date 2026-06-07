"""Sequence-tab LIVE refresh helpers (incremental-globals era).

globals.json now flushes per-seqid during the run, so the viewed point resolves immediately
and the old per-point globals-pending poll loops never start. The dashboard instead drives a
single live loop off three server helpers tested here:
  * `_seq_scan_is_running`  -- is THIS scan the one currently running (precise match)?
  * `_xref_scan_pending`    -- scan-WIDE count of points whose bands aren't placed yet.
  * `_seq_live_status`      -- the combined block both Sequence routes return.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_sequence_live_refresh.py -v
"""
import json
import os

from yb_analysis.plotting import dashboard as dsh


def _xref(seqd, entries):
    """Write an xref.json with one by_file entry per (name, pending) in ``entries``."""
    by_file = {name: {"param_to_channels": {"A": ["x"]}, "pending_globals": pend}
               for name, pend in entries}
    (seqd / "xref.json").write_text(json.dumps({"v": 8, "by_file": by_file}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# _xref_scan_pending -- scan-wide aggregate
# --------------------------------------------------------------------------- #
def test_scan_pending_counts_points_and_bands(tmp_path):
    seqd = tmp_path / "sequence"
    seqd.mkdir()
    # 2 placed (pending 0), 3 not-yet-run (pending 4 + 1 + 2 bands)
    _xref(seqd, [("a", 0), ("b", 0), ("c", 4), ("d", 1), ("e", 2)])
    out = dsh._xref_scan_pending(str(seqd))
    assert out == {"pending_points": 3, "total_points": 5, "pending_bands": 7}


def test_scan_pending_all_placed(tmp_path):
    seqd = tmp_path / "sequence"
    seqd.mkdir()
    _xref(seqd, [("a", 0), ("b", 0)])
    out = dsh._xref_scan_pending(str(seqd))
    assert out == {"pending_points": 0, "total_points": 2, "pending_bands": 0}


def test_scan_pending_no_xref(tmp_path):
    seqd = tmp_path / "sequence"
    seqd.mkdir()
    assert dsh._xref_scan_pending(str(seqd)) == {
        "pending_points": 0, "total_points": 0, "pending_bands": 0}


# --------------------------------------------------------------------------- #
# _seq_scan_is_running -- precise "this scan is the running one" match
# --------------------------------------------------------------------------- #
def test_is_running_matches_loaded_scan(tmp_path, monkeypatch):
    base = str(tmp_path / "data_20260607_120000")
    os.makedirs(base)
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": {"id": 7}})
    monkeypatch.setattr(dsh, "_read_data", lambda: {"scan_id": "20260607120000"})
    monkeypatch.setattr(dsh, "_resolve_scan_dir", lambda sid: base)
    assert dsh._seq_scan_is_running(base) is True


def test_is_running_false_for_other_scan(tmp_path, monkeypatch):
    base = str(tmp_path / "data_20260607_120000")
    other = str(tmp_path / "data_20260607_999999")
    os.makedirs(base)
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": {"id": 7}})
    monkeypatch.setattr(dsh, "_read_data", lambda: {"scan_id": "20260607999999"})
    monkeypatch.setattr(dsh, "_resolve_scan_dir", lambda sid: other)
    assert dsh._seq_scan_is_running(base) is False     # a different scan is running


def test_is_running_false_when_idle(tmp_path, monkeypatch):
    base = str(tmp_path / "data_20260607_120000")
    os.makedirs(base)
    monkeypatch.setattr(dsh, "_read_queue_data", lambda: {"running": None})
    monkeypatch.setattr(dsh, "_read_data", lambda: {"scan_id": "20260607120000"})
    monkeypatch.setattr(dsh, "_resolve_scan_dir", lambda sid: base)
    assert dsh._seq_scan_is_running(base) is False     # nothing running


# --------------------------------------------------------------------------- #
# _seq_live_status -- combined block (running + pending + .seq count)
# --------------------------------------------------------------------------- #
def test_live_status_combines_running_and_pending(tmp_path, monkeypatch):
    base = tmp_path / "data_20260607_120000"
    seqd = base / "sequence"
    seqd.mkdir(parents=True)
    (seqd / "point_00001__seqid_1.seq").write_bytes(b"\x00")
    (seqd / "point_00002__seqid_2.seq").write_bytes(b"\x00")
    _xref(seqd, [("point_00001__seqid_1.seq", 0), ("point_00002__seqid_2.seq", 0),
                 ("point_00003__seqid_3.seq", 5)])     # 3rd point not run yet
    monkeypatch.setattr(dsh, "_seq_scan_is_running", lambda b: True)
    out = dsh._seq_live_status(str(base))
    assert out["running"] is True
    assert out["n_seq_files"] == 2                      # 2 dumped (selectable) points
    assert out["pending_points"] == 1                   # 1 future point still pending
    assert out["total_points"] == 3
    assert out["pending_bands"] == 5
