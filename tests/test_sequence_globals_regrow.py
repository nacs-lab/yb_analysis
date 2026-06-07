"""Incremental-globals rebuild gate in `_maybe_autobuild_xref`.

`globals.json` is now flushed INCREMENTALLY during a run (one seqid at a time, as each
unique compiled sequence's runtime globals are captured -- see pyctrl seq_dump). A
current-but-incomplete xref (``pending_globals`` > 0) must therefore be rebuilt EACH TIME
globals.json gains a seqid, not just once -- otherwise the seqids that land after the first
rebuild never resolve their step/wait bands. Growth is detected by the captured-seqid COUNT
in the JSON (content, never mtime: OneDrive re-bumps mtime and would crash-loop).

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_sequence_globals_regrow.py -v
"""
import json
import os

from yb_analysis.plotting import dashboard as dsh
from yb_analysis import config as _cfg


def _write_globals(seqd, seqids):
    """Write a globals.json carrying ``seqids`` (a list of seqid keys)."""
    with open(os.path.join(seqd, "globals.json"), "w", encoding="utf-8") as f:
        json.dump({"globals": {str(s): [{"id": 0, "value": 1.0}] for s in seqids}}, f)


def _make_scan(tmp_path, pending):
    """A scan dir whose CURRENT-version xref still has ``pending`` global-dependent bands,
    with a .seq + a descriptor sidecar so the autobuild gate gets as far as the rebuild
    decision."""
    name = "data_20260607_000000"
    base = tmp_path / name
    seqd = base / "sequence"
    seqd.mkdir(parents=True)
    (seqd / "point_00001__seqid_1.seq").write_bytes(b"\x00")
    (seqd / "xref.json").write_text(json.dumps({
        "scan_id": "1", "v": 99,
        "by_file": {"point_00001__seqid_1.seq": {
            "param_to_channels": {"A": ["x"]}, "pending_globals": pending}},
    }), encoding="utf-8")
    # canonical config sidecar with a descriptor -> the ScanGroup is rebuildable
    (base / (name + ".json")).write_text(json.dumps({"descriptor": {"k": "v"}}),
                                         encoding="utf-8")
    return str(base), str(seqd)


def _arm(tmp_path, monkeypatch):
    """Common monkeypatching: version matches the xref, a fake tool/python exist, and the
    build spawner is replaced by a recorder so nothing actually runs."""
    monkeypatch.setattr(dsh, "_xref_tool_version", lambda: 99)
    monkeypatch.setattr(dsh, "_XREF_GLOBALS_REBUILT_COUNT", {})
    monkeypatch.setattr(dsh, "_XREF_BUILDS", {})
    monkeypatch.setattr(dsh, "_XREF_BUILD_ERRORS", {})
    # a real (existing) python + tool path so the existence checks pass
    cwd = tmp_path / "pyctrl"
    (cwd / "tools").mkdir(parents=True)
    tool = cwd / "tools" / "provenance_scan.py"
    tool.write_text("# fake", encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    monkeypatch.setattr(_cfg, "PYCTRL_CWD", str(cwd), raising=False)
    monkeypatch.setattr(_cfg, "PYCTRL_PYTHON", str(py), raising=False)
    spawned = []
    monkeypatch.setattr(dsh, "_spawn_xref_build",
                        lambda key, py, tool, base: spawned.append(key))
    return spawned


def test_rebuild_fires_each_time_globals_grow(tmp_path, monkeypatch):
    spawned = _arm(tmp_path, monkeypatch)
    base, seqd = _make_scan(tmp_path, pending=3)

    # 1 seqid captured so far -> first rebuild
    _write_globals(seqd, [1])
    assert dsh._maybe_autobuild_xref(base) is True
    assert len(spawned) == 1

    # No growth + no in-flight build -> NO new build (count unchanged, would-be idempotent)
    assert dsh._maybe_autobuild_xref(base) is False
    assert len(spawned) == 1

    # globals.json grew to 2 seqids -> a SECOND rebuild fires (the once-guard regression)
    _write_globals(seqd, [1, 2])
    assert dsh._maybe_autobuild_xref(base) is True
    assert len(spawned) == 2


def test_no_rebuild_when_nothing_pending(tmp_path, monkeypatch):
    spawned = _arm(tmp_path, monkeypatch)
    base, seqd = _make_scan(tmp_path, pending=0)   # fully placed already
    _write_globals(seqd, [1, 2, 3])
    assert dsh._maybe_autobuild_xref(base) is False
    assert spawned == []


def test_no_rebuild_when_globals_absent(tmp_path, monkeypatch):
    spawned = _arm(tmp_path, monkeypatch)
    base, _seqd = _make_scan(tmp_path, pending=2)   # pending, but no globals.json yet
    assert dsh._maybe_autobuild_xref(base) is False
    assert spawned == []
