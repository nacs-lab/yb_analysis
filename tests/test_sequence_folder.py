"""Tests for the Sequence-tab read side: folder discovery + manifest + figure.

Run in the yb_analysis env::

    python -m pytest yb_analysis/tests/test_sequence_folder.py -v
"""

import json
import os
import struct

from yb_analysis.sequence import manifest as seqman
from yb_analysis.sequence import figure as seqfig


def _seq_bytes(name, channels):
    """Minimal single-sequence .seq (no params, no backtrace).

    channels: list of (chn_name, [(t, v, pid), ...]).
    """
    out = bytearray()
    out += struct.pack("<I", 1)                  # nseqs
    out += name.encode("latin1") + b"\x00"       # seq_name
    out += struct.pack("<I", 1)                  # seq_idx
    out += struct.pack("<I", len(channels))      # nchns
    for cname, pts in channels:
        out += cname.encode("latin1") + b"\x00"
        out += struct.pack("<I", len(pts))
        for t, v, pid in pts:
            out += struct.pack("<qdI", t, v, pid)
    out += struct.pack("<B", 0)                  # has_params
    out += struct.pack("<B", 0)                  # has_bt_info
    return bytes(out)


_CHANS = [
    ("TTLx", [(0, 0.0, 0xFFFFFFFF), (1000, 1.0, 0)]),
    ("Freqy", [(0, 80e6, 0xFFFFFFFF), (2000, 120e6, 1)]),
]


def _make_scan(tmp_path, with_manifest=True, in_subdir=True):
    scan = tmp_path / "data_20250619_170317"
    seqdir = scan / "sequence" if in_subdir else scan
    seqdir.mkdir(parents=True)
    (seqdir / "point_00001__seqid_0001.seq").write_bytes(_seq_bytes("RydDet", _CHANS))
    (seqdir / "point_00002__seqid_0002.seq").write_bytes(_seq_bytes("RydDet", _CHANS))
    if with_manifest:
        manifest = {
            "scan_id": "20250619170317",
            "seq": "RydDetSeq",
            "scanned_axes": [{"dim": 1, "path": "Pushout.Time",
                              "values": [1.0e-3, 2.0e-3]}],
            "points": [
                {"n": 1, "seqid": 1, "file": "point_00001__seqid_0001.seq",
                 "scanned": {"Pushout.Time": 1.0e-3}},
                {"n": 2, "seqid": 2, "file": "point_00002__seqid_0002.seq",
                 "scanned": {"Pushout.Time": 2.0e-3}},
            ],
        }
        (seqdir / "manifest.json").write_text(json.dumps(manifest))
    return str(scan)


def test_open_prefers_sequence_subdir(tmp_path):
    scan = _make_scan(tmp_path, in_subdir=True)
    sf = seqman.SequenceFolder.open(scan)
    assert sf is not None
    assert os.path.basename(sf.dir) == "sequence"
    assert len(sf.seq_files()) == 2


def test_open_falls_back_to_folder_with_seq(tmp_path):
    scan = _make_scan(tmp_path, with_manifest=False, in_subdir=False)
    sf = seqman.SequenceFolder.open(scan)
    assert sf is not None
    assert sf.manifest is None
    # manifest-free -> one synthetic point per file
    pts = sf.points()
    assert [p["n"] for p in pts] == [1, 2]
    assert all(p["scanned"] == {} for p in pts)


def test_open_returns_none_without_seq(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert seqman.SequenceFolder.open(str(empty)) is None


def test_index_with_manifest(tmp_path):
    scan = _make_scan(tmp_path)
    sf = seqman.SequenceFolder.open(scan)
    idx = sf.index()
    assert idx["has_manifest"] is True
    assert idx["scan_id"] == "20250619170317"
    assert len(idx["scanned_axes"]) == 1
    assert idx["scanned_axes"][0]["path"] == "Pushout.Time"
    assert len(idx["points"]) == 2
    assert len(idx["files"]) == 2
    seqs = idx["files"][0]["sequences"]
    assert seqs[0]["channels"] == ["TTLx", "Freqy"]
    assert seqs[0]["nchns"] == 2


def test_file_path_rejects_traversal(tmp_path):
    scan = _make_scan(tmp_path)
    sf = seqman.SequenceFolder.open(scan)
    # a traversal attempt is reduced to its basename (and then not found)
    import pytest
    with pytest.raises((FileNotFoundError, ValueError)):
        sf.file_path("../../etc/passwd")


def test_build_figure_dual_axis_and_customdata(tmp_path):
    scan = _make_scan(tmp_path)
    sf = seqman.SequenceFolder.open(scan)
    dump = sf.load("point_00001__seqid_0001.seq")
    seq = dump.sequences[0]

    fig = seqfig.build_sequence_figure(seq, ["TTLx", "Freqy"])
    # trace 0 is the click-highlight; then one per channel
    assert len(fig.data) == 3
    names = [t.name for t in fig.data]
    assert names[0] == "Selected"
    assert "TTLx" in names and "Freqy" in names

    by_name = {t.name: t for t in fig.data}
    # TTL on primary axis, frequency on secondary
    assert by_name["TTLx"].yaxis in (None, "y")
    assert by_name["Freqy"].yaxis == "y2"
    # pulse_id carried as customdata for later backtrace
    assert list(by_name["TTLx"].customdata) == [0xFFFFFFFF, 0]
    # x converted to ms
    assert list(by_name["TTLx"].x) == [0.0, 1000 * 1e-9]
    assert fig.layout.xaxis.title.text == "Time (ms)"


def test_build_figure_skips_unknown_channels(tmp_path):
    scan = _make_scan(tmp_path)
    sf = seqman.SequenceFolder.open(scan)
    seq = sf.load("point_00001__seqid_0001.seq").sequences[0]
    fig = seqfig.build_sequence_figure(seq, ["TTLx", "DoesNotExist"])
    assert len(fig.data) == 2  # highlight + TTLx only
