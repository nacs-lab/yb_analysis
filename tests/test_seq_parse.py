"""Tests for the dashboard-side ``.seq`` reader (yb_analysis/sequence/seq_parse).

Run in the yb_analysis env::

    python -m pytest yb_analysis/tests/test_seq_parse.py -v
"""

import os
import struct

import numpy as np
import pytest

from yb_analysis.sequence import seq_parse
from yb_analysis.sequence.seq_parse import PULSE_ID_DEFAULT


# --------------------------------------------------------------------------- #
# A small hand-built .seq, so the parser is checked against bytes we control
# (independent of the pyctrl/MATLAB writers).
# --------------------------------------------------------------------------- #
def _u32(x):
    return struct.pack("<I", x & 0xFFFFFFFF)


def _cstr(s):
    return s.encode("latin1") + b"\x00"


def _point(t, v, pid):
    return struct.pack("<qdI", t, v, pid)


def _build_minimal_seq():
    out = bytearray()
    out += _u32(1)                       # nseqs
    # --- seq 0 ---
    out += _cstr("probe")                # seq_name
    out += _u32(1)                       # seq_idx
    out += _u32(2)                       # nchns
    # chn 0: a TTL-like channel (values < 1e6)
    out += _cstr("TTLx")
    out += _u32(2)
    out += _point(0, 0.0, PULSE_ID_DEFAULT)
    out += _point(1000, 1.0, 0)
    # chn 1: a frequency channel (values >= 1e6 -> secondary axis)
    out += _cstr("Freqy")
    out += _u32(2)
    out += _point(0, 80e6, PULSE_ID_DEFAULT)
    out += _point(2000, 120e6, 1)
    # params
    params_json = (
        '{"A":{"value":1.0,"type":1,"config_value":1.0},'
        '"B":{"value":2.0,"type":2,"old_value":1.0,"scanned":true}}'
    )
    out += struct.pack("<B", 1)          # has_params
    out += _cstr(params_json)
    # backtrace block
    out += struct.pack("<B", 1)          # has_bt_info
    out += _u32(0)                       # bt_idx[0]
    out += _u32(1)                       # n_bts
    # payload 0
    out += _u32(2)                       # nfilenames
    out += _cstr("RydDetSeq.py")
    out += _cstr("exp_seq.py")
    out += _u32(2)                       # nnames
    out += _cstr("add")
    out += _cstr("add_step")
    out += _u32(2)                       # nobjs
    # obj 0 (pulse_id 0): two frames
    out += _u32(2)
    out += _u32(0) + _u32(0) + _u32(12)  # RydDetSeq.py:add:12
    out += _u32(1) + _u32(1) + _u32(99)  # exp_seq.py:add_step:99
    # obj 1 (pulse_id 1): one frame
    out += _u32(1)
    out += _u32(0) + _u32(0) + _u32(20)  # RydDetSeq.py:add:20
    return bytes(out)


def test_minimal_roundtrip_fields():
    dump = seq_parse.parse(_build_minimal_seq())
    assert len(dump) == 1
    assert dump.has_bt_info is True

    s = dump.sequences[0]
    assert s.name == "probe"
    assert s.seq_idx == 1
    assert s.channel_names == ["TTLx", "Freqy"]

    ttl = s.channel("TTLx")
    np.testing.assert_array_equal(ttl.t, np.array([0, 1000], dtype=np.int64))
    np.testing.assert_array_equal(ttl.v, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(ttl.pid, np.array([PULSE_ID_DEFAULT, 0], dtype=np.uint32))
    assert ttl.is_frequency is False
    # ticks -> ms convention (1e-9)
    np.testing.assert_allclose(ttl.t_ms, np.array([0.0, 1000 * 1e-9]))

    freq = s.channel("Freqy")
    assert freq.is_frequency is True


def test_minimal_params_and_scanned_marker():
    s = seq_parse.parse(_build_minimal_seq()).sequences[0]
    assert set(s.params) == {"A", "B"}
    assert s.params["A"]["type"] == 1
    assert s.params["A"]["config_value"] == 1.0
    # the enriched "scanned" marker survives the round trip
    assert s.params["B"].get("scanned") is True
    assert s.params["B"]["old_value"] == 1.0


def test_minimal_backtrace_resolution():
    s = seq_parse.parse(_build_minimal_seq()).sequences[0]

    frames0 = s.backtrace(0)
    assert [(f.file, f.name, f.line) for f in frames0] == [
        ("RydDetSeq.py", "add", 12),
        ("exp_seq.py", "add_step", 99),
    ]

    frames1 = s.backtrace(1)
    assert [(f.file, f.name, f.line) for f in frames1] == [("RydDetSeq.py", "add", 20)]

    # default sentinel and out-of-range -> no frames
    assert s.backtrace(PULSE_ID_DEFAULT) == []
    assert s.backtrace(999) == []


def test_decode_rejects_truncated_and_trailing():
    good = _build_minimal_seq()
    with pytest.raises(ValueError):
        seq_parse.decode(good[:-4])          # truncated mid-backtrace
    with pytest.raises(ValueError):
        seq_parse.decode(good + b"\x00\x00")  # trailing bytes


# --------------------------------------------------------------------------- #
# The real 132 KB MATLAB sample (committed under pyctrl/tests/reference).
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SAMPLE = os.path.join(_REPO_ROOT, "pyctrl", "tests", "reference",
                       "seqplotter_sample_ryddet.seq")


@pytest.mark.skipif(not os.path.exists(_SAMPLE), reason="reference .seq not present")
def test_real_matlab_sample():
    dump = seq_parse.load(_SAMPLE)
    assert len(dump) == 1
    assert dump.has_bt_info is True

    s = dump.sequences[0]
    assert s.seq_idx == 1
    assert s.name == "20250619_142657:RydDet"
    assert len(s.channels) == 73
    assert sum(c.t.size for c in s.channels) == 2820
    assert len(s.params) == 235

    # a known channel from the recon
    assert any("FPGA1/TTL23" in name for name in s.channel_names)

    # at least one real (non-default) pulse resolves to a non-empty backtrace
    resolved = 0
    for c in s.channels:
        for pid in np.unique(c.pid):
            if pid != PULSE_ID_DEFAULT and s.backtrace(int(pid)):
                resolved += 1
                break
        if resolved:
            break
    assert resolved >= 1
