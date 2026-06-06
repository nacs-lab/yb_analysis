"""The dashboard-side auto-dump toggle (mmap-layout mirror of pyctrl runtime_state).

    python -m pytest yb_analysis/tests/test_dump_toggle.py -v
"""

import struct

from yb_analysis.sequence import dump_toggle


def test_default_off(tmp_path, monkeypatch):
    monkeypatch.setattr(dump_toggle, "_PATH", str(tmp_path / "rs.dat"))
    assert dump_toggle.get_save_sequence_dumps() is False


def test_set_get_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(dump_toggle, "_PATH", str(tmp_path / "rs.dat"))
    assert dump_toggle.set_save_sequence_dumps(True) is True
    assert dump_toggle.get_save_sequence_dumps() is True
    dump_toggle.set_save_sequence_dumps(False)
    assert dump_toggle.get_save_sequence_dumps() is False


def test_dashboard_first_write_leaves_eom_slot_unset(tmp_path, monkeypatch):
    # A dashboard-first write must produce the full 9-byte layout with a NaN freq at
    # offset 0 (so pyctrl reads its 616-EOM slot as "unset") and the flag at offset 8.
    p = tmp_path / "rs.dat"
    monkeypatch.setattr(dump_toggle, "_PATH", str(p))
    dump_toggle.set_save_sequence_dumps(True)
    data = p.read_bytes()
    assert len(data) == 9
    freq = struct.unpack("<d", data[0:8])[0]
    assert freq != freq        # NaN
    assert data[8] == 1


def test_preserves_existing_eom_freq(tmp_path, monkeypatch):
    # A legacy 8-byte EOM-only file keeps its freq when the dashboard first writes the flag.
    p = tmp_path / "rs.dat"
    p.write_bytes(struct.pack("<d", 555666.0))
    monkeypatch.setattr(dump_toggle, "_PATH", str(p))
    dump_toggle.set_save_sequence_dumps(True)
    data = p.read_bytes()
    assert struct.unpack("<d", data[0:8])[0] == 555666.0
    assert data[8] == 1
