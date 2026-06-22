"""Tests for the molecube client <-> mock daemon round-trip and unit conversions.

These exercise the FULL ZMQ wire protocol against the in-process mock (no real hardware,
no live daemon). Run in the yb_analysis env:
    python -m pytest yb_analysis/tests/test_molecube_client.py -v
"""

import pytest

from yb_analysis.control import molecube_client as mc
from yb_analysis.control.mock_molecube_server import MockMolecubeServer


# -- unit conversions (verified in-lab 2026-06-02) --------------------------
def test_freq_conversion_known_point():
    # ftw 1227134 <-> 1.000 MHz
    assert mc.ftw_to_hz(1227134) == pytest.approx(1.0e6, rel=1e-5)
    assert mc.hz_to_ftw(1.0e6) == 1227134


def test_amp_conversion_known_point():
    # amp word 410 <-> 0.100
    assert mc.amp_to_frac(410) == pytest.approx(0.1001, abs=1e-3)
    assert mc.frac_to_amp(0.1) == 410


def test_phase_conversion_roundtrip():
    for deg in (0, 45, 90, 180, 270, 359):
        assert mc.phase_to_deg(mc.deg_to_phase(deg)) == pytest.approx(deg, abs=0.01)


def test_amp_clamps_to_12bit():
    assert mc.frac_to_amp(2.0) == 0xFFF
    assert mc.frac_to_amp(-1.0) == 0


def test_chn_byte_pack_unpack():
    for typ in (0, 1, 2):
        for chn in (0, 9, 21):
            assert mc.unpack_chn_byte(mc.chn_byte(typ, chn)) == (typ, chn)


# -- client <-> mock round trips --------------------------------------------
@pytest.fixture
def server():
    # ephemeral-ish fixed port for the mock; REP/REQ on loopback
    srv = MockMolecubeServer("tcp://127.0.0.1:7798").start()
    yield srv
    srv.stop()


@pytest.fixture
def client(server):
    c = mc.MolecubeClient("tcp://127.0.0.1:7798", timeout_ms=1500)
    yield c
    c.close()


def test_state_id_and_ping(client):
    sid, server_id = client.state_id()
    assert server_id > 0
    assert client.ping() == server_id


def test_clock_read_write(client):
    assert client.get_clock() == 100
    client.set_clock(9)
    assert client.get_clock() == 9


def test_max_ttl(client):
    assert client.get_max_ttl() == 31


def test_dds_all_and_names(client):
    raw = client.get_dds_all()
    assert 0 in raw and 7 in raw
    assert set(raw[0]) == {"freq", "amp", "phase"}
    names = client.get_dds_names()
    assert names[7] == "EOM616"


# -- channel names: writing back to the daemon's store ----------------------
def test_encode_names_blob_roundtrip():
    # _encode_names_blob is the exact inverse of _parse_names_blob.
    names = {0: "MOTX", 7: "EOM616", 21: "2DMOT"}
    blob = mc.MolecubeClient._encode_names_blob(names)
    assert mc.MolecubeClient._parse_names_blob(blob) == names


def test_set_dds_name_roundtrip(client):
    client.set_dds_names({2: "SLM_renamed"})
    assert client.get_dds_names()[2] == "SLM_renamed"
    # and it surfaces on the snapshot row the dashboard renders
    row = next(r for r in client.snapshot()["dds"] if r["chn"] == 2)
    assert row["name"] == "SLM_renamed"


def test_set_ttl_name_roundtrip(client):
    client.set_ttl_names({10: "MyTTL"})
    assert client.get_ttl_names()[10] == "MyTTL"
    row = next(r for r in client.snapshot(ttl_max_chn=31)["ttl"] if r["chn"] == 10)
    assert row["name"] == "MyTTL"


def test_set_names_merges_not_replaces(client):
    # Renaming ONE channel must leave the others' names untouched (the daemon
    # merges per channel; we never send the whole map).
    before = client.get_dds_names()
    assert before.get(7) == "EOM616"
    client.set_dds_names({0: "freshname"})
    after = client.get_dds_names()
    assert after[0] == "freshname"
    assert after[7] == "EOM616"          # untouched


def test_set_name_empty_clears(client):
    client.set_dds_names({7: ""})        # empty -> reads back as "no name"
    assert 7 not in client.get_dds_names()


def test_set_names_bumps_name_id(client):
    nid0 = client.name_id()[0]
    client.set_dds_names({1: "bumped"})
    assert client.name_id()[0] == nid0 + 1
    # a no-op set (only an out-of-range channel) must NOT bump name_id
    client.set_dds_names({250: "ignored"})
    assert client.name_id()[0] == nid0 + 1


def test_set_names_empty_dict_is_noop(client):
    nid0 = client.name_id()[0]
    assert client.set_dds_names({}) is True
    assert client.name_id()[0] == nid0   # nothing sent, nothing changed


def test_dds_set_then_read_back(client):
    # set DDS chn 2 (SLM) freq to 123.456 MHz, read it back through snapshot
    word = mc.hz_to_ftw(123.456e6)
    client.set_dds(mc.TYP_FREQ, 2, word)
    res = client.get_dds([(mc.TYP_FREQ, 2)])
    assert res[0][2] == word
    assert mc.ftw_to_hz(res[0][2]) == pytest.approx(123.456e6, rel=1e-5)


def test_dds_override_set_and_clear(client):
    client.override_dds(mc.TYP_AMP, 8, mc.frac_to_amp(0.25))
    ovr = dict((chn, val) for typ, chn, val in client.get_override_dds() if typ == mc.TYP_AMP)
    assert 8 in ovr
    client.override_dds(mc.TYP_AMP, 8, None)  # clear
    ovr2 = [(t, c) for t, c, v in client.get_override_dds()]
    assert (mc.TYP_AMP, 8) not in ovr2


def test_ttl_read_is_nonmutating(client):
    before = client.get_ttl(0)
    again = client.get_ttl(0)
    assert before == again  # zero-mask read must not change state


def test_ttl_set_channel(client):
    client.set_ttl_chn(31, True)
    assert client.get_ttl(0) & (1 << 31)
    client.set_ttl_chn(31, False)
    assert not (client.get_ttl(0) & (1 << 31))


def test_ttl_override(client):
    client.override_ttl(0, 0, 1 << 5, 0)   # force chn 5 high
    lo, hi = client.get_override_ttl(0)
    assert hi & (1 << 5)


def test_snapshot_shape(client):
    snap = client.snapshot()
    assert snap["connected"] is True
    assert "dds" in snap and "ttl" in snap
    assert snap["clock"] == 100
    # one DDS row carries engineering units
    row = next(r for r in snap["dds"] if r["chn"] == 0)
    assert row["freq_hz"] is not None
    assert 0.0 <= row["amp"] <= 1.0
    assert len(snap["ttl"]) == 32


def test_ttl_shows_full_bank_when_max_ttl_unsupported():
    # A daemon that doesn't support get_max_ttl replies with the 1-byte error
    # status (value 1). The client must NOT read that as max_ttl=1 and show only
    # channels 0-1; it should still expose the full 32-bit bank.
    srv = MockMolecubeServer("tcp://127.0.0.1:7792")
    _orig = srv._dispatch
    srv._dispatch = lambda f: (bytes([1]) if f[0] == b"get_max_ttl" else _orig(f))
    srv.start()
    try:
        c = mc.MolecubeClient("tcp://127.0.0.1:7792", timeout_ms=1500)
        snap = c.snapshot()
        assert len(snap["ttl"]) == 32, "TTL must show the full bank, not just ch 0-1"
        assert [r["chn"] for r in snap["ttl"]] == list(range(32))
        c.close()
    finally:
        srv.stop()


def test_ttl_multibank_count(client):
    # With an explicit max channel of 55 (engine config.yml), the panel must span
    # 56 channels across two banks -- this is what labctrl-node can't do yet.
    snap = client.snapshot(ttl_max_chn=55)
    assert len(snap["ttl"]) == 56
    assert [r["chn"] for r in snap["ttl"]] == list(range(56))
    assert snap["ttl_n_banks"] == 2


def test_ttl_value_reflects_override(client):
    # Force ch5 high and ch6 low; the displayed value must follow the override.
    client.override_ttl(0, 0, 1 << 5, 0)         # ch5 forced high
    client.override_ttl(0, 1 << 6, 0, 0)         # ch6 forced low
    snap = client.snapshot(ttl_max_chn=55)
    row5 = next(r for r in snap["ttl"] if r["chn"] == 5)
    row6 = next(r for r in snap["ttl"] if r["chn"] == 6)
    assert row5["ovr_hi"] and row5["value"] is True
    assert row6["ovr_lo"] and row6["value"] is False


def test_set_values_force_high_then_clear_restores(client):
    # labctrl-node parity: clearing an override re-asserts the value first, so the
    # output doesn't glitch back to whatever the value register held.
    client.set_values({'ttl': {'val5': False}}, {})          # ch5 low
    client.set_values({'ttl': {'ovr5': True, 'val5': True}}, {})  # force high
    snap = client.snapshot(ttl_max_chn=55)
    r5 = next(r for r in snap['ttl'] if r['chn'] == 5)
    assert r5['ovr_hi'] and r5['value'] is True
    cur = mc.MolecubeClient.flatten_current(snap)
    client.set_values({'ttl': {'ovr5': False}}, cur)         # clear override
    r5b = next(r for r in client.snapshot(ttl_max_chn=55)['ttl'] if r['chn'] == 5)
    assert not r5b['ovr_hi'] and not r5b['ovr_lo']
    assert r5b['value'] is True                              # restored, no glitch to low


def test_set_values_dds_value_routes_to_override(client):
    w1, w2 = mc.frac_to_amp(0.25), mc.frac_to_amp(0.5)
    client.set_values({'dds': {'ovr_amp8': True, 'amp8': w1}}, {})
    cur = mc.MolecubeClient.flatten_current(client.snapshot())
    client.set_values({'dds': {'amp8': w2}}, cur)            # write while overridden
    ovr = {(t, c): v for t, c, v in client.get_override_dds()}
    assert ovr[(mc.TYP_AMP, 8)] == w2                        # updated the OVERRIDE, not set


def test_set_values_dds_clear_override(client):
    client.set_values({'dds': {'ovr_freq2': True, 'freq2': mc.hz_to_ftw(80e6)}}, {})
    cur = mc.MolecubeClient.flatten_current(client.snapshot())
    client.set_values({'dds': {'ovr_freq2': False}}, cur)
    assert (mc.TYP_FREQ, 2) not in [(t, c) for t, c, v in client.get_override_dds()]


def test_set_values_multibank_ttl_write(client):
    client.set_values({'ttl': {'val40': True}}, {})          # ch40 -> bank 1, bit 8
    assert client.get_ttl(1) & (1 << 8)
    client.set_values({'ttl': {'val40': False}}, {})
    assert not (client.get_ttl(1) & (1 << 8))


def test_timeout_on_dead_endpoint():
    c = mc.MolecubeClient("tcp://127.0.0.1:7700", timeout_ms=300)  # nothing listening
    with pytest.raises(mc.MolecubeTimeout):
        c.state_id()
    c.close()
