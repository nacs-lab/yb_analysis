"""mock_molecube_server.py -- a local, in-process fake of the molecube2 daemon.

Implements the same binary ZMQ protocol as molecube2 ``lib/server.cpp`` so the dashboard
and :mod:`molecube_client` can be exercised end-to-end WITHOUT touching the real, sensitive
FPGA daemon. It holds fake DDS/TTL/clock state and mutates it on writes, so the UI round-trips
realistically.

The real daemon binds a ROUTER; a REQ client works against either ROUTER or REP, so the mock
uses a REP socket for simplicity (the client behaves identically).

Usage (tests / manual):
    srv = MockMolecubeServer("tcp://127.0.0.1:7799")
    srv.start()
    ...                         # point YB_MOLECUBE_URL at the same address
    srv.stop()

Or standalone:  python -m yb_analysis.control.mock_molecube_server tcp://127.0.0.1:7799
"""

import struct
import threading
import time

import zmq

from .molecube_client import (
    chn_byte, unpack_chn_byte, hz_to_ftw, frac_to_amp, deg_to_phase,
    TYP_FREQ, TYP_AMP, TYP_PHASE, NUM_DDS,
)

_U32 = 0xFFFFFFFF


def _default_dds():
    """Seed a few active DDS channels with realistic words (mirrors expConfig labels)."""
    # (chn, name, freq_hz, amp_frac, phase_deg)
    seed = [
        (0, "556RydbergMOTh", 80e6, 0.50, 0.0),
        (1, "556MOTX", 107.7e6, 0.60, 0.0),
        (2, "SLM", 100e6, 0.80, 0.0),
        (7, "EOM616", 252.07e6, 0.30, 0.0),
        (8, "AOM616", 120e6, 0.55, 0.0),
        (12, "369", 310e6, 0.40, 0.0),
        (19, "BlueMOT", 200e6, 0.70, 0.0),
        (21, "2DMOT", 90e6, 1.00, 0.0),
    ]
    dds = {}
    names = {}
    for chn, name, fhz, amp, deg in seed:
        dds[chn] = {
            TYP_FREQ: hz_to_ftw(fhz),
            TYP_AMP: frac_to_amp(amp),
            TYP_PHASE: deg_to_phase(deg),
        }
        names[chn] = name
    return dds, names


class MockMolecubeServer:
    def __init__(self, url="tcp://127.0.0.1:7799", max_ttl=31):
        self.url = url
        self.max_ttl = max_ttl
        self._ctx = None
        self._sock = None
        self._thread = None
        self._running = False
        # fake device state
        self.dds, self.dds_names = _default_dds()
        self.dds_ovr = {}                     # {chn: {typ: word}}
        # Per-bank 32-bit TTL state (bank 0 = ch 0-31, bank 1 = ch 32-63, ...).
        self.ttl_value = {}                   # {bank: 32-bit output mask}
        self.ttl_ovr_lo = {}                  # {bank: forced-low mask}
        self.ttl_ovr_hi = {}                  # {bank: forced-high mask}
        self.clock = 100
        self.ttl_names = {31: "BlueMOTShutter", 15: "SampleAndHold", 40: "Bank1Chn40"}
        self.startup = "# mock startup cmdlist\n"
        self._server_id = int(time.time() * 1000) & _U32
        self._state_id = 1
        self._name_id = 1

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.bind(self.url)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="mock-molecube", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- main loop ----------------------------------------------------------
    def _loop(self):
        while self._running:
            try:
                frames = self._sock.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            try:
                reply = self._dispatch(frames)
            except Exception:
                reply = bytes([1])      # generic error status
            try:
                self._sock.send(reply)
            except zmq.ZMQError:
                break

    # -- helpers ------------------------------------------------------------
    def _active(self):
        return sorted(self.dds)

    def _bump_state(self):
        self._state_id += 1

    def _dds_blob(self, items):
        out = bytearray()
        for typ, chn, val in items:
            out.append(chn_byte(typ, chn))
            out += struct.pack("<I", val & _U32)
        return bytes(out)

    def _names_blob(self, names):
        out = bytearray()
        for chn in sorted(names):
            name = names[chn]
            if not name:
                continue
            out.append(chn & 0xFF)
            out += name.encode("utf-8") + b"\x00"
        return bytes(out)

    def _apply_set_names(self, data, names, nchn):
        """Merge packed [chn u8][NUL-terminated name] pairs into ``names`` -- mirrors
        molecube2 process_set_names: per-channel update (not a full replace),
        out-of-range channels skipped, ``name_id`` bumped only if something was set."""
        i, n, has_set = 0, len(data), False
        while i + 1 < n:
            chn = data[i]
            i += 1
            end = data.find(b"\x00", i)
            if end < 0:
                break
            name = data[i:end].decode("utf-8", "replace")
            i = end + 1
            if chn < nchn:
                names[chn] = name
                has_set = True
        if has_set:
            self._name_id += 1
        return has_set

    # -- command dispatch (mirrors server.cpp) ------------------------------
    def _dispatch(self, frames):
        cmd = frames[0]
        data = frames[1] if len(frames) > 1 else b""

        if cmd == b"state_id":
            return struct.pack("<QQ", self._state_id, self._server_id)
        if cmd == b"name_id":
            return struct.pack("<QQ", self._name_id, self._server_id)
        if cmd == b"get_clock":
            return bytes([self.clock & 0xFF])
        if cmd == b"set_clock":
            self.clock = data[0] if data else 0
            self._bump_state()
            return bytes([0])
        if cmd == b"get_max_ttl":
            return bytes([self.max_ttl & 0xFF])

        if cmd == b"get_dds":
            if data:
                items = []
                for b in data:
                    typ, chn = unpack_chn_byte(b)
                    items.append((typ, chn, self.dds.get(chn, {}).get(typ, 0)))
                return self._dds_blob(items)
            items = []
            for chn in self._active():
                for typ in (TYP_FREQ, TYP_AMP, TYP_PHASE):
                    items.append((typ, chn, self.dds[chn].get(typ, 0)))
            return self._dds_blob(items)

        if cmd == b"set_dds":
            for i in range(0, len(data), 5):
                typ, chn = unpack_chn_byte(data[i])
                (val,) = struct.unpack("<I", data[i + 1:i + 5])
                self.dds.setdefault(chn, {})[typ] = val & _U32
            self._bump_state()
            return bytes([0])

        if cmd == b"get_override_dds":
            items = []
            for chn in sorted(self.dds_ovr):
                for typ, val in self.dds_ovr[chn].items():
                    if val != _U32:
                        items.append((typ, chn, val))
            return self._dds_blob(items)

        if cmd == b"override_dds":
            for i in range(0, len(data), 5):
                typ, chn = unpack_chn_byte(data[i])
                (val,) = struct.unpack("<I", data[i + 1:i + 5])
                if val == _U32:
                    self.dds_ovr.get(chn, {}).pop(typ, None)
                else:
                    self.dds_ovr.setdefault(chn, {})[typ] = val
            self._bump_state()
            return bytes([0])

        if cmd == b"reset_dds":
            chn = data[0] if data else 0
            self.dds_ovr.pop(chn, None)
            self._bump_state()
            return bytes([0])

        if cmd == b"set_ttl":
            # data = lo, hi [, bank]; all-zero masks = a no-op read of that bank
            if len(data) >= 12:
                lo, hi, bank = struct.unpack("<III", data[:12])
            else:
                lo, hi = struct.unpack("<II", data[:8])
                bank = 0
            cur = self.ttl_value.get(bank, 0)
            self.ttl_value[bank] = (cur & ~lo) | hi
            if lo or hi:
                self._bump_state()
            return struct.pack("<I", self.ttl_value[bank])

        if cmd == b"override_ttl":
            # data = lo, hi, normal [, bank]; all-zero = a no-op read of that bank
            if len(data) >= 16:
                lo, hi, normal, bank = struct.unpack("<IIII", data[:16])
            else:
                lo, hi, normal = struct.unpack("<III", data[:12])
                bank = 0
            clo = self.ttl_ovr_lo.get(bank, 0)
            chi = self.ttl_ovr_hi.get(bank, 0)
            self.ttl_ovr_lo[bank] = (clo & ~normal & ~hi) | lo
            self.ttl_ovr_hi[bank] = (chi & ~normal & ~lo) | hi
            if lo or hi or normal:
                self._bump_state()
            return struct.pack("<II", self.ttl_ovr_lo[bank], self.ttl_ovr_hi[bank])

        if cmd == b"get_dds_names":
            return self._names_blob(self.dds_names)
        if cmd == b"get_ttl_names":
            return self._names_blob(self.ttl_names)
        if cmd == b"set_dds_names":
            self._apply_set_names(data, self.dds_names, NUM_DDS)
            return bytes([0])
        if cmd == b"set_ttl_names":
            self._apply_set_names(data, self.ttl_names, self.max_ttl + 1)
            return bytes([0])
        if cmd == b"get_startup":
            return self.startup.encode("utf-8") + b"\x00"

        return bytes([1])               # unknown command -> error status


def main(argv=None):
    import sys
    url = (argv or sys.argv[1:] or ["tcp://127.0.0.1:7799"])[0]
    srv = MockMolecubeServer(url).start()
    print("mock molecube daemon listening on", url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
