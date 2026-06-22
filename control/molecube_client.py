"""molecube_client.py -- pure-pyzmq client for the molecube2 Zynq control daemon.

The lab's FPGA1 (DDS / TTL / clock) is fronted by the **molecube2** C++ ZMQ daemon
(https://github.com/nacs-lab/molecube2, master). The libnacs C++ engine talks to it to
run sequences; the Next.js web page at ``https://yb.nigrp.org/s/zynq/1/dds`` talks to it
for live device control. This module is a *Python* re-implementation of that same binary
ZMQ protocol so the exp-control dashboard can read AND control the FPGA **through the
molecube daemon** (never the devices directly).

  ┌ exp-control dashboard (Flask) ┐   ZMQ REQ    ┌ molecube2 daemon ┐   ┌ FPGA1 ┐
  │  /api/molecube/* routes       │ ───────────▶ │  ROUTER :7777    │ ─▶│ DDS/TTL│
  └───────────────────────────────┘              └──────────────────┘   └────────┘

SAFETY
------
* This module **does not connect on import** -- a socket is created lazily on the first
  request. Construct a client freely; nothing hits the network until you call a method.
* The dashboard wraps every molecube endpoint in a master gate that is CLOSED by default,
  so in normal operation this client is never exercised against the live daemon.
* For development/testing, point ``url`` at the local mock (``mock_molecube_server.py``)
  -- the wire protocol is identical, so the mock fully exercises this client.

PROTOCOL (from molecube2 ``lib/server.cpp`` / ``test/clients/test_client.py``)
------------------------------------------------------------------------------
A plain ZMQ ``REQ`` socket speaks to the daemon's ``ROUTER``. A request is one or two
frames: ``[command]`` or ``[command][binary-args]``. The reply is a single frame.

  DDS channel byte = ``(type << 6) | chn``  (type 0=freq, 1=amp, 2=phase; chn 0..21)
  Reads:  state_id, name_id, get_clock, get_max_ttl, get_dds[chns], get_override_dds,
          get_dds_names, get_ttl_names, get_startup
          (TTL value/override are read by issuing set_ttl/override_ttl with ZERO masks --
           the daemon treats an all-zero mask as a no-op "get", per server.cpp.)
  Writes: set_dds, override_dds, reset_dds, set_ttl, override_ttl, set_clock,
          set_dds_names, set_ttl_names

CHANNEL NAMES
-------------
Names are NOT a local concept -- the daemon owns them. ``get_dds_names``/``get_ttl_names``
read the daemon's ``dds.yaml``/``ttl.yaml``; ``set_dds_names``/``set_ttl_names`` write back
to those same files (the daemon MERGES per channel, persists to yaml, and bumps ``name_id``).
This is the exact path the labctrl-node web page uses, so a rename here shows up everywhere
that reads the daemon. We deliberately keep NO independent name map of our own.

REGISTER <-> ENGINEERING UNITS  (verified in-lab 2026-06-02 by set->read on DDS9/TTL31)
--------------------------------------------------------------------------------------
  freq : Hz   = ftw * 3.5e9 / 2**32     (sysclk 3.5 GHz; ftw=1227134 <-> 1.000 MHz)
  amp  : frac = word / 4095             (12-bit DAC; full-scale word 4095 = 1.0)
  phase: deg  = word / 2**16 * 360      (16-bit; == word * 90 / 2**14)

The freq/amp/phase scales match the labctrl-node web UI field components exactly
(DDSFreqField scale 3.5e9/2**32, DDSAmpField scale 1/4095, DDSPhaseField scale 90/2**14).
"""

import re
import struct
import threading

import zmq

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
TYP_FREQ = 0
TYP_AMP = 1
TYP_PHASE = 2
_TYP_NAME = {TYP_FREQ: "freq", TYP_AMP: "amp", TYP_PHASE: "phase"}
_NAME_TYP = {v: k for k, v in _TYP_NAME.items()}

NUM_DDS = 22                 # molecube2 fixes 22 DDS channels (server.cpp)
DDS_SYSCLK_HZ = 3.5e9        # AD9914 system clock
_FTW_SCALE = 2 ** 32
_AMP_FULLSCALE = 4095        # 12-bit DAC full scale (word 4095 = 1.0; matches labctrl-node)
_PHASE_SCALE = 2 ** 16       # 16-bit phase word
_U32_MASK = 0xFFFFFFFF
_DDS_NO_OVR = _U32_MASK      # value 0xffffffff (-1) means "no override / clear override"


# ---------------------------------------------------------------------------
# Unit conversions (module-level, pure functions -- safe to use anywhere)
# ---------------------------------------------------------------------------
def ftw_to_hz(ftw):
    """Frequency tuning word (u32) -> Hz."""
    return (ftw & _U32_MASK) * DDS_SYSCLK_HZ / _FTW_SCALE


def hz_to_ftw(hz):
    """Hz -> frequency tuning word (u32), rounded + clamped to 32 bits."""
    ftw = int(round(float(hz) * _FTW_SCALE / DDS_SYSCLK_HZ))
    return max(0, min(ftw, _U32_MASK))


def amp_to_frac(word):
    """Amplitude DAC word (12-bit) -> normalized fraction 0..1 (word 4095 = 1.0)."""
    return (word & 0xFFF) / float(_AMP_FULLSCALE)


def frac_to_amp(frac):
    """Normalized fraction 0..1 -> amplitude DAC word (12-bit, 4095 = full scale)."""
    word = int(round(max(0.0, min(1.0, float(frac))) * _AMP_FULLSCALE))
    return min(word, 0xFFF)


def phase_to_deg(word):
    """Phase word (16-bit) -> degrees 0..360."""
    return (word & 0xFFFF) / float(_PHASE_SCALE) * 360.0


def deg_to_phase(deg):
    """Degrees -> phase word (16-bit, wrapped)."""
    return int(round((float(deg) % 360.0) / 360.0 * _PHASE_SCALE)) & 0xFFFF


def chn_byte(typ, chn):
    """Pack (type, channel) -> the single DDS channel byte used on the wire."""
    return ((typ & 0x3) << 6) | (chn & 0x3F)


def unpack_chn_byte(b):
    """Unpack a DDS channel byte -> (type, channel)."""
    return (b >> 6) & 0x3, b & 0x3F


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class MolecubeError(Exception):
    """Base class for molecube client errors."""


class MolecubeTimeout(MolecubeError):
    """No reply from the daemon within the timeout (also fired on connect failure)."""


class MolecubeProtocolError(MolecubeError):
    """The daemon replied with an unexpected size or an error status."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class MolecubeClient:
    """Thread-safe ZMQ REQ client for the molecube2 daemon.

    Lazily connects on the first request. A REQ socket has a strict send/recv FSM, so a
    timed-out request leaves it unusable -- we close + recreate the socket on any timeout
    (and serialize all requests with a lock so the FSM is never interleaved).
    """

    def __init__(self, url, timeout_ms=2000):
        self.url = url
        self.timeout_ms = int(timeout_ms)
        self._ctx = None
        self._sock = None
        self._lock = threading.Lock()

    # -- socket lifecycle ---------------------------------------------------
    def _ensure_socket(self):
        if self._ctx is None:
            self._ctx = zmq.Context.instance()
        if self._sock is None:
            sock = self._ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.connect(self.url)
            self._sock = sock
        return self._sock

    def _reset_socket(self):
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
            self._sock = None

    def close(self):
        with self._lock:
            self._reset_socket()

    # -- low-level request --------------------------------------------------
    def _request(self, frames):
        """Send a multipart REQ (list of bytes), return the single reply frame.

        On any ZMQ error / timeout the socket is recreated and ``MolecubeTimeout`` is
        raised, so the next call starts from a clean FSM.
        """
        with self._lock:
            sock = self._ensure_socket()
            try:
                sock.send_multipart([bytes(f) for f in frames])
                return sock.recv()
            except zmq.Again:
                self._reset_socket()
                raise MolecubeTimeout(
                    "no reply from molecube daemon at %s within %d ms"
                    % (self.url, self.timeout_ms))
            except zmq.ZMQError as e:
                self._reset_socket()
                raise MolecubeError("ZMQ error talking to %s: %s" % (self.url, e))

    @staticmethod
    def _check_status(reply, what):
        """Many writes reply with a single status byte (0=ok, else error)."""
        if len(reply) != 1:
            raise MolecubeProtocolError(
                "%s: expected 1-byte status, got %d bytes" % (what, len(reply)))
        if reply[0] != 0:
            raise MolecubeProtocolError("%s: daemon returned error status %d"
                                        % (what, reply[0]))
        return True

    # -- liveness / ids -----------------------------------------------------
    def state_id(self):
        """Return (state_id, server_id). state_id changes whenever device state changes."""
        r = self._request([b"state_id"])
        if len(r) != 16:
            raise MolecubeProtocolError("state_id: expected 16 bytes, got %d" % len(r))
        return struct.unpack("<QQ", r)

    def name_id(self):
        """Return (name_id, server_id). name_id changes when channel names change."""
        r = self._request([b"name_id"])
        if len(r) != 16:
            raise MolecubeProtocolError("name_id: expected 16 bytes, got %d" % len(r))
        return struct.unpack("<QQ", r)

    def ping(self):
        """Cheap liveness check -- returns the server_id, or raises on failure."""
        return self.state_id()[1]

    # -- clock --------------------------------------------------------------
    def get_clock(self):
        """Return the clock divider byte (0..255). 255 == clock output disabled."""
        r = self._request([b"get_clock"])
        if len(r) != 1:
            raise MolecubeProtocolError("get_clock: expected 1 byte, got %d" % len(r))
        return r[0]

    def set_clock(self, div):
        """Set the clock divider byte (0..255). WRITE."""
        self._check_status(self._request([b"set_clock", struct.pack("B", int(div) & 0xFF)]),
                           "set_clock")
        return True

    # -- DDS reads ----------------------------------------------------------
    def get_max_ttl(self):
        """Return the max TTL channel index (e.g. 31 for a single 32-bit bank)."""
        r = self._request([b"get_max_ttl"])
        if len(r) != 1:
            raise MolecubeProtocolError("get_max_ttl: expected 1 byte, got %d" % len(r))
        return r[0]

    @staticmethod
    def _parse_dds_blob(blob):
        """Parse an N*5 DDS reply -> list of (typ, chn, value_u32)."""
        if len(blob) % 5 != 0:
            raise MolecubeProtocolError("DDS reply not a multiple of 5 bytes (%d)"
                                        % len(blob))
        out = []
        for i in range(0, len(blob), 5):
            typ, chn = unpack_chn_byte(blob[i])
            (val,) = struct.unpack("<I", blob[i + 1:i + 5])
            out.append((typ, chn, val))
        return out

    def get_dds_all(self):
        """Read every active DDS channel (all 3 types).

        Returns ``{chn: {'freq': w, 'amp': w, 'phase': w}}`` of RAW register words for
        channels the daemon reports as active.
        """
        blob = self._request([b"get_dds"])
        out = {}
        for typ, chn, val in self._parse_dds_blob(blob):
            out.setdefault(chn, {})[_TYP_NAME[typ]] = val
        return out

    def get_dds(self, requests):
        """Read specific DDS channels. ``requests`` = list of (typ, chn).

        Returns list of (typ, chn, value_u32) in reply order.
        """
        data = bytes(chn_byte(t, c) for t, c in requests)
        blob = self._request([b"get_dds", data])
        return self._parse_dds_blob(blob)

    def get_override_dds(self):
        """Read active DDS overrides -> list of (typ, chn, value_u32)."""
        return self._parse_dds_blob(self._request([b"get_override_dds"]))

    # -- DDS writes ---------------------------------------------------------
    def set_dds(self, typ, chn, word):
        """Set a single DDS register (raw word). WRITE."""
        return self.set_dds_many([(typ, chn, word)])

    def set_dds_many(self, items):
        """Set multiple DDS registers at once. ``items`` = list of (typ, chn, word). WRITE."""
        data = b"".join(bytes([chn_byte(t, c)]) + struct.pack("<I", w & _U32_MASK)
                        for t, c, w in items)
        return self._check_status(self._request([b"set_dds", data]), "set_dds")

    def override_dds(self, typ, chn, word):
        """Set a DDS override (raw word), or pass ``None`` to clear the override. WRITE."""
        w = _DDS_NO_OVR if word is None else (word & _U32_MASK)
        data = bytes([chn_byte(typ, chn)]) + struct.pack("<I", w)
        return self._check_status(self._request([b"override_dds", data]), "override_dds")

    def reset_dds(self, chn):
        """Reset/reinitialize a DDS channel. WRITE."""
        return self._check_status(self._request([b"reset_dds", bytes([int(chn) & 0xFF])]),
                                  "reset_dds")

    # -- TTL ----------------------------------------------------------------
    def get_ttl(self, bank=0):
        """Read the current TTL output mask for a bank (u32). **READ-ONLY.**

        There is no dedicated "get TTL" command, so we send a ``set_ttl`` frame whose
        masks are HARD-CODED to zero (and asserted zero before send). The daemon executes
        an all-zero set as a no-op "get" (server.cpp). Built directly here -- NOT via the
        ``set_ttl`` write path -- so this method can never carry a nonzero mask and thus
        can never change an output.
        """
        data = struct.pack("<III", 0, 0, bank) if bank else struct.pack("<II", 0, 0)
        assert all(b == 0 for b in data[:8]), "get_ttl must send zero masks"
        r = self._request([b"set_ttl", data])
        if len(r) != 4:
            raise MolecubeProtocolError("get_ttl: expected 4 bytes, got %d" % len(r))
        return struct.unpack("<I", r)[0]

    def set_ttl(self, bank, lo_mask, hi_mask):
        """Turn channels off (``lo_mask`` bits) and on (``hi_mask`` bits). Returns the new
        mask. With both masks 0 this is a pure read. WRITE (unless masks are 0)."""
        if bank:
            data = struct.pack("<III", lo_mask & _U32_MASK, hi_mask & _U32_MASK, bank)
        else:
            data = struct.pack("<II", lo_mask & _U32_MASK, hi_mask & _U32_MASK)
        r = self._request([b"set_ttl", data])
        if len(r) != 4:
            raise MolecubeProtocolError("set_ttl: expected 4 bytes, got %d" % len(r))
        return struct.unpack("<I", r)[0]

    def set_ttl_chn(self, chn, on, bank=None):
        """Convenience: drive a single TTL channel on/off. WRITE.

        The bank is derived from the channel (``chn // 32``) unless given explicitly,
        so channels >= 32 (e.g. 32..55 on a 56-channel rig) hit the correct bank."""
        chn = int(chn)
        if bank is None:
            bank = chn // 32
        bit = 1 << (chn % 32)
        if on:
            return self.set_ttl(bank, 0, bit)
        return self.set_ttl(bank, bit, 0)

    def get_override_ttl(self, bank=0):
        """Read TTL override masks for a bank -> (lo_mask, hi_mask). **READ-ONLY.**

        Sends an ``override_ttl`` frame whose masks are HARD-CODED to zero (asserted),
        which the daemon runs as a no-op "get". Built directly here -- NOT via the
        ``override_ttl`` write path -- so it can never carry a nonzero mask."""
        body = struct.pack("<III", 0, 0, 0)
        if bank:
            body += struct.pack("<I", bank)
        assert all(b == 0 for b in body[:12]), "get_override_ttl must send zero masks"
        r = self._request([b"override_ttl", body])
        if len(r) != 8:
            raise MolecubeProtocolError("get_override_ttl: expected 8 bytes, got %d" % len(r))
        return struct.unpack("<II", r)

    def override_ttl(self, bank, lo_mask, hi_mask, normal_mask):
        """Force channels low (``lo_mask``), high (``hi_mask``), or back to normal
        (``normal_mask``). Returns the new (lo, hi) override masks. All-zero is a read.
        WRITE (unless all masks 0)."""
        body = struct.pack("<III", lo_mask & _U32_MASK, hi_mask & _U32_MASK,
                           normal_mask & _U32_MASK)
        if bank:
            body += struct.pack("<I", bank)
        r = self._request([b"override_ttl", body])
        if len(r) != 8:
            raise MolecubeProtocolError("override_ttl: expected 8 bytes, got %d" % len(r))
        return struct.unpack("<II", r)

    # -- names --------------------------------------------------------------
    @staticmethod
    def _parse_names_blob(blob):
        """Parse a names reply: repeated ``[chn u8][NUL-terminated name]`` -> {chn: name}."""
        out = {}
        i = 0
        n = len(blob)
        while i + 1 < n:
            chn = blob[i]
            i += 1
            end = blob.find(b"\x00", i)
            if end < 0:
                break
            out[chn] = blob[i:end].decode("utf-8", "replace")
            i = end + 1
        return out

    @staticmethod
    def _encode_names_blob(names):
        """Encode ``{chn: name}`` -> packed ``[chn u8][NUL-terminated utf-8 name]`` pairs,
        the wire format ``set_dds_names``/``set_ttl_names`` expect (inverse of
        ``_parse_names_blob``). ``chn`` is masked to a byte; ``None`` -> empty name."""
        out = bytearray()
        for chn, name in names.items():
            out.append(int(chn) & 0xFF)
            out += ("" if name is None else str(name)).encode("utf-8") + b"\x00"
        return bytes(out)

    def get_dds_names(self):
        """Return ``{chn: name}`` for named DDS channels (daemon's dds.yaml)."""
        return self._parse_names_blob(self._request([b"get_dds_names"]))

    def get_ttl_names(self):
        """Return ``{chn: name}`` for named TTL channels (daemon's ttl.yaml)."""
        return self._parse_names_blob(self._request([b"get_ttl_names"]))

    def set_dds_names(self, names):
        """Set DDS channel names. ``names`` = ``{chn: name}``. WRITE.

        The daemon MERGES per channel (it does NOT replace the whole map), persists to its
        ``dds.yaml``, and bumps ``name_id`` when anything changed. An empty string reads
        back as "no name" (the daemon stores it but ``get_dds_names`` skips empties). A
        channel index outside the daemon's range is logged + skipped by the daemon. This is
        the same store/path the labctrl-node web UI uses -- we keep no name map of our own."""
        if not names:
            return True
        return self._check_status(
            self._request([b"set_dds_names", self._encode_names_blob(names)]),
            "set_dds_names")

    def set_ttl_names(self, names):
        """Set TTL channel names. ``names`` = ``{chn: name}``. WRITE.

        Same semantics as :meth:`set_dds_names`, but persisted to the daemon's ``ttl.yaml``."""
        if not names:
            return True
        return self._check_status(
            self._request([b"set_ttl_names", self._encode_names_blob(names)]),
            "set_ttl_names")

    # -- startup sequence ---------------------------------------------------
    def get_startup(self):
        """Return the startup command-list text (NUL-terminated on the wire)."""
        r = self._request([b"get_startup"])
        return r.split(b"\x00", 1)[0].decode("utf-8", "replace")

    # -- batched writes (labctrl-node parity) -------------------------------
    def _set_dds_raw(self, cmd, items):
        """Send set_dds/override_dds for a list of (chn_byte, word) pairs (labctrl-node
        builds the buffer the same way: [chn_byte][i32] repeated)."""
        data = b"".join(bytes([cmdi & 0xFF]) + struct.pack("<I", w & _U32_MASK)
                        for cmdi, w in items)
        return self._check_status(self._request([cmd.encode(), data]), cmd)

    def _apply_ttl(self, ttl_set, ttl_cur, sent):
        """Port of labctrl-node #set_ttl_vals (extended to multiple 32-bit banks).

        ``ttl_set`` = {'val<i>': bool, 'ovr<i>': bool} of REQUESTED changes;
        ``ttl_cur`` = {'val<i>': bool, 'ovr<i>': bool} CURRENT state (for the coupling).
        Forms per-bank set_ttl(lo,hi) + override_ttl(lo,hi,normal); set_ttl is sent FIRST
        (before disabling override) to minimize output flip-flop, exactly as labctrl-node."""
        values, ovrs = {}, {}
        for key, v in ttl_set.items():
            m = re.match(r'^val(\d+)$', key)
            if m:
                values[int(m.group(1))] = bool(v)
                continue
            m = re.match(r'^ovr(\d+)$', key)
            if m:
                ovrs[int(m.group(1))] = bool(v)
        banks = {}

        def bank_acc(b):
            return banks.setdefault(b, {'ovr_hi': 0, 'ovr_lo': 0, 'ovr_normal': 0,
                                        'hi': 0, 'lo': 0})
        for i in set(values) | set(ovrs):
            bank, mask = i // 32, 1 << (i % 32)
            acc = bank_acc(bank)
            ovr_set = ovrs.get(i)
            ovr = ovr_set if ovr_set is not None else bool(ttl_cur.get('ovr%d' % i, False))
            val_set = values.get(i)
            val = val_set if val_set is not None else bool(ttl_cur.get('val%d' % i, False))
            if ovr_set is not None or ovr:
                val_set = None
                if not ovr:
                    acc['ovr_normal'] |= mask
                    val_set = val               # restore value as override releases
                elif val:
                    acc['ovr_hi'] |= mask
                else:
                    acc['ovr_lo'] |= mask
            if val_set is not None:
                if val:
                    acc['hi'] |= mask
                else:
                    acc['lo'] |= mask
        for bank, acc in sorted(banks.items()):
            if acc['hi'] or acc['lo']:
                self.set_ttl(bank, acc['lo'], acc['hi'])
            if acc['ovr_hi'] or acc['ovr_lo'] or acc['ovr_normal']:
                self.override_ttl(bank, acc['ovr_lo'], acc['ovr_hi'], acc['ovr_normal'])
        sent['ttl'] = banks

    def _apply_dds(self, dds_set, dds_cur, sent):
        """Port of labctrl-node #set_dds_vals. ``dds_set``/``dds_cur`` are flat dicts:
        {'<type><i>': word, 'ovr_<type><i>': bool}. A value written to an OVERRIDDEN
        channel routes to override_dds (not set_dds); clearing an override sends -1."""
        values = {0: {}, 1: {}, 2: {}}
        ovrs = {0: {}, 1: {}, 2: {}}
        for key, v in dds_set.items():
            m = re.match(r'^(freq|amp|phase)(\d+)$', key)
            if m:
                t, i = _NAME_TYP[m.group(1)], int(m.group(2))
                if i < NUM_DDS:
                    values[t][i] = int(round(float(v))) & _U32_MASK
                continue
            m = re.match(r'^ovr_(freq|amp|phase)(\d+)$', key)
            if m:
                t, i = _NAME_TYP[m.group(1)], int(m.group(2))
                if i < NUM_DDS:
                    ovrs[t][i] = bool(v)
        cmd, ovr_cmd = [], []
        for t in (TYP_FREQ, TYP_AMP, TYP_PHASE):
            tname = _TYP_NAME[t]
            for i in range(NUM_DDS):
                cmdi = i | (t << 6)
                ovr_set = ovrs[t].get(i)
                ovr = ovr_set if ovr_set is not None else bool(
                    dds_cur.get('ovr_%s%d' % (tname, i), False))
                val_set = values[t].get(i)
                val_cur = dds_cur.get('%s%d' % (tname, i))
                val = val_set if val_set is not None else val_cur
                skip_cmd = False
                if ovr_set is not None or (ovr and val_set is not None):
                    if not ovr:
                        ovr_cmd.append((cmdi, _DDS_NO_OVR))           # clear override (-1)
                    else:
                        ovr_cmd.append((cmdi, (val if val is not None else 0) & _U32_MASK))
                        skip_cmd = True
                if val_set is not None and not skip_cmd:
                    cmd.append((cmdi, val_set))
        if cmd:
            self._set_dds_raw('set_dds', cmd)
        if ovr_cmd:
            self._set_dds_raw('override_dds', ovr_cmd)
        sent['dds'] = {'set': cmd, 'ovr': ovr_cmd}

    def set_values(self, vals, cur=None):
        """Apply a batch of control changes, replicating labctrl-node's ``set_values``.

        ``vals`` / ``cur`` are flat dicts (cur = CURRENT values, for the override coupling):
          {'clock': int,
           'ttl': {'val<i>': bool, 'ovr<i>': bool},
           'dds': {'<type><i>': word, 'ovr_<type><i>': bool}}
        Emits the SAME set_clock / set_ttl / override_ttl / set_dds / override_dds frames
        labctrl-node does. Returns a dict describing what was sent."""
        cur = cur or {}
        sent = {}
        clock = vals.get('clock')
        if clock is not None:
            clock = int(clock)
            if 0 <= clock <= 255:
                self.set_clock(clock)
                sent['clock'] = clock
        if vals.get('ttl'):
            self._apply_ttl(vals['ttl'], cur.get('ttl', {}), sent)
        if vals.get('dds'):
            self._apply_dds(vals['dds'], cur.get('dds', {}), sent)
        return sent

    @staticmethod
    def flatten_current(snap):
        """Flatten a snapshot() result into the {'ttl':{val/ovr}, 'dds':{word/ovr}} shape
        that set_values() expects for ``cur``."""
        ttl_cur, dds_cur = {}, {}
        for r in snap.get('ttl', []) or []:
            c = r['chn']
            ttl_cur['val%d' % c] = bool(r.get('value'))
            ttl_cur['ovr%d' % c] = bool(r.get('ovr_lo') or r.get('ovr_hi'))
        for r in snap.get('dds', []) or []:
            c = r['chn']
            if r.get('freq_word') is not None:
                dds_cur['freq%d' % c] = r['freq_word']
            if r.get('amp_word') is not None:
                dds_cur['amp%d' % c] = r['amp_word']
            if r.get('phase_word') is not None:
                dds_cur['phase%d' % c] = r['phase_word']
            dds_cur['ovr_freq%d' % c] = r.get('ovr_freq') is not None
            dds_cur['ovr_amp%d' % c] = r.get('ovr_amp') is not None
            dds_cur['ovr_phase%d' % c] = r.get('ovr_phase') is not None
        return {'ttl': ttl_cur, 'dds': dds_cur}

    # -- high-level snapshot ------------------------------------------------
    def snapshot(self, include_ttl=True, ttl_max_chn=None):
        """Gather a full, dashboard-ready view in engineering units.

        One call performs several requests; individual sub-reads that fail are reported in
        the result rather than aborting the whole snapshot. Returns a JSON-able dict.

        Reading TTL state requires issuing ZERO-mask ``set_ttl``/``override_ttl`` frames
        (the daemon treats those as no-op "get"s). Pass ``include_ttl=False`` to skip them
        entirely -- nothing TTL-related is sent and ``ttl`` comes back empty.

        ``ttl_max_chn`` sets the highest TTL channel index to expose (e.g. 55 -> channels
        0..55, spanning two 32-bit banks). When None, the daemon's ``get_max_ttl`` is used
        if it looks valid (>=31), else a single bank (0..31). Pass the value from the
        engine ``config.yml`` (FPGA1.max_ttl_chn) -- it is authoritative for the rig and
        avoids the unreliable daemon ``get_max_ttl``.
        """
        out = {"url": self.url, "connected": False, "errors": {}}

        # liveness + ids
        try:
            sid, server_id = self.state_id()
            out["connected"] = True
            out["state_id"] = sid
            out["server_id"] = server_id
        except MolecubeError as e:
            out["errors"]["state_id"] = str(e)
            return out  # nothing else will work if the daemon is unreachable

        # NB: we deliberately do NOT call get_max_ttl -- labctrl-node doesn't either,
        # and on some daemon builds it returns the 1-byte error status (read as 1),
        # which would collapse the panel to channels 0-1. The channel count comes
        # from ttl_max_chn (engine config.yml) instead.
        for key, fn in (("name_id", lambda: self.name_id()[0]),
                        ("clock", self.get_clock)):
            try:
                out[key] = fn()
            except MolecubeError as e:
                out["errors"][key] = str(e)

        # names
        dds_names, ttl_names = {}, {}
        try:
            dds_names = self.get_dds_names()
        except MolecubeError as e:
            out["errors"]["dds_names"] = str(e)
        try:
            ttl_names = self.get_ttl_names()
        except MolecubeError as e:
            out["errors"]["ttl_names"] = str(e)

        # DDS values + overrides
        try:
            raw = self.get_dds_all()
            try:
                ovr = self.get_override_dds()
            except MolecubeError as e:
                ovr = []
                out["errors"]["override_dds"] = str(e)
            ovr_map = {}
            for typ, chn, val in ovr:
                ovr_map.setdefault(chn, {})[typ] = val
            chans = []
            for chn in sorted(raw):
                words = raw[chn]
                o = ovr_map.get(chn, {})
                chans.append({
                    "chn": chn,
                    "name": dds_names.get(chn, ""),
                    "freq_word": words.get("freq"),
                    "freq_hz": ftw_to_hz(words["freq"]) if "freq" in words else None,
                    "amp_word": words.get("amp"),
                    "amp": amp_to_frac(words["amp"]) if "amp" in words else None,
                    "phase_word": words.get("phase"),
                    "phase_deg": phase_to_deg(words["phase"]) if "phase" in words else None,
                    "ovr_freq": ftw_to_hz(o[TYP_FREQ]) if TYP_FREQ in o else None,
                    "ovr_amp": amp_to_frac(o[TYP_AMP]) if TYP_AMP in o else None,
                    "ovr_phase": phase_to_deg(o[TYP_PHASE]) if TYP_PHASE in o else None,
                })
            out["dds"] = chans
        except MolecubeError as e:
            out["errors"]["dds"] = str(e)
            out["dds"] = []

        # TTL values + overrides (bank 0). Skipped entirely when include_ttl is
        # False, so NO set_ttl/override_ttl frames are sent to the daemon.
        if not include_ttl:
            out["ttl"] = []
            out["ttl_reads_disabled"] = True
            return out
        try:
            # Channel count. Prefer the explicit ttl_max_chn (from the engine
            # config.yml, authoritative for the rig); else the daemon's get_max_ttl
            # only when it looks valid (>=31) -- a daemon that doesn't support it
            # replies with the 1-byte error status (value 1), which would wrongly
            # show only channels 0-1; else fall back to a single 32-bit bank.
            if ttl_max_chn is not None:
                n_ttl = int(ttl_max_chn) + 1
            else:
                mt = out.get("max_ttl")
                n_ttl = (int(mt) + 1) if isinstance(mt, int) and mt >= 31 else 32
            n_banks = (n_ttl + 31) // 32

            # Each bank is a 32-bit register; read the value + override masks for
            # every bank we need to cover channels 0..n_ttl-1.
            bank_val, bank_lo, bank_hi = {}, {}, {}
            for bank in range(n_banks):
                bank_val[bank] = self.get_ttl(bank)
                try:
                    bank_lo[bank], bank_hi[bank] = self.get_override_ttl(bank)
                except MolecubeError as e:
                    bank_lo[bank], bank_hi[bank] = 0, 0
                    out["errors"].setdefault("override_ttl", str(e))

            ttl = []
            for chn in range(0, n_ttl):
                bank, bit = chn // 32, 1 << (chn % 32)
                ovr_lo = bool(bank_lo.get(bank, 0) & bit)   # forced low
                ovr_hi = bool(bank_hi.get(bank, 0) & bit)   # forced high
                # Effective output: a forced channel shows its forced level (this
                # mirrors labctrl-node); otherwise the raw output-register bit.
                if ovr_hi:
                    value = True
                elif ovr_lo:
                    value = False
                else:
                    value = bool(bank_val.get(bank, 0) & bit)
                ttl.append({
                    "chn": chn,
                    "name": ttl_names.get(chn, ""),
                    "value": value,
                    "ovr_lo": ovr_lo,
                    "ovr_hi": ovr_hi,
                })
            out["ttl"] = ttl
            out["ttl_n_banks"] = n_banks
            out["ttl_value_masks"] = [bank_val.get(b, 0) for b in range(n_banks)]
        except MolecubeError as e:
            out["errors"]["ttl"] = str(e)
            out["ttl"] = []

        return out
