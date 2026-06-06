"""Reader for the flattened ``.seq`` files that SeqPlotter consumes.

A ``.seq`` is the *evaluated* output of a compiled sequence: for every channel a
list of ``(time, value, pulse_id)`` points, plus an optional parameters-JSON
block and an optional debug-backtrace block.  It is **not** the symbolic
``ExpSeq.serialize()`` blob the libnacs engine compiles -- it is what the engine
emits via ``get_nominal_output(pts_per_ramp)`` and what the MATLAB
``ExpSeq.dump_output_to_file`` / pyctrl ``lib/dump_output.py`` writers pack.

This module is the ``yb_analysis``-side reader. The wire-level :func:`decode` is
a standard-library port of the canonical codec in
``pyctrl/tools/compare_seq_bytes.py`` (which round-trips the real MATLAB sample
byte-for-byte); :func:`parse` shapes the result into numpy-backed views the
dashboard figure builder uses.

Byte format (all multibyte values little-endian)::

    [nseqs u32]
      per seq: [seq_name \\0][seq_idx u32]
               [nchns u32]( [chn_name \\0][npts u32]( [t i64][v f64][pid u32] )*npts )*nchns
               [has_params u8][params_json \\0]?
    [has_bt_info u8]
      if set: [bt_idx u32]*nseqs  [n_bts u32]
              ( [nfilenames u32][name\\0]* [nnames u32][name\\0]*
                [nobjs u32]( [nframes u32]( [fname_id u32][name_id u32][line u32] )*nframes )*nobjs )*n_bts

Times are engine ticks (1 tick = 1 ps with the production ``tick_per_sec=1e12``);
multiply by ``1e-9`` to get milliseconds, matching SeqPlotter's x-axis.
"""

import json
import struct
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# pulse_id sentinel for a default / non-pulse point (carries no backtrace).
PULSE_ID_DEFAULT = 0xFFFFFFFF  # 2**32 - 1


# --------------------------------------------------------------------------- #
# Wire-level decode -- stdlib only; mirrors pyctrl/tools/compare_seq_bytes.py.
# --------------------------------------------------------------------------- #
class _Cur:
    """Forward cursor over a byte buffer (little-endian readers)."""

    def __init__(self, data):
        self.d = bytes(data)
        self.i = 0

    def take(self, n):
        if self.i + n > len(self.d):
            raise ValueError("overrun at byte %d (+%d > %d)" % (self.i, n, len(self.d)))
        b = self.d[self.i:self.i + n]
        self.i += n
        return b

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def cstr(self):
        start = self.i
        while self.i < len(self.d) and self.d[self.i] != 0:
            self.i += 1
        if self.i >= len(self.d):
            raise ValueError("unterminated string from byte %d" % start)
        s = self.d[start:self.i].decode("latin1")
        self.i += 1  # skip the NUL
        return s


def _decode_channels(c):
    """One ``get_nominal_output`` channel block -> list of raw point arrays."""
    chns = []
    for _ in range(c.u32()):
        name = c.cstr()
        npts = c.u32()
        # 20 bytes per point: '<qdI' (time i64, value f64, pulse_id u32).
        raw = c.take(20 * npts)
        chns.append({"name": name, "npts": npts, "raw": raw})
    return chns


def _decode_backtrace_payload(c):
    filenames = [c.cstr() for _ in range(c.u32())]
    names = [c.cstr() for _ in range(c.u32())]
    objs = []
    for _ in range(c.u32()):
        frames = [{"fname_id": c.u32(), "name_id": c.u32(), "line": c.u32()}
                  for _ in range(c.u32())]
        objs.append(frames)
    return {"filenames": filenames, "names": names, "objs": objs}


def decode(data):
    """Decode ``.seq`` bytes into a nested dict. Raises ``ValueError`` on
    malformed input (overrun, unterminated string, or trailing bytes).

    Channel points are left as the packed 20-byte records (``raw``) so the
    numpy unpacking happens once, in :func:`parse`.
    """
    c = _Cur(data)
    seqs = []
    for _ in range(c.u32()):
        s = {"seq_name": c.cstr(), "seq_idx": c.u32(), "channels": _decode_channels(c)}
        s["has_params"] = c.u8()
        if s["has_params"]:
            raw = c.cstr()
            s["params_raw"] = raw
            try:
                s["params"] = json.loads(raw) if raw else None
            except ValueError:
                s["params"] = None  # keep raw; tolerate non-strict-JSON blobs
        else:
            s["params_raw"] = None
            s["params"] = None
        seqs.append(s)

    out = {"seqs": seqs, "has_bt_info": c.u8()}
    if out["has_bt_info"]:
        out["bt_idx"] = [c.u32() for _ in range(len(seqs))]
        out["backtraces"] = [_decode_backtrace_payload(c) for _ in range(c.u32())]

    if c.i != len(c.d):
        raise ValueError("trailing %d bytes after decode" % (len(c.d) - c.i))
    return out


# --------------------------------------------------------------------------- #
# Numpy-backed views for the dashboard.
# --------------------------------------------------------------------------- #
@dataclass
class Channel:
    """One channel's evaluated waveform."""

    name: str
    t: np.ndarray    # int64, engine ticks
    v: np.ndarray    # float64
    pid: np.ndarray  # uint32 (PULSE_ID_DEFAULT marks default/non-pulse points)

    @property
    def t_ms(self):
        """Times in milliseconds (ticks * 1e-9), matching SeqPlotter's x-axis."""
        return self.t.astype(np.float64) * 1e-9

    @property
    def is_frequency(self):
        """Heuristic SeqPlotter uses to route a channel to the 2nd y-axis."""
        return self.v.size > 0 and float(np.nanmax(self.v)) >= 1e6


@dataclass
class Frame:
    """One resolved backtrace frame (source location of a pulse)."""

    file: str
    name: str
    line: int


@dataclass
class Sequence:
    """One basic sequence within a ``.seq`` (a scan may pack several)."""

    name: str
    seq_idx: int
    channels: List[Channel]
    params: Optional[dict] = None
    params_raw: Optional[str] = None
    _bt: Optional[dict] = field(default=None, repr=False)

    @property
    def channel_names(self):
        return [c.name for c in self.channels]

    def channel(self, name):
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)

    def backtrace(self, pulse_id):
        """Resolved :class:`Frame` list for ``pulse_id``; ``[]`` if the point is
        a default value (``PULSE_ID_DEFAULT``), out of range, or no backtrace
        block was present."""
        pulse_id = int(pulse_id)
        if self._bt is None or pulse_id == PULSE_ID_DEFAULT:
            return []
        objs = self._bt["objs"]
        if pulse_id < 0 or pulse_id >= len(objs):
            return []
        fnames = self._bt["filenames"]
        names = self._bt["names"]
        frames = []
        for fr in objs[pulse_id]:
            fi, ni = fr["fname_id"], fr["name_id"]
            frames.append(Frame(
                file=fnames[fi] if 0 <= fi < len(fnames) else "?",
                name=names[ni] if 0 <= ni < len(names) else "?",
                line=fr["line"],
            ))
        return frames


@dataclass
class SeqDump:
    """A parsed ``.seq`` file: one or more :class:`Sequence` objects."""

    sequences: List[Sequence]
    has_bt_info: bool

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def by_name(self, name):
        for s in self.sequences:
            if s.name == name:
                return s
        raise KeyError(name)


def _unpack_points(raw, npts):
    """Vectorized unpack of ``npts`` packed ``<qdI`` records into (t, v, pid)."""
    if npts == 0:
        return (np.empty(0, np.int64), np.empty(0, np.float64), np.empty(0, np.uint32))
    # Read each field with its native stride out of the 20-byte records.
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(npts, 20)
    t = buf[:, 0:8].copy().view(np.int64).reshape(npts)
    v = buf[:, 8:16].copy().view(np.float64).reshape(npts)
    pid = buf[:, 16:20].copy().view(np.uint32).reshape(npts)
    return t, v, pid


def parse(data):
    """Parse ``.seq`` bytes into a :class:`SeqDump` of numpy-backed views."""
    raw = decode(data)
    has_bt = bool(raw["has_bt_info"])
    bts = raw.get("backtraces")
    bt_idx = raw.get("bt_idx")

    sequences = []
    for i, s in enumerate(raw["seqs"]):
        channels = []
        for ch in s["channels"]:
            t, v, pid = _unpack_points(ch["raw"], ch["npts"])
            channels.append(Channel(name=ch["name"], t=t, v=v, pid=pid))

        bt = None
        if has_bt and bts is not None and bt_idx is not None and i < len(bt_idx):
            k = bt_idx[i]
            if 0 <= k < len(bts):
                bt = bts[k]

        sequences.append(Sequence(
            name=s["seq_name"], seq_idx=s["seq_idx"], channels=channels,
            params=s.get("params"), params_raw=s.get("params_raw"), _bt=bt))

    return SeqDump(sequences=sequences, has_bt_info=has_bt)


def load(path):
    """Read and :func:`parse` a ``.seq`` file."""
    with open(path, "rb") as f:
        return parse(f.read())
