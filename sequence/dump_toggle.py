"""Dashboard side of the "save sequence dumps" toggle.

The toggle rides the pyctrl backend's tiny mmap runtime-state store (the same one
the 616-EOM ramp uses): ``<tempdir>/nacsctl/pyctrl_runtime_state.dat``, a uint8
flag at **offset 8** (0=off, 1=on). The dashboard (this module) writes it; the
pyctrl runner reads it at scan start via ``pyctrl/YbExptCtrl/runtime_state.py``.

This is a deliberate, documented layout MIRROR of that file (the two processes run
under different interpreters, so we don't import across them). Keep the offsets in
sync with ``runtime_state.py``:

    offset 0 (float64) -- FreqEOM616Old   (owned by pyctrl; we never touch it)
    offset 8 (uint8)   -- save_sequence_dumps toggle  (this module)

Plain file I/O (not mmap): the toggle changes at human speed, so a seek + 1-byte
read/write per call is plenty, and it avoids holding a second mmap handle on a file
the backend also maps. Writing pads a fresh/short file to the full layout with a NaN
freq at offset 0, so a dashboard-first write leaves the EOM slot correctly "unset".
"""

import logging
import os
import struct
import tempfile

logger = logging.getLogger(__name__)

_PATH = os.path.join(tempfile.gettempdir(), "nacsctl", "pyctrl_runtime_state.dat")
_SIZE = 9
_FLAG_OFF = 8
_DEFAULT = struct.pack("<d", float("nan")) + b"\x00"   # NaN freq + toggle off


def _ensure_file():
    """Create/pad the store to the full layout, preserving any existing freq bytes."""
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    if os.path.isfile(_PATH) and os.path.getsize(_PATH) >= _SIZE:
        return
    existing = b""
    try:
        if os.path.isfile(_PATH):
            with open(_PATH, "rb") as f:
                existing = f.read()
    except OSError:
        existing = b""
    buf = bytearray(_DEFAULT)
    if len(existing) >= 8:
        buf[0:8] = existing[0:8]            # never clobber the 616-EOM freq
    with open(_PATH, "wb") as f:
        f.write(bytes(buf))


def get_save_sequence_dumps(default=False):
    """Read the toggle (offset 8). ``default`` on a missing/short/unreadable store."""
    try:
        if not os.path.isfile(_PATH) or os.path.getsize(_PATH) < _SIZE:
            return bool(default)
        with open(_PATH, "rb") as f:
            f.seek(_FLAG_OFF)
            b = f.read(1)
        return b == b"\x01"
    except OSError as e:
        logger.debug("dump_toggle read failed (%s); using default", e)
        return bool(default)


def set_save_sequence_dumps(on):
    """Set the toggle (offset 8); returns the bool actually written."""
    on = bool(on)
    _ensure_file()
    with open(_PATH, "r+b") as f:
        f.seek(_FLAG_OFF)
        f.write(b"\x01" if on else b"\x00")
    return on
