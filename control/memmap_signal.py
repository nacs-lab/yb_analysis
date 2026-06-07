"""Shared MATLAB MemoryMap signalling layout + helpers.

The MATLAB runtime (``matlab_new/YbExptCtrl/MemoryMap.m``) memory-maps a small
file under ``<tempdir>/nacsctl/nacs_mem_map.dat`` as the IPC channel between the
StartScan / runSeq session and the monitor. Both the Tkinter control panel
(``yb_analysis.gui.control_panel``) and the web dashboard
(``yb_analysis.plotting.dashboard``) write Pause / Abort / Start flags into it.

This module is the **single source of truth** for the byte offsets so the two
UIs can't drift from each other or from ``MemoryMap.m``. The layout below
mirrors the ``memmapfile`` Format cell array in ``MemoryMap.m`` field-for-field
(doubles are 8 bytes, uint8 is 1 byte; little-endian, matching MATLAB on x86);
offsets are derived from it rather than hand-counted.
"""

import logging
import mmap
import os
import struct
import tempfile

logger = logging.getLogger(__name__)

# Don't put a file with shared mapping on a network drive (per MemoryMap.m).
MMAP_PATH = os.path.join(tempfile.gettempdir(), 'nacsctl', 'nacs_mem_map.dat')

# (field, dtype, count) in MemoryMap.m order — the authoritative layout.
_LAYOUT = [
    ('ScanParamsSet',   'double', 1),
    ('AndorConfigured', 'double', 1),
    ('ScanComplete',    'double', 1),
    ('NumImages',       'double', 1),
    ('NumPerParamAvg',  'double', 1),
    ('NumPerGroup',     'double', 1),
    ('DateStamp',       'double', 1),
    ('TimeStamp',       'double', 1),
    ('AbortRunSeq',     'double', 1),
    ('PauseRunSeq',     'double', 1),
    ('IsPausedRunSeq',  'double', 1),
    ('CurrentSeqNum',   'double', 1),
    ('Email',           'uint8',  32),
    ('FreqEOM616Old',   'double', 1),
    ('AWGFreqs',        'double', 32),
    ('DummyRunning',    'double', 1),
]
_DTYPE_SIZE = {'double': 8, 'uint8': 1}


def _build_offsets():
    out, off = {}, 0
    for name, dtype, count in _LAYOUT:
        out[name] = off
        off += _DTYPE_SIZE[dtype] * count
    return out


#: ``{field_name: byte_offset}`` for every field in the MemoryMap.
OFFSETS = _build_offsets()

# Named byte offsets — the subset the UIs read/write. Keep the underscore-free
# public names here; ``control_panel`` aliases them back to its historical
# ``_OFF_*`` names on import so its method bodies stay byte-for-byte unchanged.
OFF_SCAN_COMPLETE = OFFSETS['ScanComplete']
OFF_NUM_PER_GROUP = OFFSETS['NumPerGroup']
OFF_ABORT = OFFSETS['AbortRunSeq']
OFF_PAUSE = OFFSETS['PauseRunSeq']
OFF_ISPAUSED = OFFSETS['IsPausedRunSeq']
OFF_CURSEQNUM = OFFSETS['CurrentSeqNum']
OFF_DUMMY_RUNNING = OFFSETS['DummyRunning']


def mmap_open(backend='matlab'):
    """Open the MemoryMap file for read/write, or ``None`` if unavailable.

    ``backend`` gates access: the MemoryMap is a **MATLAB-only** IPC channel.
    The pyctrl backend has NO local memmap and speaks pure ZMQ, so when
    ``backend != 'matlab'`` this returns ``None`` **unconditionally** — without
    even probing the path. This is the safety boundary: a *stale*
    ``nacs_mem_map.dat`` left on disk by a prior MATLAB session must NOT be read
    or written while pyctrl is the live backend (otherwise Pause/Abort would
    silently poke a file nobody reads). Callers treat ``None`` as "MemoryMap
    unavailable" and fall back to ZMQ verbs.

    For ``backend == 'matlab'`` the behavior is unchanged: open the file if
    present, else ``None`` (the MATLAB runtime may not have created it yet).
    """
    if backend != 'matlab':
        return None
    if not os.path.isfile(MMAP_PATH):
        return None
    try:
        f = open(MMAP_PATH, 'r+b')
        return mmap.mmap(f.fileno(), 0)
    except Exception as e:
        logger.debug('Could not open MemoryMap: %s', e)
        return None


def mmap_write_double(mm, offset, value):
    mm.seek(offset)
    mm.write(struct.pack('d', float(value)))


def mmap_read_double(mm, offset):
    mm.seek(offset)
    return struct.unpack('d', mm.read(8))[0]


# --- High-level signals shared by both UIs ---------------------------------
# Each opens the MemoryMap, writes the flag(s), and closes. Returns True when
# the MemoryMap was present and written, False otherwise — i.e. the file didn't
# exist OR ``backend != 'matlab'`` (pyctrl has no local memmap). On False the
# caller MUST fall back to the ZMQ pause_seq / start_seq / abort_seq verbs.
# These mirror ControlPanel._on_pause / _on_start / _on_abort exactly.

def signal_pause(backend='matlab'):
    mm = mmap_open(backend)
    if not mm:
        return False
    try:
        mmap_write_double(mm, OFF_PAUSE, 1.0)
    finally:
        mm.close()
    return True


def signal_start(backend='matlab'):
    mm = mmap_open(backend)
    if not mm:
        return False
    try:
        mmap_write_double(mm, OFF_PAUSE, 0.0)
    finally:
        mm.close()
    return True


def signal_abort(backend='matlab'):
    mm = mmap_open(backend)
    if not mm:
        return False
    try:
        mmap_write_double(mm, OFF_ABORT, 1.0)
        mmap_write_double(mm, OFF_PAUSE, 0.0)
    finally:
        mm.close()
    return True
