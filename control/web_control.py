"""Reverse-channel command queue: web dashboard -> main run_monitor process.

Under the MATLAB backend, dashboard Pause / Start / Abort write the MemoryMap
directly (see :mod:`yb_analysis.control.memmap_signal`). Under the pyctrl
backend there is NO local MemoryMap, so those three are spooled here as
``pause`` / ``start`` / ``abort`` and the main process issues the ZMQ verb
(``ControlPanel._poll_web_control`` drains them on a fast ~300 ms loop so a
remote Abort stays sub-second). A few other commands must always run in the
MAIN process because they touch state the dashboard subprocess doesn't own:

* ``dummy_mode``   — the ExptServer ZMQ client + the Tkinter radio var,
* ``init_dir``     — the on-disk calibration loader that feeds the live
                     DataManager / detection (a dashboard-local load would
                     only change the dashboard's own view),
* ``restart_dash`` — the supervisor handle to the Dash subprocess,
* ``restart_all``  — respawning run_monitor itself,
* ``set_backend``  — switching the sequence backend (MATLAB <-> pyctrl),
                     a restart_all handoff with a different ``--backend``.

Those are enqueued here as small JSON files in a spool dir; the Tkinter
``ControlPanel`` drains and executes them on its main-thread status poll.

One file per command (atomic create + rename), so producer and consumer never
race on a shared file; the consumer deletes each file after handling it.
"""

import json
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)

CMD_DIR = os.path.join(tempfile.gettempdir(), 'nacsctl', 'web_cmds')

#: Forward-channel status the MAIN process publishes for the web sidebar
#: (dummy mode string, last-seq meta, current scan/seq/file/runner state).
#: The dashboard subprocess reads it via ``/api/control/status``. Distinct
#: from the snapshot's ``_dummy_mode`` (a mere "dummy active" bool).
STATUS_FILE = os.path.join(tempfile.gettempdir(), 'nacsctl', 'web_ctrl_status.json')

#: Forward-channel camera status the MAIN process publishes for the web
#: sidebar's Camera card (connected/roi/exposure/error + a busy flag).
#: Mirrors what CameraPane shows in the Tkinter window. Read by the
#: dashboard subprocess via ``/api/control/camera/status``.
CAMERA_STATUS_FILE = os.path.join(tempfile.gettempdir(), 'nacsctl',
                                  'web_camera_status.json')

#: Commands ControlPanel knows how to execute. The dashboard rejects anything
#: not in here before spooling, so a bad request can't pile up junk files.
#: The ``camera_*`` commands carry a roi (``[x,y,w,h]``) and/or exposure (s)
#: and are dispatched to the Tkinter CameraPane, which owns the ZMQ client.
VALID_CMDS = ('dummy_mode', 'init_dir', 'restart_dash', 'restart_all',
              'set_backend', 'shutdown',
              # Sequence control via ZMQ (pyctrl backend has no local memmap).
              'pause', 'start', 'abort',
              'camera_connect', 'camera_disconnect', 'camera_apply')


def enqueue(cmd, **fields):
    """Producer (dashboard side): spool a command for the main process.

    Returns the spool-file path. Raises ``ValueError`` for unknown commands.
    """
    if cmd not in VALID_CMDS:
        raise ValueError('unknown web-control command: %r' % (cmd,))
    os.makedirs(CMD_DIR, exist_ok=True)
    rec = dict(fields)
    rec['cmd'] = cmd
    rec['ts'] = time.time()
    # Unique + lexically sortable by creation order: <ns>_<pid>.json
    name = '%021d_%d.json' % (time.time_ns(), os.getpid())
    path = os.path.join(CMD_DIR, name)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return path


def publish_status(status):
    """Producer (main side): atomically publish the control-status dict."""
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    tmp = STATUS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(status, f)
    os.replace(tmp, STATUS_FILE)


def read_status():
    """Consumer (dashboard side): the latest control status, or None."""
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def publish_camera_status(status):
    """Producer (main side): atomically publish the camera-status dict
    (connected / roi / exposure_time / error / busy). Best-effort."""
    os.makedirs(os.path.dirname(CAMERA_STATUS_FILE), exist_ok=True)
    tmp = CAMERA_STATUS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(status, f)
    os.replace(tmp, CAMERA_STATUS_FILE)


def read_camera_status():
    """Consumer (dashboard side): the latest camera status, or None."""
    try:
        with open(CAMERA_STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def drain():
    """Consumer (main side): return pending commands oldest-first, removing
    each spool file. Best-effort — unreadable files are dropped, not retried."""
    if not os.path.isdir(CMD_DIR):
        return []
    out = []
    for name in sorted(os.listdir(CMD_DIR)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(CMD_DIR, name)
        try:
            with open(path) as f:
                out.append(json.load(f))
        except Exception as e:
            logger.debug('web_control.drain: bad cmd file %s: %s', name, e)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return out
