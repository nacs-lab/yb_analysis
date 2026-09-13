"""Single-instance guard for run_monitor.

Why this exists
---------------
`kill_port(8050)` already scrubs a stale listener on the dashboard port, but the
dashboard listener is a *multiprocessing child* of run_monitor -- not the monitor
itself. So a second `python -m yb_analysis.scripts.run_monitor` would take over
port 8050 while the FIRST monitor process stayed alive and kept pulling frames
from the camera's ZMQ image socket.

Two live consumers on that socket is silently destructive: ZMQ round-robins
between equal peers, so each monitor receives roughly HALF the frames and each
writes half the shots to its own HDF5. On 2026-08-28 this cost 51% of the shots
in scan 20260828_170124 (467 sequences run, 230 shots on disk) and, because the
loss interleaves rather than truncates, it did not look like data loss downstream
-- it looked like physics (a freq-2D ridge slope of 0.52 instead of 1, a 10x-low
optimum survival, and a target-site count 3x the pattern's size). A duplicate
monitor also drives the job queue, which spawned a scan nobody submitted and
aborted the running one.

The guard therefore keys on the *monitor process*, not on any port: a lock file
holding the owning PID, validated against the live process table so a crashed
monitor's stale lock never blocks a restart.
"""
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

LOCK_NAME = 'yb_run_monitor.pid'


def _lock_path():
    return os.path.join(tempfile.gettempdir(), LOCK_NAME)


def _is_run_monitor(pid):
    """True if `pid` is alive AND its command line looks like a run_monitor.

    The command-line check matters: PIDs are recycled on Windows, and a bare
    liveness test would let an unrelated process's PID block startup forever.
    """
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        try:
            with open('/proc/%d/cmdline' % pid, 'rb') as fh:
                return b'run_monitor' in fh.read()
        except OSError:
            return False

    import subprocess
    try:
        out = subprocess.run(
            ['wmic', 'process', 'where', 'ProcessId=%d' % pid,
             'get', 'CommandLine', '/format:list'],
            capture_output=True, text=True, timeout=15).stdout
    except Exception as ex:                      # wmic missing / slow / blocked
        logger.warning('single-instance: cannot inspect pid %d (%s); '
                       'treating lock as stale', pid, ex)
        return False
    return 'run_monitor' in out


def acquire(force=False):
    """Claim the run_monitor lock, or raise RuntimeError naming the holder.

    Pass ``force=True`` (``--allow-duplicate``) to bypass. Registers no atexit
    hook of its own -- the caller owns release() so shutdown ordering stays
    explicit.
    """
    path = _lock_path()
    holder = None
    try:
        with open(path) as fh:
            holder = int((fh.read() or '').strip())
    except (OSError, ValueError):
        holder = None

    if holder and holder != os.getpid() and _is_run_monitor(holder):
        if not force:
            raise RuntimeError(
                'another run_monitor is already running (pid=%d).\n'
                'Two monitors split the camera ZMQ frames between them and each '
                'writes only ~half the shots -- silently, and the corruption '
                'mimics a physics anomaly downstream. Stop the other instance '
                'first, or use the GUI "Restart All" button.\n'
                'If you are certain the other process is not consuming frames, '
                'override with --allow-duplicate.' % holder)
        logger.warning('single-instance: overriding live monitor pid=%d '
                       '(--allow-duplicate); EXPECT ~50%% FRAME LOSS IN BOTH.',
                       holder)
    elif holder:
        logger.info('single-instance: clearing stale lock (pid=%d gone)', holder)

    try:
        with open(path, 'w') as fh:
            fh.write(str(os.getpid()))
    except OSError as ex:
        # A guard that cannot write its lock must not block the experiment.
        logger.warning('single-instance: cannot write %s (%s); '
                       'proceeding WITHOUT the guard', path, ex)
    return path


def release():
    """Drop the lock if we own it. Never raises."""
    path = _lock_path()
    try:
        with open(path) as fh:
            if int((fh.read() or '').strip()) != os.getpid():
                return
    except (OSError, ValueError):
        return
    try:
        os.remove(path)
    except OSError:
        pass
