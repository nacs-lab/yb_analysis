"""Utilities for TCP port management.

Used by run_monitor and the runner launcher to clear a stale listener on a
port before binding a fresh one. Works on Windows (netstat/taskkill) and
POSIX (lsof/kill).
"""

import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)


def _stale_pids_windows(port: int):
    # Use Get-NetTCPConnection rather than parsing `netstat -ano` because
    # netstat localizes the State column ("LISTENING" → "ABHÖREN" on
    # German, "ÉCOUTE" on French, etc.), so a regex match on "LISTENING"
    # silently fails on any non-English Windows.  Get-NetTCPConnection
    # filters on the State enum, which is locale-independent.
    ps_cmd = (
        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
        f"-ErrorAction SilentlyContinue | ForEach-Object {{ $_.OwningProcess }}"
    )
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_cmd],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception as e:
        logger.debug('Get-NetTCPConnection failed: %s', e)
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _stale_pids_posix(port: int):
    try:
        out = subprocess.check_output(
            ['lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
            text=True, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.debug('lsof not installed; cannot clean port %d', port)
        return []
    except subprocess.CalledProcessError:
        return []  # nothing listening
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                pids.append(int(line))
            except ValueError:
                pass
    return pids


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists (best-effort)."""
    if os.name == 'nt':
        try:
            out = subprocess.check_output(
                ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
        except Exception:
            return True  # can't tell -> assume alive so the caller keeps waiting
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_pids_gone(pids, timeout: float) -> bool:
    """Poll until none of ``pids`` exist, or ``timeout`` seconds elapse.

    Returns True iff all are gone. The point is correctness of a *handoff*: a
    freshly force-killed backend releases OS-level handles -- notably the single
    DCAM camera handle -- only as the process is torn down. A replacement
    spawned before that finishes races the release, and a contended DCAM open
    blocks indefinitely (holding the GIL, so nothing in the new backend can
    recover). Waiting here closes that window.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    remaining = list(pids)
    while remaining and time.monotonic() < deadline:
        remaining = [p for p in remaining if _pid_alive(p)]
        if remaining:
            time.sleep(0.2)
    return not remaining


def kill_port(port: int, *, wait: bool = True, wait_timeout: float = 10.0) -> int:
    """Kill any process listening on the given TCP port.

    Returns the number of processes killed. Safe to call when nothing is
    listening; logs at INFO if a kill actually happens so the caller can
    correlate stuck-port symptoms with cleanup.

    With ``wait=True`` (default) the call does not return until the killed
    processes have actually exited (or ``wait_timeout`` elapses), so a caller
    that is about to spawn a replacement can rely on the old process's handles
    -- including the single DCAM camera handle -- being released first.
    """
    own_pid = os.getpid()
    if os.name == 'nt':
        pids = _stale_pids_windows(port)
    else:
        pids = _stale_pids_posix(port)

    killed = 0
    killed_pids = []
    for pid in pids:
        if pid == own_pid:
            continue
        logger.info('Killing stale process on port %d (pid=%d)', port, pid)
        if os.name == 'nt':
            rc = subprocess.call(
                ['taskkill', '/PID', str(pid), '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc == 0:
                killed += 1
                killed_pids.append(pid)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
                killed_pids.append(pid)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                logger.warning('Could not kill pid=%d: %s', pid, e)
    if wait and killed_pids:
        if not _wait_pids_gone(killed_pids, wait_timeout):
            logger.warning(
                'port %d: killed pid(s) %s still present after %.0fs',
                port, killed_pids, wait_timeout)
    return killed
