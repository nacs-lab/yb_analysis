"""Utilities for TCP port management.

Used by run_monitor and the runner launcher to clear a stale listener on a
port before binding a fresh one. Works on Windows (netstat/taskkill) and
POSIX (lsof/kill).
"""

import logging
import os
import signal
import subprocess

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


def kill_port(port: int) -> int:
    """Kill any process listening on the given TCP port.

    Returns the number of processes killed. Safe to call when nothing is
    listening; logs at INFO if a kill actually happens so the caller can
    correlate stuck-port symptoms with cleanup.
    """
    own_pid = os.getpid()
    if os.name == 'nt':
        pids = _stale_pids_windows(port)
    else:
        pids = _stale_pids_posix(port)

    killed = 0
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
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
            except PermissionError as e:
                logger.warning('Could not kill pid=%d: %s', pid, e)
    return killed
