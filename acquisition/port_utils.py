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
    try:
        out = subprocess.check_output(
            ['netstat', '-ano', '-p', 'tcp'], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.debug('netstat failed: %s', e)
        return []
    pids = []
    # Look for the port at either end of the local address column. Use a
    # regex tolerant of variable whitespace / tabs (netstat formatting
    # differs across Windows versions and locales).
    import re
    pat = re.compile(
        r'^\s*TCP\s+\S*?[.:]' + re.escape(str(port)) + r'\b.*LISTENING\s+(\d+)\s*$',
        re.IGNORECASE)
    for line in out.splitlines():
        m = pat.match(line)
        if not m:
            continue
        try:
            pids.append(int(m.group(1)))
        except ValueError:
            pass
    if not pids:
        # Only log a short snippet of candidate lines so a missed match is
        # diagnosable without flooding the log.
        candidates = [l for l in out.splitlines() if f':{port}' in l][:5]
        if candidates:
            logger.debug(
                'netstat has :%d entries but none parsed as LISTENING. '
                'Sample: %r', port, candidates)
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
