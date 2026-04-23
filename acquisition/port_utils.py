"""Utilities for TCP port management (Windows-focused).

Used by run_monitor and the runner launcher to clear a stale listener on a
port before binding a fresh one.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def kill_port(port: int) -> int:
    """Kill any process listening on the given TCP port.

    Returns the number of processes killed. Safe to call when nothing is
    listening; logs at INFO if a kill actually happens so the caller can
    correlate stuck-port symptoms with cleanup.
    """
    killed = 0
    try:
        out = subprocess.check_output(
            ['netstat', '-ano'], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.debug('netstat failed: %s', e)
        return 0

    own_pid = os.getpid()
    seen = set()
    for line in out.splitlines():
        if f':{port} ' not in line or 'LISTENING' not in line:
            continue
        parts = line.strip().split()
        if not parts:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid == own_pid or pid in seen:
            continue
        seen.add(pid)
        logger.info('Killing stale process on port %d (pid=%d)', port, pid)
        rc = subprocess.call(
            ['taskkill', '/PID', str(pid), '/F'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc == 0:
            killed += 1
    return killed
