"""Spawn + supervise the background MATLAB SequenceRunner.

Owned by run_monitor.py — launched on startup, shut down via atexit.
"""

import logging
import os
import subprocess
import time
from typing import Optional

import zmq

from yb_analysis.acquisition.port_utils import kill_port

logger = logging.getLogger(__name__)


def _url_to_port(url: str) -> int:
    return int(url.rsplit(':', 1)[-1].split('/')[0])


def _ping(url: str, timeout_ms: int = 500) -> bool:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(url)
        sock.send_string("ping")
        if sock.poll(timeout_ms) == 0:
            return False
        return sock.recv_string() == "pong"
    except Exception:
        return False
    finally:
        sock.close(linger=0)


class RunnerLauncher:
    def __init__(self, matlab_exe, matlab_root, url,
                 *, mock=False, extra_env=None):
        self._exe = matlab_exe
        self._root = matlab_root
        self._url = url
        self._mock = mock
        self._extra_env = extra_env or {}
        self._proc = None  # type: Optional[subprocess.Popen]

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, boot_timeout: float = 45.0) -> None:
        """Kill stale port binders, spawn the runner, wait for ping."""
        kill_port(_url_to_port(self._url))
        self._kill_stale_runners()

        env = os.environ.copy()
        if self._mock:
            env['NACS_MOCK'] = '1'
        env.update({k: str(v) for k, v in self._extra_env.items()})

        # -minimize on Windows keeps the JVM (needed for yaml) but hides the
        # window. On macOS/Linux -nodesktop is enough; -minimize is a no-op.
        cmd = [
            self._exe,
            '-nodesktop', '-nosplash',
        ]
        if os.name == 'nt':
            cmd += ['-wait', '-minimize']
        # Pass the URL explicitly so Python and MATLAB can never drift —
        # escape single quotes inside the URL for MATLAB's string literal.
        url_esc = self._url.replace("'", "''")
        r_arg = (
            f"addpath(genpath(pwd)); "
            f"SequenceRunner('{url_esc}'); "
            f"exit"
        )
        cmd += ['-sd', self._root, '-r', r_arg]
        logger.info('Spawning runner: %s', ' '.join(cmd))
        self._proc = subprocess.Popen(cmd, env=env)

        deadline = time.monotonic() + boot_timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
                raise RuntimeError(
                    f"Runner process exited during boot "
                    f"(rc={self._proc.returncode})")
            if _ping(self._url):
                logger.info('Runner alive at %s', self._url)
                return
            time.sleep(0.5)

        # boot timed out — tear down
        self.stop(grace=2.0)
        raise TimeoutError(
            f"Runner did not respond to ping within {boot_timeout:.0f}s")

    def stop(self, grace: float = 20.0) -> None:
        """Graceful shutdown: ZMQ `shutdown` → wait → force-kill if needed."""
        if not self.is_alive():
            self._proc = None
            return
        try:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._url)
            sock.send_string("shutdown")
            sock.poll(1000)  # best-effort; ignore reply content
            sock.close(linger=0)
        except Exception as e:
            logger.debug('shutdown ZMQ send failed: %s', e)

        try:
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning('Runner did not exit within %.1fs — force killing', grace)
            self._force_kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error('Runner kill timed out')
        self._proc = None

    @staticmethod
    def _kill_stale_runners():
        """Kill leftover MATLAB runners from previous sessions."""
        if os.name != 'nt':
            return
        try:
            out = subprocess.check_output(
                ['wmic', 'process', 'where',
                 "Name='MATLAB.exe' AND CommandLine LIKE '%SequenceRunner%'",
                 'get', 'ProcessId'],
                text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    logger.info('Killing stale runner MATLAB pid=%s', line)
                    subprocess.call(
                        ['taskkill', '/PID', line, '/F'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _force_kill(self) -> None:
        """Kill the MATLAB runner process on all platforms.

        On Windows, ``matlab.exe -wait`` spawns the real MATLAB JVM
        (``bin\\win64\\MATLAB.exe``) as a child.  Three strategies:
          1. Kill the launcher process tree (``taskkill /T``).
          2. Find the JVM by ParentProcessId and kill it directly.
          3. Kill whoever owns port 1408.
        """
        launcher_pid = self._proc.pid
        if os.name == 'nt':
            # 1. Kill the launcher tree
            subprocess.call(
                ['taskkill', '/PID', str(launcher_pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # 2. Find the JVM child by ParentProcessId (survives launcher death)
            try:
                out = subprocess.check_output(
                    ['wmic', 'process', 'where',
                     f'ParentProcessId={launcher_pid}',
                     'get', 'ProcessId'],
                    text=True, stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        logger.info('Killing MATLAB JVM child pid=%s', line)
                        subprocess.call(
                            ['taskkill', '/PID', line, '/F'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            except Exception:
                pass
            # 3. Kill any process still holding the port
            killed = kill_port(_url_to_port(self._url))
            if killed:
                logger.info('Killed %d process(es) on port', killed)
        else:
            self._proc.kill()
