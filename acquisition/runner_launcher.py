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
            cmd.append('-minimize')
        cmd += [
            '-sd', self._root,
            '-r', "addpath(genpath(pwd)); SequenceRunner(); exit",
        ]
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

    def stop(self, grace: float = 10.0) -> None:
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
            logger.warning('Runner did not exit within %.1fs — killing', grace)
            self._proc.kill()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.error('Runner kill timed out')
        self._proc = None
