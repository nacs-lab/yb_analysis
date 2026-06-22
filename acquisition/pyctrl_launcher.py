"""Spawn + supervise the background pyctrl Python backend.

Drop-in alternative to :class:`RunnerLauncher` (the MATLAB ``SequenceRunner``).
Both expose the same ``start()/stop()/is_alive()`` surface so ``run_monitor``
can pick one at launch based on the selected backend. The pyctrl backend hosts
the *same* ExptServer (same ZMQ verbs at the same URL) the MATLAB runner does,
so everything downstream of the wire — ZmqClient, detection, dashboard — is
unchanged regardless of which backend is live.

The cross-backend handoff (releasing the single DCAM camera handle + the ZMQ
port) is handled by run_monitor's relaunch sequence, not here: the *previous*
monitor process force-kills its backend before this process binds (gated on
``YB_WAIT_FOR_PID``). So by the time ``start()`` runs, the port and camera are
already free.

Note: the pyctrl run-loop entry point (``config.PYCTRL_MODULE``) is a Phase-5
deliverable. Until it exists, ``start()`` raises (the spawned process exits with
"No module named ..."); run_monitor catches that and brings up a backend-down
GUI the user can switch back from.
"""

import logging
import os
import subprocess
import time
from typing import Optional

# Reuse the ZMQ liveness probe + port parser from the MATLAB launcher — both
# are backend-agnostic (they speak to the ExptServer over the wire).
from yb_analysis.acquisition.runner_launcher import _ping, _url_to_port
from yb_analysis.acquisition.port_utils import kill_port

logger = logging.getLogger(__name__)


class PyctrlLauncher:
    def __init__(self, python, module, url, *, cwd=None,
                 extra_env=None, reuse=False):
        self._python = python
        self._module = module
        self._url = url
        self._cwd = cwd
        self._extra_env = extra_env or {}
        self._reuse = reuse
        self._owned = True  # set False when reusing an existing backend
        self._proc = None  # type: Optional[subprocess.Popen]

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def take_ownership(self) -> None:
        """Force stop() to actually tear the backend down even if we adopted
        it in reuse mode. Used by a backend SWITCH: the old backend must die
        (it holds the single camera handle + port) so the new one can bind."""
        self._owned = True

    def start(self, boot_timeout: float = 45.0) -> None:
        """Clear the port, spawn ``python -m <module> <url>``, wait for ping.

        With reuse=True, an already-responding backend is adopted and never
        shut down on stop() (mirrors RunnerLauncher's zombie-mitigation API)."""
        if self._reuse:
            self._owned = False
            if _ping(self._url, timeout_ms=1000):
                logger.info('Reusing existing pyctrl backend at %s', self._url)
                return
            logger.info('No existing pyctrl backend — spawning one (reuse mode: '
                        'will leave it alive on stop())')

        # Free the port in case a stale binder (a dead backend of either kind)
        # is still holding it. The MATLAB side is torn down by the previous
        # monitor's _on_close before we get here; this is a belt-and-braces
        # scrub so the bind never races. kill_port now WAITS for the killed
        # process to fully exit, so its single DCAM camera handle is released
        # before we spawn below.
        kill_port(_url_to_port(self._url))
        # Settle: even after the old process is gone, the USB/DCAM device needs
        # a moment to re-enumerate. Spawning the replacement too eagerly makes
        # its camera open race the release, and a contended DCAM open blocks the
        # new backend indefinitely (the "Orca not responding to the controller"
        # wedge). Override with YB_BACKEND_SPAWN_SETTLE_S (seconds; 0 disables).
        try:
            settle = float(os.environ.get('YB_BACKEND_SPAWN_SETTLE_S', '1.5'))
        except (TypeError, ValueError):
            settle = 1.5
        if settle > 0:
            logger.info('Settling %.1fs after port clear so the DCAM handle '
                        'is free before spawn', settle)
            time.sleep(settle)

        env = os.environ.copy()
        # Put the pyctrl package root on sys.path so `-m <module>` resolves
        # even if cwd differs. `-m` already prepends cwd, but be explicit.
        if self._cwd:
            existing = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = (
                self._cwd + (os.pathsep + existing if existing else ''))
        env.update({k: str(v) for k, v in self._extra_env.items()})

        cmd = [self._python, '-u', '-m', self._module, self._url]
        logger.info('Spawning pyctrl backend: %s', ' '.join(cmd))
        creationflags = 0
        if os.name == 'nt':
            # Own process group so we can signal it without hitting our console.
            creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        self._proc = subprocess.Popen(
            cmd, env=env, cwd=self._cwd or None, creationflags=creationflags)

        deadline = time.monotonic() + boot_timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
                raise RuntimeError(
                    f"pyctrl backend exited during boot "
                    f"(rc={self._proc.returncode}). Is "
                    f"'{self._module}' importable by {self._python}? "
                    f"(Phase-5 run loop may not be built yet.)")
            if _ping(self._url):
                logger.info('pyctrl backend alive at %s', self._url)
                return
            time.sleep(0.5)

        self.stop(grace=2.0)
        raise TimeoutError(
            f"pyctrl backend did not respond to ping within {boot_timeout:.0f}s")

    def stop(self, grace: float = 5.0) -> None:
        """Terminate the backend process. No DCAM-zombie dance needed: the
        pyctrl backend is a plain Python process, so a normal terminate()
        releases the camera via pylablib's handle teardown. We still kill the
        port as a final scrub. With reuse=True the backend is left running."""
        if not self._owned:
            logger.info('Leaving externally-owned pyctrl backend running')
            return
        if not self.is_alive():
            self._proc = None
            return
        logger.info('Stopping pyctrl backend (terminate)')
        try:
            self._proc.terminate()
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning('pyctrl backend did not exit on terminate; killing')
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception as e:
                logger.error('pyctrl backend kill failed: %s', e)
        except Exception as e:
            logger.warning('pyctrl backend terminate failed: %s', e)
        # Final scrub: whatever still holds the ZMQ port.
        try:
            kill_port(_url_to_port(self._url))
        except Exception:
            pass
        self._proc = None
