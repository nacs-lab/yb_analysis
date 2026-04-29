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
                 *, mock=False, extra_env=None, reuse=False):
        self._exe = matlab_exe
        self._root = matlab_root
        self._url = url
        self._mock = mock
        self._extra_env = extra_env or {}
        self._reuse = reuse
        self._owned = True  # set to False when reusing an existing runner
        self._proc = None  # type: Optional[subprocess.Popen]

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, boot_timeout: float = 45.0) -> None:
        """Kill stale port binders, spawn the runner, wait for ping.

        When reuse=True the launcher acts as a zombie mitigation:
          * If a SequenceRunner already responds at self._url, skip the
            spawn and reuse it.
          * stop() never shuts the runner down regardless of who started
            it — caller takes responsibility for MATLAB's lifecycle.
        The AslDma (Active Silicon FireBird) kernel driver hangs MATLAB
        at process exit on this rig, so each GUI close currently leaves
        a Windows-zombie MATLAB. Reusing one long-lived MATLAB across
        many GUI sessions limits zombie creation to one per reboot."""
        if self._reuse:
            self._owned = False  # never kill the runner we touch in reuse mode
            if _ping(self._url, timeout_ms=1000):
                logger.info('Reusing existing SequenceRunner at %s', self._url)
                return
            logger.info('No existing runner — spawning one (reuse mode: '
                        'will leave it alive on stop())')
        # Kill stale MATLAB runners first — _kill_stale_runners calls
        # _kill_dcam_tray() before taskkill, which unblocks any DCAM kernel
        # wait so the kill actually lands. Running kill_port first risks
        # taskkilling a stuck MATLAB without dcamtray cleanup, which leaks
        # the DCAM handle and makes the next OrcaInit fail with 0 frames.
        # Mirrors the order used in _force_kill().
        self._kill_stale_runners()
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
            cmd += ['-wait', '-minimize']
        # MATLAB stdout capture. Defaults to
        #   <project_root>/log/matlab_log/runner_<timestamp>.log
        # so each runner session leaves a record on disk; previous sessions'
        # logs are preserved (MATLAB's -logfile truncates per session, so we
        # use a per-launch filename). Override with the env var
        # YB_MATLAB_LOGFILE — '%t' in that path is substituted with the
        # launch timestamp.
        # NOTE: -logfile flushes synchronously, which is what makes shutdown
        # hangs visible. Cost is negligible on a local SSD; avoid network paths.
        log_path = os.environ.get('YB_MATLAB_LOGFILE') or os.path.join(
            os.path.dirname(self._root),
            'log', 'matlab_log', 'runner_%t.log')
        log_path = log_path.replace('%t', time.strftime('%Y%m%d_%H%M%S'))
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.isdir(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError:
                pass
        cmd += ['-logfile', log_path]
        logger.info('MATLAB logfile: %s', log_path)
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
        """Force-kill the runner without sending the graceful 'shutdown' ZMQ.

        Why no graceful shutdown? On this rig the graceful path makes MATLAB
        call its own `exit`, which triggers DLL_PROCESS_DETACH on the
        DCAM/Phoenix adaptor DLLs, which in turn calls into the
        Active-Silicon FireBird (AslDma) kernel driver. That driver hangs
        and leaves MATLAB.exe as an unkillable Windows zombie (1 thread,
        ~1920 handles, taskkill /F → "no running instance").

        TerminateProcess called from a different process (taskkill /F)
        skips DllMain per MS docs — so killing MATLAB from outside before
        it tries to gracefully exit avoids the hang. Empirically: 0 new
        zombies across many close cycles.

        No settle delay. An earlier version slept 3s on the theory that
        MATLAB needed time to run closeCameraGracefully or that AslDma
        needed to settle after imaqreset. That was untested
        speculation. 31 cycles tested across settle=0.0/0.2/3.0 (with
        wait_for_camera_connected gating scan submission) showed 0
        failures regardless of settle: the next session's
        imaqreset-at-init in handleCameraCmd 'init' recovers from any
        state the previous force-kill leaves behind. The "sometimes
        0 frames" failure that motivated the 3s was actually a race
        between camera_init and concurrent submitter MATLAB processes —
        see ZmqClient.camera_init for that fix.

        The `grace` argument is kept for API compat but is unused. With
        reuse=True, leave the runner alive entirely (caller owns it).
        """
        if not self._owned:
            logger.info('Leaving externally-owned SequenceRunner running')
            return
        if not self.is_alive():
            self._proc = None
            return
        logger.info('Force-killing SequenceRunner (skipping graceful exit '
                    'to avoid DCAM DLL_PROCESS_DETACH hang)')
        self._force_kill()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error('Runner kill timed out')
        self._proc = None

    @staticmethod
    def _find_runner_pids():
        """Return PIDs of all `MATLAB.exe` processes whose command line
        contains `SequenceRunner`. Uses PowerShell's Get-CimInstance because
        `wmic` on Windows 11 emits UTF-16-with-BOM which Python's text-mode
        subprocess misdecodes — every line becomes unparseable garbage and
        the old regex/isdigit check silently matched nothing, which is why
        zombies accumulated indefinitely on this box."""
        if os.name != 'nt':
            return []
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='MATLAB.exe'\" "
            "| Where-Object { $_.CommandLine -match 'SequenceRunner' } "
            "| ForEach-Object { $_.ProcessId }"
        )
        try:
            out = subprocess.check_output(
                ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_cmd],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
        except Exception as ex:
            logger.debug('_find_runner_pids failed: %s', ex)
            return []
        pids = []
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    @staticmethod
    def _kill_dcam_tray():
        """Kill dcamtray.exe before killing MATLAB.

        When MATLAB holds a live DCAM camera handle, its image-acquisition
        thread is stuck in an uninterruptible kernel wait inside the DCAM
        driver. taskkill /F sends TerminateProcess() but Windows cannot
        terminate a thread that is blocked inside a non-paged driver routine.

        Killing dcamtray.exe causes the DCAM API to signal a device-lost
        error to any thread currently waiting inside the driver. That
        unblocks MATLAB's acquisition thread, allowing TerminateProcess()
        to succeed on the next attempt. Dcamtray.exe restarts automatically
        on next camera init."""
        if os.name != 'nt':
            return
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='dcamtray.exe'\" "
            "| ForEach-Object { $_.ProcessId }"
        )
        try:
            out = subprocess.check_output(
                ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_cmd],
                text=True, stderr=subprocess.DEVNULL, timeout=10)
        except Exception as ex:
            logger.debug('_kill_dcam_tray: query failed: %s', ex)
            return
        pids = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
        if not pids:
            return
        logger.info('Killing dcamtray.exe (PIDs %s) to unblock DCAM kernel wait', pids)
        for pid in pids:
            subprocess.call(
                ['taskkill', '/PID', str(pid), '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)  # give the driver time to signal device-lost to MATLAB

    @staticmethod
    def _kill_stale_runners():
        """Kill leftover MATLAB runners from previous sessions."""
        pids = RunnerLauncher._find_runner_pids()
        if not pids:
            return
        logger.info('Found %d stale runner MATLAB(s) to kill: %s', len(pids), pids)
        # Kill dcamtray first — if MATLAB is stuck in a DCAM kernel wait,
        # taskkill /F alone cannot reach it. Killing dcamtray signals a
        # device-lost error that unblocks the stuck thread.
        RunnerLauncher._kill_dcam_tray()
        unkilled = []
        for pid in pids:
            rc = subprocess.call(
                ['taskkill', '/PID', str(pid), '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc != 0:
                unkilled.append(pid)
        if unkilled:
            logger.warning(
                'Could NOT kill stale runner MATLAB(s) %s — stuck in DCAM '
                'kernel wait even after dcamtray kill. A reboot may be needed.',
                unkilled)

    def _force_kill(self) -> None:
        """Kill the MATLAB runner process on all platforms.

        On Windows, ``matlab.exe -wait`` spawns the real MATLAB JVM
        (``bin\\win64\\MATLAB.exe``) as a child.  Strategies:
          0. Kill dcamtray.exe to unblock any DCAM kernel wait.
          1. Kill the launcher process tree (``taskkill /T``).
          2. Find the JVM by ParentProcessId and kill it directly.
          3. Kill whoever owns port 1408.
        """
        launcher_pid = self._proc.pid
        if os.name == 'nt':
            # 0. Unblock DCAM kernel wait before killing MATLAB
            self._kill_dcam_tray()
            # 1. Kill the launcher tree
            subprocess.call(
                ['taskkill', '/PID', str(launcher_pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # 2. Kill any surviving SequenceRunner JVM
            for pid in self._find_runner_pids():
                logger.info('Killing MATLAB runner JVM pid=%s', pid)
                subprocess.call(
                    ['taskkill', '/PID', str(pid), '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # 3. Kill any process still holding the port
            killed = kill_port(_url_to_port(self._url))
            if killed:
                logger.info('Killed %d process(es) on port', killed)
        else:
            self._proc.kill()
