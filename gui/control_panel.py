"""Tkinter GUI for experiment control.

Processing runs in a background thread to keep the UI responsive.
Pause/Abort/Start are backend-aware: under the MATLAB backend they signal via
the MemoryMap file; under pyctrl (no local memmap) they use the ZMQ verbs.
"""

import os
import tkinter as tk
import tkinter.ttk as ttk
import logging
import traceback
import threading
import time

import numpy as np

from yb_analysis.acquisition.data_manager import (
    get_data_manager, record_loading, get_loading_history,
)
# MemoryMap layout + helpers now live in a shared module so the web dashboard
# and this Tkinter window can't drift (Phase 5.5). Aliased back to the local
# _OFF_* / _mmap_* names so the method bodies below stay unchanged.
from yb_analysis.control.memmap_signal import (
    MMAP_PATH as _MMAP_PATH,
    OFF_SCAN_COMPLETE as _OFF_SCAN_COMPLETE,
    OFF_NUM_PER_GROUP as _OFF_NUM_PER_GROUP,
    OFF_ABORT as _OFF_ABORT,
    OFF_PAUSE as _OFF_PAUSE,
    OFF_ISPAUSED as _OFF_ISPAUSED,
    OFF_CURSEQNUM as _OFF_CURSEQNUM,
    OFF_DUMMY_RUNNING as _OFF_DUMMY_RUNNING,
    mmap_open as _mmap_open,
    mmap_write_double as _mmap_write_double,
    mmap_read_double as _mmap_read_double,
)
from yb_analysis.control import web_control as _web_control
from yb_analysis.config import VALID_BACKENDS

logger = logging.getLogger(__name__)

# Sentinel scan_id for FAILING-shot frames the backend publishes for LIVE DISPLAY ONLY (never
# persisted / accumulated). Distinct from dummy mode's -1 so the view can label them "failing"
# rather than "dummy". MUST match pyctrl/YbExptCtrl/rearrange_runtime.py FAILING_DISPLAY_SCAN_ID.
FAILING_DISPLAY_SCAN_ID = -2


def _monitor_logfile():
    """Timestamped logfile for a windowless 'Restart All' respawn.

    ``<project_root>/log/monitor_log/run_monitor_<YYYYMMDD_HHMMSS>.log`` -- sits
    beside the MATLAB ``log/matlab_log/`` and pyctrl ``log/pyctrl_log/`` dirs.
    Override the directory with ``YB_MONITOR_LOG_DIR``. Best-effort: returns
    ``None`` if the directory can't be created (caller falls back to a console).
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))  # yb_analysis/gui/ -> project root
        log_dir = os.environ.get('YB_MONITOR_LOG_DIR') or os.path.join(
            project_root, 'log', 'monitor_log')
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(
            log_dir, 'run_monitor_%s.log' % time.strftime('%Y%m%d_%H%M%S'))
    except OSError:
        return None

# Human-facing labels for the backend toggle.
_BACKEND_LABELS = {'matlab': 'MATLAB', 'pyctrl': 'pyctrl'}

_FONT = ('Segoe UI', 10)
_FONT_SM = ('Segoe UI', 9)
_FONT_TITLE = ('Segoe UI', 14, 'bold')


_STATUS = {0: 'Stopped', 1: 'Running', 2: 'Paused', 3: 'Unknown'}
# Shown from the moment Abort is pressed until the scan actually stops. Abort is
# per-SEQUENCE: the in-flight FPGA shot can't be interrupted, so a stop lands at
# the next sequence boundary (~0.1-1 s). Surfacing this avoids the "I pressed
# Abort and nothing happened" confusion. (Phase-5 latency/UX must-fix #6.)
_ABORT_HINT = 'Aborting… (stops after current shot)'
_STATUS_COLORS = {
    'Idle': '#555555', 'Idle (dummy off)': '#888888',
    'Idle (default)': '#555555', 'Idle (last seq)': '#0a5d8a',
    'Idle (last fallback)': '#cc6600',
    'Running': '#006600', 'Paused': '#cc6600',
    'Pausing...': '#cc6600', 'Stopped': '#aa0000',
    _ABORT_HINT: '#aa0000',
}

# Dummy-mode radio values must match the strings ExptServer.dummy_mode accepts.
_DUMMY_MODES = ('off', 'default', 'last')


def _release_camera_before_teardown(client, backend, *, timeout_s=5.0,
                                    sleep=time.sleep, clock=time.monotonic, log=logger):
    """Release the camera GRACEFULLY before the backend is killed, so a restart /
    backend-switch / window-close can't wedge the Orca. (Testable core of
    ``ControlPanel._release_camera_for_teardown``; pure + injectable, no Tk needed.)

    The wedge risk is pyctrl-specific: that backend holds the single DCAM handle via
    ``pylablib`` and reflects it in ``camera_status`` (the MATLAB backend owns IMAQ and is
    released by ``RunnerLauncher``'s own dcamtray teardown). A fire-and-forget ``camera_close``
    does NOT guarantee the handle is released before the kill (``runner.stop()`` runs in
    ``run_monitor``'s atexit, AFTER ``_on_close``) -- and during a running scan the consume loop
    is busy inside a job and won't service ``camera_close`` until the job ends. So, for pyctrl:
      1. ``abort_seq`` -- free the run loop to service the camera command (the abort flag dies
         with the process on the imminent restart, so it can't poison the next run --
         cf. clear-at-job-start / bug-runjob-stale-abortrunseq);
      2. ``camera_close`` -- ask the backend to ``camera.close()`` the DCAM handle;
      3. poll ``camera_status`` until ``connected`` is False (bounded by ``timeout_s``).
    Best-effort: never raises, never blocks past ``timeout_s``. MATLAB is unchanged -- the
    unconditional ``camera_close`` is sent, then IMAQ release is left to ``RunnerLauncher``.
    Returns True iff the camera is confirmed released (pyctrl) / the close was sent (MATLAB);
    False on a client error or the timeout."""
    is_pyctrl = (backend == 'pyctrl')
    if is_pyctrl:
        try:
            client.abort_seq()                # free the run loop to service camera_close
        except Exception:
            pass
    try:
        client.camera_close()                 # both backends (unchanged for MATLAB)
    except Exception:
        return False
    if not is_pyctrl:
        return True                           # MATLAB: IMAQ released by RunnerLauncher
    deadline = clock() + timeout_s
    while clock() < deadline:
        try:
            st = client.camera_status()
        except Exception:
            return False                      # can't query -> stop waiting (best-effort)
        if not (isinstance(st, dict) and st.get('connected')):
            log.info('camera released cleanly before teardown')
            return True
        sleep(0.15)
    log.warning('camera still connected after %.1fs -- proceeding with teardown anyway', timeout_s)
    return False


class ControlPanel(tk.Tk):

    def __init__(self, zmq_client, dashboard=None, init_dir=None, init_status='',
                 backend='matlab', backend_runner=None):
        super().__init__()
        self.title('Yb Experiment Control')
        self.geometry('640x720')
        self.minsize(520, 560)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._client = zmq_client
        self._dashboard = dashboard
        self._init_dir = init_dir
        self._init_status = init_status
        # The sequence backend this monitor process was launched against
        # ('matlab' | 'pyctrl'). The backend toggle relaunches run_monitor
        # with a different value (a Restart-All-style handoff), so within one
        # process this never changes — it's display + the toggle's "from".
        self._backend = backend if backend in VALID_BACKENDS else 'matlab'
        # The launcher owning the current backend process (RunnerLauncher or
        # PyctrlLauncher). Needed so a backend SWITCH can force the old backend
        # to actually die on exit even under --reuse-runner (else the new one
        # adopts the stale, wrong-backend server). None when --no-runner.
        self._backend_runner = backend_runner
        self._cur_scan_id = 0
        self._cur_seq_id = 0
        self._refresh_ms = 500
        self._running = True
        # True from an Abort press until the scan actually stops; drives the
        # _ABORT_HINT status so the per-sequence abort latency is visible.
        self._abort_pending = False
        # Mirrors the server's __dummy_mode. Used by the status-label rendering
        # to distinguish 'Idle (default)' from 'Idle (last seq)' etc.
        self._dummy_mode = 'last'
        # Last value of last_seq_status the GUI has seen — drives the cached-
        # seq label and the enable state of the "Last seq" radio button.
        self._last_seq_meta = {
            'available': False, 'name': '', 'file_id': '',
            'fallback_active': False,
        }
        # Last per-shot health rollup from the (pyctrl) backend, refreshed in
        # the background poll and republished for the web dashboard's "shots
        # failing" banner. None until first fetched / when the backend lacks
        # the verb (MATLAB).
        self._shot_health = None
        # Background (calibration) lane mirror (from last_seq_status): the global enable
        # toggle, whether a background calibration is currently running, and its name -- so the
        # status label can show "Idle (background calibration: <name>)" and the toggle reflects
        # the server. Defaults assume enabled (the server default) until first polled.
        self._bg_state = {'enabled': True, 'running': False, 'name': ''}
        # Reference to the last real-scan DataManager. Dummy frames borrow
        # its plot-data dict so the dashboard can show the live frame without
        # mutating the real DM's internal state or its save buffers.
        self._last_real_dm = None

        style = ttk.Style(self)
        style.configure('Abort.TButton', foreground='red',
                        font=('Segoe UI', 10, 'bold'))

        self._build_ui()

        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()
        self._poll_status()
        self._poll_web_control_loop()
        self._tick_alive()

    def _tick_alive(self):
        self.after(200, self._tick_alive)

    def _poll_web_control_loop(self):
        """Drain web-dashboard control commands on a fast, dedicated cadence
        (decoupled from the 1 Hz status poll). This bounds the latency of a
        REMOTE Start/Pause/Abort — which under the pyctrl backend route through
        the spool to this process's ZMQ client — to ~300 ms + one ZMQ
        round-trip, rather than up to ~1 s if it rode the status poll. (The
        local Tkinter Abort button is direct ZMQ and never waits on this.)"""
        if not self._running:
            return
        self._poll_web_control()
        self.after(300, self._poll_web_control_loop)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        # ---- Top bar: status + buttons ----
        top = ttk.Frame(self)
        top.pack(fill='x', padx=10, pady=(8, 4))

        self._lbl_status = ttk.Label(top, text='Status: Unknown',
                                     font=_FONT_TITLE)
        self._lbl_status.pack(side='left')

        ttk.Button(top, text='ABORT', command=self._on_abort,
                   style='Abort.TButton').pack(side='right', padx=(4, 0))
        ttk.Button(top, text='Start', command=self._on_start).pack(
            side='right', padx=4)
        ttk.Button(top, text='Pause', command=self._on_pause).pack(
            side='right', padx=4)
        ttk.Button(top, text='Restart Dash', command=self._on_restart_dash).pack(
            side='right', padx=4)
        ttk.Button(top, text='Restart All',
                   command=self._on_restart_all).pack(
            side='right', padx=4)

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=10, pady=2)

        # ---- Backend toggle: MATLAB <-> pyctrl ----
        # Switching tears down this monitor + its backend and relaunches on
        # the chosen backend (a Restart-All handoff). Selecting the backend
        # already active is a no-op. Gated by a confirmation dialog because it
        # stops any running scan.
        backend_frame = ttk.LabelFrame(self, text='Sequence backend')
        backend_frame.pack(fill='x', padx=10, pady=(2, 4))
        brow = ttk.Frame(backend_frame)
        brow.pack(fill='x', padx=6, pady=(4, 4))
        self._backend_var = tk.StringVar(value=self._backend)
        for be in VALID_BACKENDS:
            ttk.Radiobutton(
                brow, text=_BACKEND_LABELS.get(be, be),
                variable=self._backend_var, value=be,
                command=self._on_backend_radio).pack(side='left', padx=(0, 12))
        self._lbl_backend = ttk.Label(
            brow, text=f'(active: {_BACKEND_LABELS.get(self._backend, self._backend)})',
            font=_FONT_SM, foreground='#0a5d8a')
        self._lbl_backend.pack(side='left', padx=(8, 0))

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=10, pady=2)

        # Dummy keep-alive mode selector — three radios:
        #   off     -> runner pauses between jobs
        #   default -> runner runs DummySeq (the canonical no-imaging filler)
        #   last    -> runner replays the last successful real seq; frames
        #              flow to the dashboard but scan_id<0 routes them around
        #              the disk-save path (see _process_once).
        # The "Last seq" radio shows what's cached and an amber fallback
        # indicator when the runner had to drop back to default.
        dummy_frame = ttk.LabelFrame(self, text='Dummy')
        dummy_frame.pack(fill='x', padx=10, pady=(2, 4))
        radios = ttk.Frame(dummy_frame)
        radios.pack(fill='x', padx=6, pady=(4, 0))
        self._dummy_mode_var = tk.StringVar(value='last')
        for mode, label in (('off', 'Off'),
                             ('default', 'Default dummy'),
                             ('last', 'Last seq')):
            rb = ttk.Radiobutton(
                radios, text=label, variable=self._dummy_mode_var,
                value=mode, command=self._on_dummy_mode_changed)
            rb.pack(side='left', padx=(0, 12))
            if mode == 'last':
                self._rb_last = rb
        self._lbl_last_seq = ttk.Label(
            dummy_frame, text='Last: --', font=_FONT_SM, foreground='#555555')
        self._lbl_last_seq.pack(anchor='w', padx=6, pady=(2, 4))
        # Background (calibration) lane: a global toggle to run/halt low-priority calibration
        # scans (they run only when no foreground scan is running/queued, and yield instantly
        # when one is queued). Halting leaves queued calibrations in place -- they resume when
        # re-enabled. The label shows which calibration is currently running.
        bg_row = ttk.Frame(dummy_frame)
        bg_row.pack(fill='x', padx=6, pady=(0, 4))
        self._bg_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bg_row, text='Run background calibrations',
            variable=self._bg_enabled_var,
            command=self._on_background_toggle).pack(side='left')
        self._lbl_bg = ttk.Label(bg_row, text='', font=_FONT_SM, foreground='#0a7a5d')
        self._lbl_bg.pack(side='left', padx=(8, 0))
        # Sync UI from the server on startup so a persistent runner that
        # was last left in a particular mode is reflected in the UI.
        threading.Thread(target=self._load_dummy_state, daemon=True).start()

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=10, pady=2)

        # ---- Middle: camera (left) + scan info (right) ----
        mid = ttk.Frame(self)
        mid.pack(fill='x', padx=10, pady=(2, 4))
        mid.columnconfigure(1, weight=1)

        from yb_analysis.gui.camera_pane import CameraPane
        self._camera_pane = CameraPane(mid, self._client, refresh_ms=2000)
        self._camera_pane.grid(row=0, column=0, sticky='nsew', padx=(0, 4))

        info = ttk.LabelFrame(mid, text='Current scan')
        info.grid(row=0, column=1, sticky='nsew', padx=(4, 0))
        gi = ttk.Frame(info)
        gi.pack(fill='both', expand=True, padx=6, pady=4)

        for r, (label, attr) in enumerate([('Scan:', '_lbl_scan'),
                                            ('Seq:', '_lbl_seq')]):
            ttk.Label(gi, text=label, font=_FONT).grid(row=r, column=0,
                                                        sticky='w', padx=(0, 4))
            lbl = ttk.Label(gi, text='--', font=_FONT)
            lbl.grid(row=r, column=1, sticky='w')
            setattr(self, attr, lbl)

        ttk.Label(gi, text='File:', font=_FONT_SM).grid(
            row=2, column=0, sticky='nw', padx=(0, 4))
        self._lbl_file = ttk.Label(gi, text='--', font=_FONT_SM,
                                   wraplength=0)
        self._lbl_file.grid(row=2, column=1, sticky='w')

        self._lbl_save_err = ttk.Label(gi, text='', font=_FONT_SM,
                                        foreground='red', wraplength=190)
        self._lbl_save_err.grid(row=3, column=0, columnspan=2, sticky='w')
        self._lbl_save_err.grid_remove()

        rf = ttk.Frame(gi)
        rf.grid(row=4, column=0, columnspan=2, sticky='w', pady=(4, 0))
        ttk.Label(rf, text='Refresh (s):', font=_FONT_SM).pack(side='left')
        self._rate_entry = ttk.Entry(rf, width=4, font=_FONT_SM)
        # Display in seconds; %g shows "0.5" / "2" without trailing zeros.
        self._rate_entry.insert(0, '%g' % (self._refresh_ms / 1000))
        self._rate_entry.pack(side='left', padx=4)
        self._rate_entry.bind('<Return>', self._on_rate)

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=10, pady=2)

        # ---- Init folder selector ----
        from yb_analysis.gui.init_pane import InitPane
        self._init_pane = InitPane(
            self, on_change=self._on_init_loaded,
            init_dir=self._init_dir, init_status=self._init_status,
        )
        self._init_pane.pack(fill='x', padx=10, pady=(2, 4))

        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=10, pady=2)

        # ---- Bottom: queue pane ----
        from yb_analysis.gui.queue_pane import QueuePane
        self._queue_pane = QueuePane(self, self._client,
                                     dashboard=self._dashboard, refresh_ms=1000)
        self._queue_pane.pack(fill='both', expand=True, padx=10, pady=(2, 8))

    # -------------------------------------------------------------- Actions

    # Control buttons are backend-aware: for 'matlab' they write the MemoryMap
    # (with the existing ZMQ fallback when the file is absent); for 'pyctrl'
    # _mmap_open(self._backend) returns None unconditionally, so they always
    # take the ZMQ verb path and never touch a (possibly stale) local memmap.
    def _on_pause(self):
        mm = _mmap_open(self._backend)
        if mm:
            _mmap_write_double(mm, _OFF_PAUSE, 1.0)
            mm.close()
        else:
            self._client.pause_seq()

    def _on_start(self):
        mm = _mmap_open(self._backend)
        if mm:
            _mmap_write_double(mm, _OFF_PAUSE, 0.0)
            mm.close()
        else:
            self._client.start_seq()

    def _on_abort(self):
        mm = _mmap_open(self._backend)
        if mm:
            _mmap_write_double(mm, _OFF_ABORT, 1.0)
            _mmap_write_double(mm, _OFF_PAUSE, 0.0)
            mm.close()
        else:
            self._client.abort_seq()
        self._note_abort_requested()

    def _note_abort_requested(self):
        """Mark an abort in flight and show the latency hint IMMEDIATELY (don't wait for the
        next ~1 s status poll). Abort lands at the next sequence boundary (the in-flight FPGA
        shot can't be interrupted); the hint clears in _poll_status once the scan is Stopped/Idle.
        Called from both the local Abort button and the web-spooled abort."""
        self._abort_pending = True
        try:
            self._lbl_status.config(text=f'Status: {_ABORT_HINT}',
                                    foreground=_STATUS_COLORS[_ABORT_HINT])
        except tk.TclError:
            pass
        self._publish_ctrl_status(_ABORT_HINT)

    def _apply_abort_hint(self, status):
        """While an abort is in flight, show _ABORT_HINT until the scan actually stops.

        The abort has LANDED once the poll reports a terminal/idle state (Stopped, or any
        ``Idle …`` keep-alive badge); then we clear the pending flag and let the real status
        through. Until then a still-Running/Pausing/Paused poll is masked by the hint so the
        user isn't told "Running" right after pressing Abort. A Paused scan that is then aborted
        still shows the hint until it reaches Stopped."""
        if not self._abort_pending:
            return status
        if status == 'Stopped' or status.startswith('Idle'):
            self._abort_pending = False
            return status
        return _ABORT_HINT

    def _on_restart_dash(self):
        """Kill and immediately respawn the Dash subprocess.

        Picks up code changes in dashboard.py without restarting the main GUI
        (so the camera connection stays up).
        """
        if not self._dashboard:
            return
        try:
            self._dashboard.restart()
            logger.info('Dashboard subprocess restarted')
        except Exception:
            logger.error('Dashboard restart failed:\n%s', traceback.format_exc())

    @staticmethod
    def _argv_with_backend(argv, backend_override):
        """Return ``argv`` with any existing ``--backend`` (and its value)
        removed; if ``backend_override`` is given, append ``--backend <v>``.
        Handles both ``--backend X`` and ``--backend=X`` forms. With
        backend_override=None the args are passed through untouched (plain
        Restart All keeps the current backend + reuse/no-runner flags).

        On a switch (``backend_override`` set) we also strip ``--reuse-runner``
        and ``--no-runner``: the chosen backend must be spawned FRESH, never
        adopted from a still-running server (ping can't tell the two backends
        apart) and never skipped."""
        if backend_override is None:
            return list(argv)
        # Flags to drop on a switch: --backend (takes a value) plus the
        # store_true flags that would prevent a fresh spawn of the target.
        drop_with_value = {'--backend'}
        drop_flags = {'--reuse-runner', '--no-runner'}
        out = []
        skip_next = False
        for tok in argv:
            if skip_next:
                skip_next = False
                continue
            if tok in drop_with_value:
                skip_next = True  # drop the following value token too
                continue
            if tok.startswith('--backend='):
                continue
            if tok in drop_flags:
                continue
            out.append(tok)
        out += ['--backend', backend_override]
        return out

    def _on_backend_radio(self):
        """Backend radio clicked. Route to the switch handler; the handler
        resets the radio if the user cancels or picks the active backend."""
        self._on_switch_backend(self._backend_var.get())

    def _on_switch_backend(self, target, confirmed=False):
        """Switch the live sequence backend (MATLAB <-> pyctrl).

        Implemented as a Restart-All handoff with ``--backend <target>``
        injected: this process tears down its current backend (releasing the
        DCAM camera handle + ZMQ port) and a fresh run_monitor comes up on the
        chosen backend. Selecting the active backend is a no-op.

        ``confirmed=True`` skips the Tk dialog (used by the web dashboard,
        which has its own hold-to-confirm + confirm-token interlock).
        """
        import tkinter.messagebox as mb
        if target not in VALID_BACKENDS:
            self._backend_var.set(self._backend)
            return
        if target == self._backend:
            return  # already active — radio is just reasserting current state
        if not confirmed and not mb.askokcancel(
                'Switch backend',
                f'Switch the sequence backend from '
                f'{_BACKEND_LABELS.get(self._backend, self._backend)} to '
                f'{_BACKEND_LABELS.get(target, target)}?\n\n'
                f'This stops the current backend and any running scan, then '
                f'relaunches the monitor on {_BACKEND_LABELS.get(target, target)}. '
                f'The control window will reappear once it is up.'):
            # User cancelled — put the radio back on the active backend.
            self._backend_var.set(self._backend)
            return
        logger.info('Backend switch requested: %s -> %s', self._backend, target)
        # Force the current backend to actually be torn down on exit (even if
        # we started in --reuse-runner mode): a switch must free the camera
        # handle + port so the new backend can bind, and must not leave a
        # stale server the new launcher would wrongly adopt.
        if self._backend_runner is not None:
            try:
                self._backend_runner.take_ownership()
            except Exception as e:
                logger.warning('take_ownership failed (continuing): %s', e)
        self._on_restart_all(confirmed=True, backend_override=target)

    def _on_restart_all(self, confirmed=False, backend_override=None):
        """Spawn a fresh run_monitor process and exit this one.

        Use after editing files that this process imports at startup
        (run_monitor.py, control_panel.py, zmq_client.py, slm_proxy.py).
        For changes to dashboard.py + its downstream imports, use
        "Restart Dash" instead -- it's faster and doesn't drop the
        camera connection.

        Confirmation: a tk dialog gates the action so an accidental
        click doesn't tear down a running scan. ``confirmed=True`` skips
        the dialog (used by the web dashboard, which gates the action
        behind its own hold-to-confirm + confirm-token interlock).

        ``backend_override`` (used by the backend toggle): relaunch on a
        different sequence backend by rewriting the ``--backend`` argument.
        """
        import sys
        import subprocess
        import tkinter.messagebox as mb

        if not confirmed and not mb.askokcancel(
                'Restart All',
                'Shut down this process (including the sequence backend) '
                'and respawn a fresh run_monitor? No terminal opens -- output '
                'goes to log/monitor_log/. The control window will reappear '
                'once the new process is up.'):
            return

        # Reconstruct the command line: prefer the module form so this
        # works regardless of whether the user launched via `python -m
        # yb_analysis.scripts.run_monitor` or via a script wrapper.
        passthru = self._argv_with_backend(sys.argv[1:], backend_override)
        cmd = [sys.executable, '-m', 'yb_analysis.scripts.run_monitor'] + passthru
        logger.info('Restart All: spawning %s', cmd)

        # Tell the new process to wait until WE die before binding any
        # ports. Without this, the new process and the still-shutting-
        # down old process race for port 8050 / 1408 -- whichever the
        # OS hands out the socket to wins, and the loser logs a confusing
        # "address in use" error. The env var carries our PID; the new
        # process polls it and blocks until it's gone (max 30 s).
        env = os.environ.copy()
        env['YB_WAIT_FOR_PID'] = str(os.getpid())

        # Spawn the new process WINDOWLESS: no terminal pops up on restart.
        # Its stdout/stderr (run_monitor's own logging + the backend it spawns,
        # whose stdio it inherits) are redirected to a timestamped logfile under
        # log/monitor_log/ -- the visible-terminal replacement. The pyctrl
        # backend additionally writes its own organized log/pyctrl_log/ files.
        # If the logfile can't be opened we fall back to a visible console so
        # output is never silently lost.
        stdio = None
        log_path = _monitor_logfile()
        if log_path is not None:
            try:
                stdio = open(log_path, 'a', buffering=1, encoding='utf-8')
                logger.info('Restart All: redirecting new process output to %s',
                            log_path)
            except OSError:
                stdio = None
        try:
            if os.name == 'nt':
                CREATE_NO_WINDOW = 0x08000000
                CREATE_NEW_CONSOLE = 0x00000010
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                if stdio is not None:
                    subprocess.Popen(
                        cmd, env=env,
                        creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                        stdout=stdio, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, close_fds=True)
                else:
                    # No logfile -> keep output visible in a new console.
                    subprocess.Popen(cmd, env=env,
                                     creationflags=CREATE_NEW_CONSOLE,
                                     close_fds=True)
            else:
                subprocess.Popen(cmd, env=env,
                                 start_new_session=True,
                                 stdout=stdio, stderr=(subprocess.STDOUT
                                                       if stdio else None),
                                 stdin=subprocess.DEVNULL,
                                 close_fds=True)
        except Exception:
            logger.error('Restart All: spawn failed:\n%s',
                         traceback.format_exc())
            mb.showerror('Restart All', 'Failed to spawn the new process. '
                         'Check the log; you may need to restart manually.')
            return
        finally:
            # The child inherited its own copy of the handle; close ours so the
            # file isn't held open by this (about-to-exit) process.
            if stdio is not None:
                stdio.close()

        # Brief pause so the new console window appears before this one
        # disappears; the new process is sleeping on YB_WAIT_FOR_PID so
        # there's no port race even if we're slow to die.
        time.sleep(0.5)

        # Now cleanly tear down this process. _on_close handles the
        # dashboard, the runner, the SLM proxy, and any other cleanup
        # the main thread registered. The new process resumes once
        # this PID is gone.
        logger.info('Restart All: handing off to new process; closing self')
        try:
            self._on_close()
        except Exception:
            # Last resort: nuke the process. The new run_monitor's
            # kill_port(8050) will scrub any orphaned dashboard.
            os._exit(0)

    def _on_shutdown(self, confirmed=False):
        """Thoroughly tear down and EXIT run_monitor -- no respawn.

        Same graceful teardown as closing the window (``_on_close``: camera
        release -> ZMQ client cleanup -> dashboard close -> destroy), after
        which ``run_monitor``'s ``mainloop()`` returns and its ``atexit``
        ``_cleanup`` stops the SLM proxy + the sequence backend and force-frees
        ports 8050/1408. Unlike "Restart All", nothing is spawned to take over.

        ``confirmed=True`` skips the Tk dialog (the web dashboard gates it
        behind its own hold-to-confirm + confirm-token interlock).
        """
        import tkinter.messagebox as mb
        if not confirmed and not mb.askokcancel(
                'Shutdown',
                'Shut down this process AND the sequence backend, then exit '
                '(no restart)? You will need to relaunch run_monitor '
                '(desktop shortcut or terminal) to bring it back.'):
            return
        logger.info('Shutdown: full teardown, no respawn')
        try:
            self._on_close()
        except Exception:
            # Cleanup raised mid-teardown. End the mainloop so run_monitor's
            # atexit _cleanup still runs (stops the backend, frees ports); only
            # if even that fails do we hard-exit as a last resort.
            logger.error('Shutdown: _on_close failed:\n%s',
                         traceback.format_exc())
            try:
                self.destroy()
            except Exception:
                os._exit(0)

    def _on_init_loaded(self, data):
        if self._dashboard and data is not None:
            self._dashboard.update(data)

    def _on_rate(self, _=None):
        try:
            v = float(self._rate_entry.get())
            self._refresh_ms = max(500, int(round(v * 1000)))
        except ValueError:
            pass

    def _on_dummy_mode_changed(self):
        mode = self._dummy_mode_var.get()
        if mode not in _DUMMY_MODES:
            mode = 'default'
            self._dummy_mode_var.set(mode)
        self._dummy_mode = mode
        # ZMQ REQ/REP off the UI thread to avoid freezing the radio click.
        threading.Thread(
            target=self._do_push_dummy_mode, args=(mode,), daemon=True).start()

    def _do_push_dummy_mode(self, mode):
        try:
            self._client.set_dummy_mode(mode)
        except Exception as e:
            logger.warning('set_dummy_mode(%r) failed: %s', mode, e)

    def _on_background_toggle(self):
        enabled = bool(self._bg_enabled_var.get())
        self._bg_state['enabled'] = enabled
        # ZMQ REQ/REP off the UI thread so the checkbox click never blocks the UI.
        threading.Thread(
            target=self._do_push_background_enabled, args=(enabled,), daemon=True).start()

    def _do_push_background_enabled(self, enabled):
        try:
            self._client.set_background_enabled(enabled)
        except Exception as e:
            logger.warning('set_background_enabled(%r) failed: %s', enabled, e)

    def _load_dummy_state(self):
        """Read mode + last-seq metadata from the server on startup so the
        UI reflects whatever the persistent runner already had. Runs on a
        worker; main-thread Tk update via after(0)."""
        mode = None
        try:
            mode = self._client.get_dummy_mode()
        except Exception as e:
            logger.warning('get_dummy_mode failed: %s', e)
        meta = None
        try:
            meta = self._client.last_seq_status()
        except Exception as e:
            logger.warning('last_seq_status failed: %s', e)
        self.after(0, self._apply_dummy_state_from_server, mode, meta)

    def _apply_dummy_state_from_server(self, mode, meta):
        if mode in _DUMMY_MODES:
            self._dummy_mode_var.set(mode)
            self._dummy_mode = mode
        self._update_last_seq_label(meta)

    def _update_last_seq_label(self, meta):
        """Refresh the cached-seq label. The label answers two questions:
        what is currently being replayed, and what's in the cache.

        Layout (depends on (mode, available)):
            mode='last',  cache empty   -> "Running default (no last seq cached)"  amber
            mode='last',  cache present -> "Replaying: name (fid)"                 blue
            mode!='last', cache empty   -> "Cached: --"                            gray
            mode!='last', cache present -> "Cached: name (fid)"                    gray
        """
        if not isinstance(meta, dict):
            return
        self._last_seq_meta = {
            'available': bool(meta.get('available')),
            'name': str(meta.get('name', '') or ''),
            'file_id': str(meta.get('file_id', '') or ''),
            'fallback_active': bool(meta.get('fallback_active')),
        }
        # Background (calibration) lane mirror -- rides on the same last_seq_status payload
        # (absent on an older/MATLAB backend -> defaults). Reflect the toggle + running label.
        self._bg_state = {
            'enabled': bool(meta.get('background_enabled', True)),
            'running': bool(meta.get('background_running')),
            'name': str(meta.get('background_name', '') or ''),
        }
        try:
            if self._bg_enabled_var.get() != self._bg_state['enabled']:
                self._bg_enabled_var.set(self._bg_state['enabled'])
            if self._bg_state['running']:
                self._lbl_bg.config(
                    text='running: %s' % (self._bg_state['name'] or '(calibration)'))
            else:
                self._lbl_bg.config(text='')
        except (tk.TclError, AttributeError):
            pass
        available = self._last_seq_meta['available']
        name = self._last_seq_meta['name'] or '(unnamed)'
        fid = self._last_seq_meta['file_id']
        suffix = f' ({fid})' if fid else ''
        if self._dummy_mode == 'last':
            if available:
                txt = f'Replaying: {name}{suffix}'
                color = '#0a5d8a'
            else:
                txt = 'Running default (no last seq cached)'
                color = '#cc6600'
        else:
            if available:
                txt = f'Cached: {name}{suffix}'
            else:
                txt = 'Cached: --'
            color = '#888888'
        try:
            self._lbl_last_seq.config(text=txt, foreground=color)
        except tk.TclError:
            pass

    # --------------------------------------------------------- Status poll

    def _poll_status(self):
        if not self._running:
            return
        try:
            mm = _mmap_open(self._backend)
            if mm:
                abort = _mmap_read_double(mm, _OFF_ABORT)
                pause = _mmap_read_double(mm, _OFF_PAUSE)
                is_paused = _mmap_read_double(mm, _OFF_ISPAUSED)
                scan_complete = _mmap_read_double(mm, _OFF_SCAN_COMPLETE)
                dummy = _mmap_read_double(mm, _OFF_DUMMY_RUNNING)
                num_per_group = _mmap_read_double(mm, _OFF_NUM_PER_GROUP)
                mm.close()

                # NumPerGroup is positive iff runJob is currently in flight
                # (set at line 275 of SequenceRunner.m, zeroed at line 364).
                # We use it as a tie-breaker so an Idle badge can never paper
                # over a real job — DummyRunning can lag a poll if the
                # MATLAB write hasn't been picked up yet.
                if dummy > 0 and num_per_group <= 0:
                    if self._dummy_mode == 'off':
                        status = 'Idle (dummy off)'
                    elif self._dummy_mode == 'last':
                        if self._last_seq_meta.get('available'):
                            status = 'Idle (last seq)'
                        else:
                            status = 'Idle (last fallback)'
                    else:
                        status = 'Idle (default)'
                elif is_paused > 0:
                    status = 'Paused'
                elif pause > 0:
                    status = 'Pausing...'
                elif abort > 0:
                    status = 'Stopped'
                elif scan_complete > 0:
                    status = 'Stopped'
                else:
                    status = 'Running'
                status = self._apply_abort_hint(status)
                self._lbl_status.config(
                    text=f'Status: {status}',
                    foreground=_STATUS_COLORS.get(status, '#000000'))
                self._publish_ctrl_status(status)
            else:
                # pyctrl backend: get_status() is a BLOCKING ZMQ round-trip. If
                # the backend is dead it blocks for the full client timeout
                # (~30 s). Running it on the Tk main thread would freeze the
                # WHOLE UI for that whole time -- buttons AND _poll_web_control_loop
                # (the web-command drain), so the operator could not even click
                # "Restart All" to recover (exactly the wedge that stranded a dead
                # backend). Offload to a daemon thread and marshal the label
                # update back via after(0, ...). The busy guard means a slow/hung
                # call can't pile up one worker thread per 1 s tick.
                if not getattr(self, '_status_poll_busy', False):
                    self._status_poll_busy = True
                    threading.Thread(target=self._poll_status_pyctrl_async,
                                     daemon=True).start()
        except Exception:
            pass
        # Refresh the cached-last-seq label in the background. Throttled to
        # every other tick (~2 s) so the runner isn't hammered with REQ
        # round-trips when the cache is stable.
        self._last_seq_poll_tick = getattr(self, '_last_seq_poll_tick', 0) + 1
        if self._last_seq_poll_tick % 2 == 0:
            threading.Thread(target=self._refresh_last_seq_async,
                             daemon=True).start()
        self.after(1000, self._poll_status)

    def _poll_status_pyctrl_async(self):
        """Off-main-thread status fetch for the pyctrl backend (see _poll_status).

        Runs the blocking get_status() ZMQ call in a daemon thread so the Tk
        main loop stays responsive even when the backend is unreachable, then
        marshals the label/web-mirror update back onto the main thread. A dead
        backend simply shows 'Unknown' and the rest of the UI keeps working --
        crucially the web/local Restart All, which needs no backend at all."""
        try:
            s = _STATUS.get(self._client.get_status(), 'Unknown')
        except Exception:
            s = 'Unknown'   # backend unreachable -> keep the UI alive
        finally:
            self._status_poll_busy = False

        def _apply():
            if not self._running:
                return
            s2 = self._apply_abort_hint(s)
            color = _STATUS_COLORS.get(s2, '#000000')
            # A running background calibration reads as foreground-idle to the operator: the
            # coarse get_status flickers Running (active shot) / Stopped (between scans) as the
            # calibration cycles, so override to a stable "background" label. An explicit Pause
            # or a pending Abort still wins (the user is acting on whatever is running).
            if (self._bg_state.get('running') and not self._abort_pending
                    and s2 not in ('Paused', _ABORT_HINT)):
                nm = self._bg_state.get('name') or ''
                s2 = 'Idle (background: %s)' % nm if nm else 'Idle (background calibration)'
                color = '#0a7a5d'
            self._lbl_status.config(text=f'Status: {s2}', foreground=color)
            self._publish_ctrl_status(s2)
        try:
            self.after(0, _apply)
        except Exception:
            pass

    def _refresh_last_seq_async(self):
        # Per-shot health for the web "shots failing" banner. Cheap; pulled on
        # the same throttled background tick as last_seq. None (MATLAB backend /
        # wire error) just leaves the banner cleared.
        try:
            self._shot_health = self._client.shot_health()
        except Exception:
            self._shot_health = None
        try:
            meta = self._client.last_seq_status()
        except Exception:
            return
        if meta is not None:
            self.after(0, self._update_last_seq_label, meta)

    # ----------------------------------------- Web-dashboard control mirror

    def _publish_ctrl_status(self, status):
        """Publish full-fidelity control state for the web sidebar (the
        snapshot's ``_dummy_mode`` is only a bool). Best-effort."""
        try:
            _web_control.publish_status({
                'dummy_mode': self._dummy_mode,
                'last_seq': dict(self._last_seq_meta),
                'background': dict(self._bg_state),
                'scan_id': self._cur_scan_id,
                'seq_id': self._cur_seq_id,
                'state': status,
                'backend': self._backend,
                'shot_health': (dict(self._shot_health)
                                if isinstance(self._shot_health, dict) else None),
            })
        except Exception:
            pass

    def _poll_web_control(self):
        """Execute control commands the web dashboard spooled for the main
        process (Phase 5.5 Track A). Under the MATLAB backend, the dashboard
        writes Pause/Start/Abort straight to the MemoryMap; under pyctrl there
        is NO local memmap, so the dashboard spools 'pause'/'start'/'abort'
        here and we issue the ZMQ verb through this process's (sole) client.
        The other commands always need *this* process's state (ZMQ client,
        calibration loader, supervisor). Each is wrapped so a bad command can't
        break the loop."""
        try:
            cmds = _web_control.drain()
        except Exception:
            return
        for rec in cmds:
            cmd = (rec or {}).get('cmd')
            try:
                if cmd == 'pause':
                    self._client.pause_seq()
                elif cmd == 'start':
                    self._client.start_seq()
                elif cmd == 'abort':
                    # Safety-critical: log so the abort path is auditable.
                    logger.info('Web abort -> ZMQ abort_seq (backend=%s)',
                                self._backend)
                    self._client.abort_seq()
                    self._note_abort_requested()    # surface the "stops after current shot" hint
                elif cmd == 'dummy_mode':
                    mode = rec.get('mode')
                    if mode in _DUMMY_MODES:
                        self._dummy_mode_var.set(mode)
                        self._on_dummy_mode_changed()
                elif cmd == 'background_enabled':
                    enabled = bool(rec.get('enabled'))
                    self._bg_enabled_var.set(enabled)
                    self._on_background_toggle()
                elif cmd == 'init_dir':
                    path = rec.get('path')
                    if path:
                        self._init_pane._load_in_background(path)
                elif cmd == 'restart_dash':
                    self._on_restart_dash()
                elif cmd == 'restart_all':
                    # Web side already gated this behind a confirm token;
                    # skip the local Tk confirmation dialog.
                    self._on_restart_all(confirmed=True)
                elif cmd == 'shutdown':
                    # Web side gated this behind a confirm token; skip the
                    # local Tk dialog. Full teardown, no respawn.
                    self._on_shutdown(confirmed=True)
                elif cmd == 'set_backend':
                    # Web side gated this behind a confirm token; skip the
                    # local Tk dialog. Reflect the radio first so the UI is
                    # consistent during the brief teardown.
                    target = rec.get('target')
                    if target in VALID_BACKENDS:
                        try:
                            self._backend_var.set(target)
                        except Exception:
                            pass
                        self._on_switch_backend(target, confirmed=True)
                elif cmd in ('camera_connect', 'camera_apply',
                             'camera_disconnect'):
                    # Dispatch to CameraPane, which owns the ZMQ client and
                    # the expConfig.m persistence. action = the bit after
                    # 'camera_'.
                    action = cmd.split('_', 1)[1]
                    self._camera_pane.apply_web_command(
                        action, roi=rec.get('roi'),
                        exposure=rec.get('exposure'))
                else:
                    logger.warning('Unknown web-control command: %r', cmd)
            except Exception:
                logger.error('web-control %r failed:\n%s', cmd,
                             traceback.format_exc())

    # ------------------------------------------------ Background processing

    def _process_loop(self):
        while self._running:
            try:
                self._process_once()
            except Exception:
                logger.error('Process error:\n%s', traceback.format_exc())
            time.sleep(self._refresh_ms / 1000)

    def _process_once(self):
        info = self._client.grab_imgs()
        if len(info['scan_ids']) == 0:
            return
        valid_mask = info['scan_ids'] != 0
        if not np.any(valid_mask):
            return
        valid_idx = np.where(valid_mask)[0]
        imgs = [info['imgs'][i] for i in valid_idx]
        scan_ids = info['scan_ids'][valid_mask]
        seq_ids = info['seq_ids'][valid_mask]

        start = 0
        while start < len(imgs):
            cur_scan = int(scan_ids[start])
            end = start + 1
            while end < len(imgs) and scan_ids[end] == cur_scan:
                end += 1
            if cur_scan > 0:
                dm = get_data_manager(cur_scan)
                dm.store_new_data({'imgs': imgs[start:end],
                                   'seq_ids': seq_ids[start:end]})
                dm.process_data()
                dm.update_data()
                # Persist to disk FIRST, then update the live display. Saving
                # must never depend on the display path: a downstream
                # get_plot_data() error (e.g. the cross-grid survival-series
                # crash) previously aborted this cycle BEFORE save_data ran,
                # so shots were dropped from the .h5 AND the live view froze /
                # lagged behind the backend. The display update is also isolated
                # so one bad frame can't starve the save or stall the loop.
                fname = ''
                save_err = ''
                try:
                    fname = dm.save_data()
                except Exception as e:
                    save_err = f'Save failed: {e}'
                    logger.error('save_data() failed for scan %d: %s',
                                 cur_scan, e)
                if self._dashboard:
                    try:
                        self._dashboard.update(dm.get_plot_data())
                    except Exception as e:
                        logger.error('dashboard update failed for scan %d: %s',
                                     cur_scan, e)
                self.after(0, self._init_pane.set_is_init_scan, dm.is_init)
                self._cur_scan_id = cur_scan
                self._cur_seq_id = int(seq_ids[-1])
                # Track this DM so dummy frames (cur_scan < 0) can borrow its
                # plot context. Updated only on real saves.
                self._last_real_dm = dm
                self.after(0, self._update_labels, fname, save_err)
            elif cur_scan == FAILING_DISPLAY_SCAN_ID:
                # Failing-shot frames published for DISPLAY ONLY (backend's
                # publish_failed_shot): the shot couldn't form a real pair, but
                # show img1 (+ img2 or "no data") so the live view keeps flashing
                # while the shot-health chip stays "failing". No save, no accum.
                self._dispatch_failing_batch(imgs[start:end], seq_ids[start:end])
            elif cur_scan < 0:
                # Dummy-mode frames: scan_id was set to -1 by SequenceRunner
                # before replaying the cached last seq. Suppress disk save and
                # in-memory accumulation, but push the latest raw frame to
                # the dashboard so the user can see camera activity.
                self._dispatch_dummy_batch(imgs[start:end], seq_ids[start:end])
            start = end

    def _dispatch_dummy_batch(self, batch_imgs, batch_seq_ids):
        """Display a dummy-mode image batch without persisting anything.

        Per-frame: run detect_atom on the live image using the last real
        DM's grid + thresholds, then push (cur_image, cur_intensities,
        logicals) so the Tweezer Array view shows green/red boxes and the
        Atom Intensities scatter updates.

        Per-batch: do NOT touch the cumulative state — the histograms,
        scan curve, grid shift, loading rates, gaussian fits, and discrim
        infidelities are inherited verbatim from the last real DM's
        get_plot_data() snapshot, so those panels freeze at the last real
        scan's values. Nothing is appended to save buffers, no HDF5 write,
        no _intensity_accum growth, no img_buffer push.

        If no real DM has been seen yet this session, drop the batch —
        the dashboard has nothing meaningful to render. Bytes still leave
        the ZMQ deque so memory is bounded by get_imgs draining."""
        if not self._dashboard or self._last_real_dm is None:
            return
        if not batch_imgs:
            return
        try:
            latest = batch_imgs[-1]
            # batch_imgs entries are (H, W, n_imgs_per_seq); pick the first
            # frame so the single-image panel has the right shape.
            if latest.ndim == 3 and latest.shape[2] >= 1:
                cur_img = latest[:, :, 0]
            else:
                cur_img = latest
            cur_img_f = cur_img.astype(np.float64)

            dm = self._last_real_dm
            data = dict(dm.get_plot_data())
            data['cur_image'] = cur_img_f
            if dm.num_sites > 0 and len(dm.grid_locations) > 0:
                # Local import: detect_atom is in a sibling module that
                # data_manager.py already pulls in, so no extra startup cost.
                from yb_analysis.detection.detect_atom import detect_atom
                logicals, intensities = detect_atom(
                    cur_img_f, dm.grid_locations,
                    dm.thresholds, dm.mask_mat,
                )
                data['cur_intensities'] = intensities
                data['logicals'] = logicals
                record_loading(logicals)
            else:
                # Grid not yet established — show bare frame, no boxes.
                data['cur_intensities'] = None
                data['logicals'] = None
            data['loading_history'] = get_loading_history()
            # Flag so the dashboard blanks out panels frozen at the last real
            # scan's values (histograms, scan curve, loading rates, etc.)
            # and labels them as Dummy mode.
            data['_dummy_mode'] = True
            self._dashboard.update(data)
        except Exception:
            logger.error('dummy dispatch error:\n%s', traceback.format_exc())

    def _dispatch_failing_batch(self, batch_imgs, batch_seq_ids):
        """Display a FAILING shot's captured frames without persisting anything.

        A failing rearrange shot can't form a real (img1, img2) pair, but we still flash whatever
        it captured so the operator sees activity instead of a frozen view: img1 always; img2 if it
        was captured, otherwise the array-2 panel shows "no data". Detection boxes use the last real
        DM's grid/thresholds when one exists (else the bare frame is shown). Cumulative panels carry
        over the last real scan's values -- the red shot-health chip already signals "failing".
        Nothing is appended to save buffers and no HDF5 write happens.

        Mirrors ``_dispatch_dummy_batch`` but (a) shows the SECOND frame too and (b) tags
        ``_failing_mode`` (not ``_dummy_mode``) so the dashboard labels it correctly. Unlike the
        dummy path it does NOT bail when no real DM exists yet -- a run that fails from its very
        first shot must still flash its frames (just without detection boxes)."""
        if not self._dashboard or not batch_imgs:
            return
        try:
            latest = batch_imgs[-1]
            # entries are (H, W, n_imgs_per_seq): img1 = [..., 0]; img2 = [..., 1] if captured.
            if getattr(latest, 'ndim', 0) == 3 and latest.shape[2] >= 1:
                img1 = latest[:, :, 0]
                img2 = latest[:, :, 1] if latest.shape[2] >= 2 else None
            else:
                img1, img2 = latest, None
            img1_f = img1.astype(np.float64)
            img2_f = img2.astype(np.float64) if img2 is not None else None

            dm = self._last_real_dm
            data = dict(dm.get_plot_data()) if dm is not None else {}
            data['cur_image'] = img1_f
            # cur_image2 only when img2 was actually captured; None -> "no data" in the array-2 panel
            # (don't let the last real img2 linger as if it were this shot's).
            data['cur_image2'] = img2_f
            if dm is not None and dm.num_sites > 0 and len(dm.grid_locations) > 0:
                from yb_analysis.detection.detect_atom import detect_atom
                logicals, intensities = detect_atom(
                    img1_f, dm.grid_locations, dm.thresholds, dm.mask_mat)
                data['cur_intensities'] = intensities
                data['logicals'] = logicals
                record_loading(logicals)
                if img2_f is not None:
                    logicals2, intensities2 = detect_atom(
                        img2_f, dm.grid_locations, dm.thresholds, dm.mask_mat)
                    data['cur_intensities2'] = intensities2
                    data['logicals2'] = logicals2
                else:
                    data.pop('logicals2', None)
                    data.pop('cur_intensities2', None)
            else:
                # No grid yet -> show the bare frame(s), no boxes.
                data['cur_intensities'] = None
                data['logicals'] = None
                data.pop('logicals2', None)
                data.pop('cur_intensities2', None)
            data['loading_history'] = get_loading_history()
            # Flag so the dashboard labels the image panels "failing shot" and the (frozen)
            # cumulative panels aren't mistaken for this shot's data.
            data['_failing_mode'] = True
            self._dashboard.update(data)
        except Exception:
            logger.error('failing dispatch error:\n%s', traceback.format_exc())

    def _update_labels(self, fname='', save_err=''):
        from yb_analysis.config import PATH_PREFIX
        self._lbl_scan.config(text=str(self._cur_scan_id))
        self._lbl_seq.config(text=str(self._cur_seq_id))
        if fname:
            try:
                display = os.path.relpath(os.path.dirname(fname), PATH_PREFIX)
            except ValueError:
                display = os.path.dirname(fname)
            self._lbl_file.config(text=display)
        # Sticky error indicator — clears only when the next save succeeds
        if save_err:
            self._lbl_save_err.config(text=save_err)
            self._lbl_save_err.grid()
        elif fname:
            self._lbl_save_err.config(text='')
            self._lbl_save_err.grid_remove()

    # ------------------------------------------------------------ Shutdown

    def _release_camera_for_teardown(self, timeout_s=5.0):
        """Release the camera gracefully before the backend is killed (thin wrapper over the
        testable module fn :func:`_release_camera_before_teardown`; full rationale there)."""
        _release_camera_before_teardown(self._client, self._backend, timeout_s=timeout_s)

    def _on_close(self):
        self._running = False
        # Graceful camera release BEFORE the backend is torn down (runner.stop() runs in
        # run_monitor's atexit, after self.destroy()), so a restart / switch / close can't
        # wedge the Orca. Replaces a fire-and-forget camera_close that didn't wait for release.
        try:
            self._release_camera_for_teardown()
        except Exception:
            pass
        self._client.cleanup()
        if self._dashboard:
            self._dashboard.close()
        self.destroy()
