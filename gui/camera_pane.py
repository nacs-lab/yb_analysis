"""Camera control pane for the Yb Experiment Control GUI.

Displays camera connection status and ROI.  ROI changes are sent to the
runner over ZMQ *and* written back to expConfig.m so both stay in sync.
"""

import logging
import threading
import tkinter as tk
import tkinter.ttk as ttk

from yb_analysis.config import write_orca_roi, write_orca_exposure

logger = logging.getLogger(__name__)


def _format_exposure(value):
    """Render exposure_time (seconds) for display. %g drops trailing zeros
    and switches to scientific notation for very small values."""
    try:
        return ('%g' % float(value))
    except (TypeError, ValueError):
        return '0'


class CameraPane(ttk.LabelFrame):

    def __init__(self, parent, zmq_client, *, refresh_ms=2000):
        super().__init__(parent, text='Camera')
        self._client = zmq_client
        self._refresh_ms = refresh_ms
        self._poll_lock = threading.Lock()
        self._poll_busy = False
        self._connected = False
        self._cmd_pending = False  # True while waiting for a command to take effect
        self._last_server_roi = None  # track what the server reports
        self._last_server_exposure = None

        # Deferred expConfig persistence: user-applied values land here in the
        # main thread before the ZMQ command is queued.  The poll loop writes
        # expConfig.m only when the server actually reflects them — that way a
        # failed init never leaves stale values in the persistent config.
        self._pending_persist_roi = None
        self._pending_persist_exposure = None

        self._build_ui()
        self._schedule_refresh()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        # Status row
        sf = ttk.Frame(self)
        sf.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Label(sf, text='Status:').pack(side='left')
        self._lbl_status = ttk.Label(sf, text='Disconnected', foreground='gray')
        self._lbl_status.pack(side='left', padx=(4, 0))

        # ROI fields
        rf = ttk.Frame(self)
        rf.pack(fill='x', padx=6, pady=2)
        self._roi_vars = {}
        self._roi_entries = {}
        for i, label in enumerate(('X', 'Y', 'W', 'H')):
            ttk.Label(rf, text=label + ':').grid(row=0, column=i * 2, padx=(6, 1))
            var = tk.StringVar(value='0')
            e = ttk.Entry(rf, textvariable=var, width=6)
            e.grid(row=0, column=i * 2 + 1, padx=(0, 4))
            self._roi_vars[label] = var
            self._roi_entries[label] = e

        # Exposure field (seconds)
        ef = ttk.Frame(self)
        ef.pack(fill='x', padx=6, pady=2)
        ttk.Label(ef, text='Exposure (s):').pack(side='left', padx=(6, 1))
        self._exposure_var = tk.StringVar(value='0.1')
        self._exposure_entry = ttk.Entry(ef, textvariable=self._exposure_var, width=10)
        self._exposure_entry.pack(side='left', padx=(0, 4))

        # Buttons
        bf = ttk.Frame(self)
        bf.pack(fill='x', padx=6, pady=(2, 2))
        self._btn_connect = ttk.Button(bf, text='Connect',
                                       command=self._on_connect)
        self._btn_connect.pack(side='left', padx=2)
        self._btn_disconnect = ttk.Button(bf, text='Disconnect',
                                          command=self._on_disconnect)
        self._btn_disconnect.pack(side='left', padx=2)
        ttk.Button(bf, text='Apply Settings',
                   command=self._on_apply_all).pack(side='left', padx=2)

        # Error label — placed below the button row so long messages
        # (e.g. ROIPosition validation errors) wrap inside the pane
        # instead of overflowing past Apply Settings.
        self._lbl_error = ttk.Label(self, text='', foreground='red',
                                    wraplength=320, justify='left')
        self._lbl_error.pack(fill='x', padx=6, pady=(0, 6))

    # ---------------------------------------------------------- ROI helpers

    def set_roi(self, roi):
        """Set the ROI entry fields (called externally on startup)."""
        for val, key in zip(roi, ('X', 'Y', 'W', 'H')):
            self._roi_vars[key].set(str(int(val)))

    def get_roi(self):
        return [int(self._roi_vars[k].get()) for k in ('X', 'Y', 'W', 'H')]

    def set_exposure(self, exposure_time):
        """Set the exposure entry (called externally on startup)."""
        self._exposure_var.set(_format_exposure(exposure_time))

    def get_exposure(self):
        return float(self._exposure_var.get())

    # ------------------------------------------------------------ Callbacks

    def _on_connect(self):
        try:
            roi = self.get_roi()
        except ValueError:
            self._lbl_error.config(text='Bad ROI values')
            return
        try:
            exposure = self.get_exposure()
            if not (exposure > 0):
                raise ValueError
        except ValueError:
            self._lbl_error.config(text='Bad exposure value')
            return
        self._lbl_status.config(text='Connecting...', foreground='orange')
        self._lbl_error.config(text='')
        self._cmd_pending = True
        self._pending_persist_roi = list(roi)
        self._pending_persist_exposure = float(exposure)
        threading.Thread(target=self._do_connect, args=(roi, exposure),
                         daemon=True).start()

    def _do_connect(self, roi, exposure):
        try:
            self._client.camera_init(roi, exposure_time=exposure)
        except Exception as e:
            # ZMQ call itself failed — command never reached the runner.
            # Clear pending so a later success doesn't wrongly persist.
            self._cmd_pending = False
            self._pending_persist_roi = None
            self._pending_persist_exposure = None
            self.after(0, self._lbl_error.config, {'text': str(e)[:60]})

    def _on_disconnect(self):
        self._lbl_status.config(text='Disconnecting...', foreground='orange')
        self._cmd_pending = True
        threading.Thread(target=self._do_disconnect, daemon=True).start()

    def _do_disconnect(self):
        try:
            self._client.camera_close()
        except Exception as e:
            self._cmd_pending = False
            self.after(0, self._lbl_error.config, {'text': str(e)[:60]})

    def _on_apply_all(self, _event=None):
        # Read and validate all values first, before any poll callback can
        # overwrite fields (poll runs in main thread via after(), so it cannot
        # fire while we're in this call, but it may have fired just before the
        # button command ran — the values are already captured here).
        try:
            roi = self.get_roi()
        except ValueError:
            self._lbl_error.config(text='Bad ROI values')
            return
        try:
            exposure = self.get_exposure()
            if not (exposure > 0):
                raise ValueError
        except ValueError:
            self._lbl_error.config(text='Bad exposure value')
            return
        self._lbl_error.config(text='')
        self._lbl_status.config(text='Applying settings...', foreground='orange')
        self._cmd_pending = True
        self._pending_persist_roi = list(roi)
        self._pending_persist_exposure = float(exposure)
        threading.Thread(target=self._do_apply_all, args=(roi, exposure),
                         daemon=True).start()

    def _do_apply_all(self, roi, exposure):
        try:
            self._client.camera_apply_settings(roi, exposure)
        except Exception as e:
            self._pending_persist_roi = None
            self._pending_persist_exposure = None
            self.after(0, self._lbl_error.config, {'text': str(e)[:60]})
            self.after(0, self._clear_cmd_pending)
            return
        # The ZMQ ack is a queue-ack only — the runner applies asynchronously,
        # so we must NOT clear _cmd_pending here or the next poll would flip
        # status to "Connected" before the camera has actually reconfigured.
        # _maybe_persist clears _cmd_pending once the server reports the new
        # values. The watchdog below covers two cases _maybe_persist can't:
        # (a) MATLAB reports a server-side error (err set, persist condition
        # rejects), and (b) the camera snaps ROI/exposure to a hardware grid
        # so the exact-match check never succeeds. Without it, the UI would
        # stay on "Applying settings..." until the user clicked Connect.
        self.after(8000, self._clear_cmd_pending)

    def _clear_cmd_pending(self):
        self._cmd_pending = False

    # --------------------------------------------------- Background polling

    def _schedule_refresh(self):
        self.after(self._refresh_ms, self._refresh)

    def _refresh(self):
        with self._poll_lock:
            if self._poll_busy:
                self._schedule_refresh()
                return
            self._poll_busy = True
        threading.Thread(target=self._poll_worker, daemon=True).start()
        self._schedule_refresh()

    def _poll_worker(self):
        try:
            status = self._client.camera_status()
            self.after(0, self._on_poll_ok, status)
        except Exception:
            pass
        finally:
            with self._poll_lock:
                self._poll_busy = False

    def _on_poll_ok(self, status):
        connected = status.get('connected', False)
        roi = status.get('roi', [0, 0, 0, 0])
        exposure = status.get('exposure_time', None)
        err = status.get('error', '')

        # Save previous server values before updating them — used below to
        # detect whether the user has made an unsaved edit (field diverged from
        # last server value), in which case the poll must not overwrite it.
        prev_server_roi = self._last_server_roi
        prev_server_exposure = self._last_server_exposure

        # connect/disconnect: detect completion via connection-state change.
        # apply_settings completion is signaled by _maybe_persist clearing
        # both persist slots once the server reports the requested values.
        if self._cmd_pending and connected != self._connected:
            self._cmd_pending = False

        self._last_server_roi = roi
        self._last_server_exposure = exposure
        self._connected = connected

        # Persist to expConfig.m only once the server reflects what the user
        # requested. This runs regardless of _cmd_pending so rapid successive
        # applies still persist correctly when each lands.
        self._maybe_persist(connected, roi, exposure, err)

        # While a command is pending, don't overwrite the UI
        if self._cmd_pending:
            return

        if connected:
            self._lbl_status.config(text='Connected', foreground='green')
            self._lbl_error.config(text='')
            # Only sync fields from server when:
            #   • the user isn't actively editing them (focus check), AND
            #   • the field still reflects the previous server value, i.e. the
            #     user hasn't typed a new value they haven't applied yet.
            # This prevents the poll from clobbering a user-edited field in
            # the brief window between the entry losing focus and the Apply
            # button command executing.
            if not self._any_roi_focused():
                try:
                    ui_roi = self.get_roi()
                except ValueError:
                    ui_roi = None
                if prev_server_roi is None or ui_roi is None or ui_roi == prev_server_roi:
                    if ui_roi != roi:
                        self.set_roi(roi)
            if exposure is not None and not self._exposure_focused():
                try:
                    ui_exp = self.get_exposure()
                except ValueError:
                    ui_exp = None
                if (prev_server_exposure is None or ui_exp is None
                        or abs(ui_exp - prev_server_exposure) < 1e-9):
                    if ui_exp is None or abs(ui_exp - float(exposure)) > 1e-9:
                        self.set_exposure(exposure)
        else:
            if err:
                self._lbl_status.config(text='Error', foreground='red')
                self._lbl_error.config(text=err[:60])
            else:
                self._lbl_status.config(text='Disconnected', foreground='gray')
                self._lbl_error.config(text='')

    # ---------------------------------------------------- Focus-aware helpers

    def _focused_widget(self):
        """Return the currently focused widget, or None on error. Tk can
        raise KeyError/TclError briefly during teardown or while the mouse
        is outside the app, so we treat those as 'no focus'."""
        try:
            return self.focus_get()
        except (KeyError, tk.TclError):
            return None

    def _any_roi_focused(self):
        focused = self._focused_widget()
        if focused is None:
            return False
        return any(focused is e for e in self._roi_entries.values())

    def _exposure_focused(self):
        return self._focused_widget() is self._exposure_entry

    # --------------------------------------------- Deferred expConfig persist

    def _maybe_persist(self, connected, roi, exposure, err):
        """Write expConfig.m when — and only when — the server confirms the
        value the user applied. A persisted command clears its own pending
        slot. We deliberately do NOT clear on errors: a stale error from a
        previous command shouldn't cancel a later successful apply.  Pending
        slots are overwritten in the main thread on each new Apply, so they
        can't accumulate."""
        cleared = False
        if self._pending_persist_roi is not None:
            if (connected and not err
                    and list(roi) == list(self._pending_persist_roi)):
                try:
                    write_orca_roi(roi)
                except Exception as e:
                    logger.warning('write_orca_roi failed: %s', e)
                self._pending_persist_roi = None
                cleared = True
        if self._pending_persist_exposure is not None and exposure is not None:
            requested = float(self._pending_persist_exposure)
            actual = float(exposure)
            # Hamamatsu snaps to discrete hardware steps so the server-reported
            # value may differ from requested by ~tens of μs.  Use 1% relative
            # tolerance — much larger than any camera quantization step, yet
            # tight enough that clearly different exposure times never match.
            if (connected and not err
                    and abs(actual - requested) / max(abs(requested), 1e-9) < 0.01):
                try:
                    write_orca_exposure(actual)
                except Exception as e:
                    logger.warning('write_orca_exposure failed: %s', e)
                self._pending_persist_exposure = None
                cleared = True
        # Drop _cmd_pending once the server has confirmed every requested
        # change. Gated on `cleared` so a stale poll (slots already None
        # from before) doesn't prematurely clear an in-flight Disconnect,
        # which never sets persist slots and relies on the connection-
        # state-change check above to clear _cmd_pending.
        if (cleared and self._cmd_pending
                and self._pending_persist_roi is None
                and self._pending_persist_exposure is None):
            self._cmd_pending = False
