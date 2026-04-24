"""Camera control pane for the Yb Experiment Control GUI.

Displays camera connection status and ROI.  ROI changes are sent to the
runner over ZMQ *and* written back to expConfig.m so both stay in sync.
"""

import logging
import threading
import tkinter as tk
import tkinter.ttk as ttk

from yb_analysis.config import write_orca_roi

logger = logging.getLogger(__name__)


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
        for i, label in enumerate(('X', 'Y', 'W', 'H')):
            ttk.Label(rf, text=label + ':').grid(row=0, column=i * 2, padx=(6, 1))
            var = tk.StringVar(value='0')
            e = ttk.Entry(rf, textvariable=var, width=6)
            e.grid(row=0, column=i * 2 + 1, padx=(0, 4))
            e.bind('<Return>', self._on_apply_roi)
            self._roi_vars[label] = var

        # Buttons
        bf = ttk.Frame(self)
        bf.pack(fill='x', padx=6, pady=(2, 6))
        self._btn_connect = ttk.Button(bf, text='Connect',
                                       command=self._on_connect)
        self._btn_connect.pack(side='left', padx=2)
        self._btn_disconnect = ttk.Button(bf, text='Disconnect',
                                          command=self._on_disconnect)
        self._btn_disconnect.pack(side='left', padx=2)
        ttk.Button(bf, text='Apply ROI',
                   command=self._on_apply_roi).pack(side='left', padx=2)
        self._lbl_error = ttk.Label(bf, text='', foreground='red')
        self._lbl_error.pack(side='right', padx=4)

    # ---------------------------------------------------------- ROI helpers

    def set_roi(self, roi):
        """Set the ROI entry fields (called externally on startup)."""
        for val, key in zip(roi, ('X', 'Y', 'W', 'H')):
            self._roi_vars[key].set(str(int(val)))

    def get_roi(self):
        return [int(self._roi_vars[k].get()) for k in ('X', 'Y', 'W', 'H')]

    # ------------------------------------------------------------ Callbacks

    def _on_connect(self):
        try:
            roi = self.get_roi()
        except ValueError:
            self._lbl_error.config(text='Bad ROI values')
            return
        self._lbl_status.config(text='Connecting...', foreground='orange')
        self._lbl_error.config(text='')
        self._cmd_pending = True
        threading.Thread(target=self._do_connect, args=(roi,),
                         daemon=True).start()

    def _do_connect(self, roi):
        try:
            self._client.camera_init(roi)
        except Exception as e:
            self._cmd_pending = False
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

    def _on_apply_roi(self, _event=None):
        try:
            roi = self.get_roi()
        except ValueError:
            self._lbl_error.config(text='Bad ROI values')
            return
        self._lbl_error.config(text='')
        self._lbl_status.config(text='Updating ROI...', foreground='orange')
        self._cmd_pending = True
        threading.Thread(target=self._do_set_roi, args=(roi,),
                         daemon=True).start()

    def _do_set_roi(self, roi):
        try:
            self._client.camera_set_roi(roi)
            write_orca_roi(roi)
        except Exception as e:
            self._cmd_pending = False
            self.after(0, self._lbl_error.config, {'text': str(e)[:60]})

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
        err = status.get('error', '')

        # Detect when a pending command has taken effect
        if self._cmd_pending:
            if connected != self._connected:
                self._cmd_pending = False  # state changed — command landed
            elif self._last_server_roi and roi != self._last_server_roi:
                self._cmd_pending = False  # ROI changed — command landed

        self._last_server_roi = roi
        self._connected = connected

        # While a command is pending, don't overwrite the UI
        if self._cmd_pending:
            return

        if connected:
            self._lbl_status.config(text='Connected', foreground='green')
            self._lbl_error.config(text='')
            # Only sync ROI fields from server when they actually differ
            # (avoids clobbering user edits while they're typing)
            try:
                ui_roi = self.get_roi()
            except ValueError:
                ui_roi = None
            if ui_roi != roi:
                self.set_roi(roi)
        else:
            if err:
                self._lbl_status.config(text='Error', foreground='red')
                self._lbl_error.config(text=err[:60])
            else:
                self._lbl_status.config(text='Disconnected', foreground='gray')
                self._lbl_error.config(text='')
