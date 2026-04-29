"""Tkinter GUI for experiment control.

Processing runs in a background thread to keep the UI responsive.
Pause/Abort/Start use the MATLAB MemoryMap file for signaling.
"""

import mmap
import os
import struct
import tempfile
import tkinter as tk
import tkinter.ttk as ttk
import logging
import traceback
import threading
import time

import numpy as np

from yb_analysis.acquisition.data_manager import get_data_manager

logger = logging.getLogger(__name__)

# MemoryMap layout (matches MemoryMap.m)
_MMAP_PATH = os.path.join(tempfile.gettempdir(), 'nacsctl', 'nacs_mem_map.dat')
_OFF_SCAN_COMPLETE = 2 * 8
_OFF_ABORT = 8 * 8
_OFF_PAUSE = 9 * 8
_OFF_ISPAUSED = 10 * 8
_OFF_CURSEQNUM = 11 * 8
_OFF_DUMMY_RUNNING = 392

_FONT = ('Segoe UI', 10)
_FONT_SM = ('Segoe UI', 9)
_FONT_TITLE = ('Segoe UI', 14, 'bold')


def _mmap_open():
    if not os.path.isfile(_MMAP_PATH):
        return None
    try:
        f = open(_MMAP_PATH, 'r+b')
        return mmap.mmap(f.fileno(), 0)
    except Exception as e:
        logger.debug('Could not open MemoryMap: %s', e)
        return None


def _mmap_write_double(mm, offset, value):
    mm.seek(offset)
    mm.write(struct.pack('d', float(value)))


def _mmap_read_double(mm, offset):
    mm.seek(offset)
    return struct.unpack('d', mm.read(8))[0]


_STATUS = {0: 'Stopped', 1: 'Running', 2: 'Paused', 3: 'Unknown'}
_STATUS_COLORS = {
    'Idle': '#555555', 'Idle (dummy off)': '#888888',
    'Running': '#006600', 'Paused': '#cc6600',
    'Pausing...': '#cc6600', 'Stopped': '#aa0000',
}


class ControlPanel(tk.Tk):

    def __init__(self, zmq_client, dashboard=None, init_dir=None, init_status=''):
        super().__init__()
        self.title('Yb Experiment Control')
        self.geometry('640x720')
        self.minsize(520, 560)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._client = zmq_client
        self._dashboard = dashboard
        self._init_dir = init_dir
        self._init_status = init_status
        self._cur_scan_id = 0
        self._cur_seq_id = 0
        self._refresh_ms = 2000
        self._running = True
        self._dummy_enabled = True  # tracks the server-side keep-alive toggle

        style = ttk.Style(self)
        style.configure('Abort.TButton', foreground='red',
                        font=('Segoe UI', 10, 'bold'))

        self._build_ui()

        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()
        self._poll_status()
        self._tick_alive()

    def _tick_alive(self):
        self.after(200, self._tick_alive)

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

        # Dummy keep-alive toggle — when off, SequenceRunner stops running
        # DummySeq between jobs (nothing hits the hardware while idle).
        self._dummy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='Run dummy seq',
                        variable=self._dummy_var,
                        command=self._on_dummy_toggle).pack(
            side='right', padx=(12, 8))
        # Sync the UI from the server on startup so a persistent runner that
        # was last left disabled doesn't silently flip back to enabled just
        # because the UI defaults to checked.
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
        self._rate_entry = ttk.Entry(rf, width=3, font=_FONT_SM)
        self._rate_entry.insert(0, str(self._refresh_ms // 1000))
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
        self._queue_pane = QueuePane(self, self._client, refresh_ms=1000)
        self._queue_pane.pack(fill='both', expand=True, padx=10, pady=(2, 8))

    # -------------------------------------------------------------- Actions

    def _on_pause(self):
        mm = _mmap_open()
        if mm:
            _mmap_write_double(mm, _OFF_PAUSE, 1.0)
            mm.close()
        else:
            self._client.pause_seq()

    def _on_start(self):
        mm = _mmap_open()
        if mm:
            _mmap_write_double(mm, _OFF_PAUSE, 0.0)
            mm.close()
        else:
            self._client.start_seq()

    def _on_abort(self):
        mm = _mmap_open()
        if mm:
            _mmap_write_double(mm, _OFF_ABORT, 1.0)
            _mmap_write_double(mm, _OFF_PAUSE, 0.0)
            mm.close()
        else:
            self._client.abort_seq()

    def _on_init_loaded(self, data):
        if self._dashboard and data is not None:
            self._dashboard.update(data)

    def _on_rate(self, _=None):
        try:
            v = int(self._rate_entry.get())
            self._refresh_ms = max(500, v * 1000)
        except ValueError:
            pass

    def _on_dummy_toggle(self):
        self._push_dummy_state()

    def _push_dummy_state(self):
        enabled = bool(self._dummy_var.get())
        self._dummy_enabled = enabled
        # Run on a background thread: ZMQ REQ/REP is not instant and we don't
        # want to freeze the UI if the runner is slow to respond.
        threading.Thread(
            target=self._do_push_dummy, args=(enabled,), daemon=True).start()

    def _do_push_dummy(self, enabled):
        try:
            self._client.set_dummy_enabled(enabled)
        except Exception as e:
            logger.warning('set_dummy_enabled failed: %s', e)

    def _load_dummy_state(self):
        """Read the server-side dummy flag and reflect it in the checkbox.
        Runs on a worker thread; posts the UI update back to the main loop
        via after(0) so Tk is only touched from the main thread."""
        try:
            enabled = self._client.get_dummy_enabled()
        except Exception as e:
            logger.warning('get_dummy_enabled failed: %s', e)
            return
        self.after(0, self._apply_dummy_state_from_server, enabled)

    def _apply_dummy_state_from_server(self, enabled):
        self._dummy_var.set(bool(enabled))
        self._dummy_enabled = bool(enabled)

    # --------------------------------------------------------- Status poll

    def _poll_status(self):
        if not self._running:
            return
        try:
            mm = _mmap_open()
            if mm:
                abort = _mmap_read_double(mm, _OFF_ABORT)
                pause = _mmap_read_double(mm, _OFF_PAUSE)
                is_paused = _mmap_read_double(mm, _OFF_ISPAUSED)
                scan_complete = _mmap_read_double(mm, _OFF_SCAN_COMPLETE)
                dummy = _mmap_read_double(mm, _OFF_DUMMY_RUNNING)
                mm.close()

                if dummy > 0:
                    status = 'Idle (dummy off)' if not self._dummy_enabled else 'Idle'
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
                self._lbl_status.config(
                    text=f'Status: {status}',
                    foreground=_STATUS_COLORS.get(status, '#000000'))
            else:
                s = self._client.get_status()
                self._lbl_status.config(text=f'Status: {_STATUS.get(s, "?")}')
        except Exception:
            pass
        self.after(1000, self._poll_status)

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
                if self._dashboard:
                    self._dashboard.update(dm.get_plot_data())
                self.after(0, self._init_pane.set_is_init_scan, dm.is_init)
                fname = ''
                save_err = ''
                try:
                    fname = dm.save_data()
                except Exception as e:
                    save_err = f'Save failed: {e}'
                    logger.error('save_data() failed for scan %d: %s',
                                 cur_scan, e)
                self._cur_scan_id = cur_scan
                self._cur_seq_id = int(seq_ids[-1])
                self.after(0, self._update_labels, fname, save_err)
            start = end

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

    def _on_close(self):
        self._running = False
        try:
            self._client.camera_close()
        except Exception:
            pass
        self._client.cleanup()
        if self._dashboard:
            self._dashboard.close()
        self.destroy()
