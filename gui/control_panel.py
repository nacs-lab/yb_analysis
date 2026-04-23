"""Tkinter GUI for experiment control.

Processing runs in a background thread to keep the UI responsive.
Pause/Abort/Start use the MATLAB MemoryMap file for signaling.
"""

import mmap
import os
import struct
import tkinter as tk
import logging
import traceback
import threading
import time

import numpy as np

from yb_analysis.acquisition.data_manager import get_data_manager

logger = logging.getLogger(__name__)

# MemoryMap layout (matches MemoryMap.m):
# 12 doubles (96 bytes) + 32 uint8 (32 bytes) + 1 double (8 bytes) + 32 doubles (256 bytes) + 1 double (8 bytes)
# Key offsets (in bytes, each double = 8 bytes):
_MMAP_PATH = os.path.join(os.environ.get('TEMP', '/tmp'), 'nacsctl', 'nacs_mem_map.dat')
_OFF_SCAN_COMPLETE = 2 * 8   # ScanComplete
_OFF_ABORT = 8 * 8           # AbortRunSeq
_OFF_PAUSE = 9 * 8           # PauseRunSeq
_OFF_ISPAUSED = 10 * 8       # IsPausedRunSeq
_OFF_CURSEQNUM = 11 * 8      # CurrentSeqNum
# After 12 doubles (96B) + 32 uint8 (32B) + 1 double (8B) + 32 doubles (256B) = 392
_OFF_DUMMY_RUNNING = 392     # DummyRunning


def _mmap_open():
    """Open the MATLAB memory map file for read/write."""
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


class ControlPanel(tk.Tk):

    def __init__(self, zmq_client, dashboard=None):
        super().__init__()
        self.title('Yb Experiment Control')
        self.geometry('560x640')
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._client = zmq_client
        self._dashboard = dashboard
        self._cur_scan_id = 0
        self._cur_seq_id = 0
        self._refresh_ms = 2000
        self._running = True

        self._build_ui()

        # Processing in background thread
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        # Status polling on main thread (lightweight)
        self._poll_status()

        # Tk on macOS blocks Python's signal handlers inside mainloop(); a
        # periodic no-op tick lets SIGINT propagate so Ctrl-C in the spawning
        # terminal works.
        self._tick_alive()

    def _tick_alive(self):
        self.after(200, self._tick_alive)

    def _build_ui(self):
        self._lbl_status = tk.Label(self, text='Status: Unknown', font=('Segoe UI', 16))
        self._lbl_status.pack(fill='x', padx=10, pady=(12, 4))

        bf = tk.Frame(self)
        bf.pack(fill='x', padx=10, pady=8)
        tk.Button(bf, text='Pause', font=('Segoe UI', 11),
                  command=self._on_pause).pack(side='left', expand=True, fill='x', padx=4)
        tk.Button(bf, text='Start', font=('Segoe UI', 11),
                  command=self._on_start).pack(side='left', expand=True, fill='x', padx=4)
        tk.Button(bf, text='ABORT', font=('Segoe UI', 12, 'bold'), fg='red',
                  command=self._on_abort).pack(side='left', expand=True, fill='x', padx=4)

        inf = tk.Frame(self)
        inf.pack(fill='x', padx=10, pady=4)
        self._lbl_scan = tk.Label(inf, text='Scan: —', font=('Segoe UI', 12), anchor='w')
        self._lbl_scan.pack(fill='x')
        self._lbl_seq = tk.Label(inf, text='Seq: —', font=('Segoe UI', 12), anchor='w')
        self._lbl_seq.pack(fill='x')
        self._lbl_file = tk.Label(inf, text='File: —', font=('Segoe UI', 9), anchor='w', wraplength=500)
        self._lbl_file.pack(fill='x', pady=(6, 0))

        rf = tk.Frame(self)
        rf.pack(fill='x', padx=10, pady=8)
        tk.Label(rf, text='Refresh (s):', font=('Segoe UI', 11)).pack(side='left')
        self._rate_entry = tk.Entry(rf, width=4, font=('Segoe UI', 11))
        self._rate_entry.insert(0, str(self._refresh_ms // 1000))
        self._rate_entry.pack(side='left', padx=8)
        self._rate_entry.bind('<Return>', self._on_rate)

        # Queue pane for SequenceRunner jobs
        from yb_analysis.gui.queue_pane import QueuePane
        self._queue_pane = QueuePane(self, self._client, refresh_ms=1000)
        self._queue_pane.pack(fill='both', expand=True, padx=10, pady=(4, 10))

    def _on_pause(self):
        mm = _mmap_open()
        if mm:
            _mmap_write_double(mm, _OFF_PAUSE, 1.0)
            mm.close()
            logger.info('MemoryMap: PauseRunSeq = 1')
        else:
            self._client.pause_seq()

    def _on_start(self):
        mm = _mmap_open()
        if mm:
            _mmap_write_double(mm, _OFF_PAUSE, 0.0)
            mm.close()
            logger.info('MemoryMap: PauseRunSeq = 0 (resume)')
        else:
            self._client.start_seq()

    def _on_abort(self):
        mm = _mmap_open()
        if mm:
            _mmap_write_double(mm, _OFF_ABORT, 1.0)
            _mmap_write_double(mm, _OFF_PAUSE, 0.0)
            mm.close()
            logger.info('MemoryMap: AbortRunSeq = 1')
        else:
            self._client.abort_seq()

    def _on_rate(self, _=None):
        try:
            v = int(self._rate_entry.get())
            self._refresh_ms = max(500, v * 1000)
            self._client.set_refresh_rate(v)
        except ValueError:
            pass

    def _poll_status(self):
        """Poll status from MemoryMap (authoritative), fallback to ZMQ."""
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

                if abort > 0 or scan_complete > 0:
                    status = 'Stopped'
                elif is_paused > 0:
                    status = 'Paused'
                elif pause > 0:
                    status = 'Pausing...'
                elif dummy > 0:
                    status = 'Dummy Running'
                else:
                    status = 'Running'
                self._lbl_status.config(text=f'Status: {status}')
            else:
                # Fallback to ZMQ
                s = self._client.get_status()
                self._lbl_status.config(text=f'Status: {_STATUS.get(s, "?")}')
        except Exception:
            pass
        self.after(1000, self._poll_status)

    def _process_loop(self):
        """Background thread: grab images, process, update dashboard."""
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

        # Process grouped by scan_id
        start = 0
        while start < len(imgs):
            cur_scan = int(scan_ids[start])
            end = start + 1
            while end < len(imgs) and scan_ids[end] == cur_scan:
                end += 1

            if cur_scan > 0:
                dm = get_data_manager(cur_scan)
                dm.store_new_data({'imgs': imgs[start:end], 'seq_ids': seq_ids[start:end]})
                dm.process_data()
                dm.update_data()
                if self._dashboard:
                    self._dashboard.update(dm.get_plot_data())
                fname = dm.save_data()

                # Update UI from background thread (thread-safe via after())
                self._cur_scan_id = cur_scan
                self._cur_seq_id = int(seq_ids[-1])
                self.after(0, self._update_labels, fname)

            start = end

    def _update_labels(self, fname=''):
        """Called on main thread via after()."""
        self._lbl_scan.config(text=f'Scan: {self._cur_scan_id}')
        self._lbl_seq.config(text=f'Seq: {self._cur_seq_id}')
        if fname:
            self._lbl_file.config(text=f'File: {fname}')

    def _on_close(self):
        self._running = False
        self._client.cleanup()
        if self._dashboard:
            self._dashboard.close()
        self.destroy()
