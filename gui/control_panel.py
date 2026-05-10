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
_OFF_NUM_PER_GROUP = 5 * 8     # set positive by runJob, zeroed at runJob exit
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
    'Idle (default)': '#555555', 'Idle (last seq)': '#0a5d8a',
    'Idle (last fallback)': '#cc6600',
    'Running': '#006600', 'Paused': '#cc6600',
    'Pausing...': '#cc6600', 'Stopped': '#aa0000',
}

# Dummy-mode radio values must match the strings ExptServer.dummy_mode accepts.
_DUMMY_MODES = ('off', 'default', 'last')


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
        # Mirrors the server's __dummy_mode. Used by the status-label rendering
        # to distinguish 'Idle (default)' from 'Idle (last seq)' etc.
        self._dummy_mode = 'last'
        # Last value of last_seq_status the GUI has seen — drives the cached-
        # seq label and the enable state of the "Last seq" radio button.
        self._last_seq_meta = {
            'available': False, 'name': '', 'file_id': '',
            'fallback_active': False,
        }
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
            mm = _mmap_open()
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
                self._lbl_status.config(
                    text=f'Status: {status}',
                    foreground=_STATUS_COLORS.get(status, '#000000'))
            else:
                s = self._client.get_status()
                self._lbl_status.config(text=f'Status: {_STATUS.get(s, "?")}')
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

    def _refresh_last_seq_async(self):
        try:
            meta = self._client.last_seq_status()
        except Exception:
            return
        if meta is not None:
            self.after(0, self._update_last_seq_label, meta)

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
                # Track this DM so dummy frames (cur_scan < 0) can borrow its
                # plot context. Updated only on real saves.
                self._last_real_dm = dm
                self.after(0, self._update_labels, fname, save_err)
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
            else:
                # Grid not yet established — show bare frame, no boxes.
                data['cur_intensities'] = None
                data['logicals'] = None
            self._dashboard.update(data)
        except Exception:
            logger.error('dummy dispatch error:\n%s', traceback.format_exc())

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
