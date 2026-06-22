"""ZMQ client wrapper with numpy conversion.

Thin GUI-side wrapper around the vendored ExptClient (acquisition/expt_client.py).
Adds numpy conversion for the image stream, the camera_init wait-for-connected
wrapper, status string-to-int translation, and a single lock to serialize REQ
access from multiple GUI threads.
"""

import logging
import threading
import time
import numpy as np

from .expt_client import ExptClient

logger = logging.getLogger(__name__)


# ExptServer returns one of these strings; ZmqClient.get_status returns the
# int code so callers (control_panel._STATUS) can dispatch.
_STATUS_STR_TO_INT = {
    'Sequence is stopped': 0,
    'Sequence is running': 1,
    'Sequence is paused': 2,
}

# After a grab_imgs failure, wait this long before trying again. The next
# call inside the cooldown window returns empty without acquiring the lock.
_GRAB_IMGS_COOLDOWN_S = 1.0

# Suppress duplicate get_status warnings within this window (seconds).
_STATUS_WARN_THROTTLE_S = 30.0


def _process_imgs(raw_data):
    """Parse the flat double array returned by ExptClient.get_imgs().

    Wire format (from ExptServer.py):
        [num_seqs, <per-sequence blocks>]
        Each sequence: scan_id, seq_id, <per-image blocks>
        Each image:    s1, s2, s3, <s1*s2*s3 pixel values>
        Sequences separated by 0.

    Returns
    -------
    dict with:
        imgs : list of ndarray, each (H, W, n_imgs_per_seq)
        scan_ids : list of int
        seq_ids : list of int
    """
    if raw_data is None or len(raw_data) == 0:
        return {'imgs': [], 'scan_ids': [], 'seq_ids': []}

    res = np.asarray(raw_data, dtype=np.float64)

    if res.size == 0:
        return {'imgs': [], 'scan_ids': [], 'seq_ids': []}

    num_seqs = int(res[0])
    if num_seqs == 0:
        return {'imgs': [], 'scan_ids': [], 'seq_ids': []}

    imgs = []
    scan_ids = []
    seq_ids = []

    idx = 1
    seq_count = 0
    first_img = True
    cur_img_stack = None

    while idx < len(res) and seq_count < num_seqs:
        # Check for sequence separator (0)
        if res[idx] == 0:
            if cur_img_stack is not None:
                imgs.append(cur_img_stack)
                cur_img_stack = None
            seq_count += 1
            idx += 1
            first_img = True
            continue

        # Read scan_id and seq_id for first image of sequence
        if first_img:
            scan_id = int(res[idx])
            idx += 1
            seq_id = int(res[idx])
            idx += 1
            scan_ids.append(scan_id)
            seq_ids.append(seq_id)
            first_img = False
            cur_img_stack = None

        # Read image dimensions
        s1 = int(res[idx])
        s2 = int(res[idx + 1])
        s3 = int(res[idx + 2])
        idx += 3

        # MATLAB sends pixel data in column-major (Fortran) order.
        # Must reshape with order='F' to get correct image orientation.
        n_pixels = s1 * s2 * s3
        img_data = res[idx:idx + n_pixels].reshape(s1, s2, s3, order='F')
        idx += n_pixels

        if cur_img_stack is None:
            cur_img_stack = img_data
        else:
            cur_img_stack = np.concatenate([cur_img_stack, img_data], axis=2)

    # Don't forget last sequence
    if cur_img_stack is not None:
        imgs.append(cur_img_stack)

    return {
        'imgs': imgs,
        'scan_ids': np.array(scan_ids, dtype=np.int64),
        'seq_ids': np.array(seq_ids, dtype=np.int64),
    }


class ZmqClient:
    """High-level ZMQ client for experiment control.

    Wraps ExptClient with numpy conversions, status string-to-int translation,
    a serialization lock, and the camera_init wait-for-connected wrapper.

    Parameters
    ----------
    url : str
        ZMQ server URL (default: tcp://127.0.0.1:1312).
    refresh_rate : optional
        Accepted for backwards compatibility (run_monitor.py CLI). Ignored;
        polling cadence is now owned by individual GUI panes.
    """

    def __init__(self, url='tcp://127.0.0.1:1312', refresh_rate=None):
        self._url = url
        self._client = ExptClient(url)
        # Multiple GUI threads call into ZmqClient (queue_pane._poll_worker,
        # camera_pane workers, control_panel._process_loop). REQ has a strict
        # SEND -> RECV state machine, so all wire access must serialize.
        self._lock = threading.Lock()
        # Circuit breaker for grab_imgs: when the runner doesn't reply,
        # cool down before trying again so a dead runner doesn't hog the
        # shared lock and starve queue/camera polls.
        self._grab_imgs_cooldown_until = 0.0
        # Throttle for get_status warnings: log at most once per
        # _STATUS_WARN_THROTTLE_S, otherwise a dead runner spams the log
        # at the GUI's 1 Hz status-poll cadence.
        self._last_get_status_warn = 0.0

    # -------- Liveness / queue --------

    def ping(self, timeout_ms=500):
        with self._lock:
            try:
                return self._client.ping(timeout_ms)
            except Exception:
                return False

    def submit_job(self, payload):
        with self._lock:
            return self._client.submit_job(payload)

    def queue_list(self, timeout_ms=400):
        with self._lock:
            return self._client.queue_list(timeout_ms)

    def queue_remove(self, job_id):
        with self._lock:
            return self._client.queue_remove(job_id)

    def queue_move(self, job_id, direction):
        with self._lock:
            return self._client.queue_move(job_id, direction)

    # -------- Descriptor queue (Phase 3) --------

    def submit_scan_descriptor(self, descriptor_json, label=''):
        """Submit a JSON scan descriptor. Returns the descriptor's queue
        id. The SequenceRunner pops it between jobs and dispatcher
        converts it to a regular job; the resulting job_id appears as
        the descriptor row's `built_job_id` in queue_list output."""
        with self._lock:
            return self._client.submit_scan_descriptor(
                descriptor_json, label=label)

    def descriptor_remove(self, desc_id):
        """Cancel a queued descriptor. Returns 'ok' or 'error: ...'."""
        with self._lock:
            return self._client.descriptor_remove(desc_id)

    # -------- Camera --------

    def camera_init(self, roi, exposure_time=None, timeout_ms=10000,
                    wait_connected_s=45.0):
        """Initialize the camera, blocking until the runner reports it
        connected.

        Why block: the 'ok' from this ZMQ call is only a queue-ack; MATLAB
        still spends ~25-30s in imaqreset + OrcaInit before the camera is
        actually usable. If anything else accesses the shared
        IMAQ/AslDma/DCAM state during that window - e.g. a separate
        `matlab.exe -batch` submitter is booting up to call submit_job,
        which loads its own IMAQ adaptors - OrcaInit can fail silently:
        the runner never sets `vid` in its base workspace, every
        subsequent scan goes down the "no camera" branch, and frames
        never flow. Empirically this caused ~30% of 2-cycle test runs
        to capture 0 frames on cycle 2; gating callers on
        `connected=True` eliminated it (31/31 cycles, vs ~70% before).

        Polls camera_status until the runner sets connected=True (only
        emitted by handleCameraCmd 'init' AFTER OrcaInit returns), or
        until `error` is set, or until wait_connected_s elapses. Pass
        wait_connected_s=0 for legacy fire-and-forget behavior.

        `roi` is [x, y, w, h]. `exposure_time` (seconds) is optional -
        when None the runner uses OrcaInit's default."""
        with self._lock:
            ack = self._client.camera_init(roi, exposure_time, timeout_ms)
        if wait_connected_s <= 0:
            return ack
        deadline = time.monotonic() + wait_connected_s
        last_status = None
        while time.monotonic() < deadline:
            try:
                st = self.camera_status(timeout_ms=500)
            except Exception:
                st = None
            if isinstance(st, dict):
                last_status = st
                if st.get('connected'):
                    return ack
                err = st.get('error') or ''
                if err:
                    raise RuntimeError(f'Camera init failed: {err}')
            time.sleep(0.5)
        raise TimeoutError(
            f'Camera did not report connected within {wait_connected_s:.1f}s '
            f'(last status: {last_status})')

    def camera_apply_settings(self, roi, exposure_time, timeout_ms=5000):
        with self._lock:
            return self._client.camera_apply_settings(roi, exposure_time, timeout_ms)

    def camera_close(self, timeout_ms=5000):
        with self._lock:
            return self._client.camera_close(timeout_ms)

    def camera_status(self, timeout_ms=1000):
        with self._lock:
            return self._client.camera_status(timeout_ms)

    # -------- Dummy keep-alive --------

    def set_dummy_enabled(self, enabled, timeout_ms=2000):
        with self._lock:
            return self._client.set_dummy_enabled(enabled, timeout_ms)

    def get_dummy_enabled(self, timeout_ms=1000):
        with self._lock:
            try:
                return self._client.get_dummy_enabled(timeout_ms)
            except Exception:
                return True

    def set_dummy_mode(self, mode, timeout_ms=2000):
        with self._lock:
            return self._client.set_dummy_mode(mode, timeout_ms)

    def get_dummy_mode(self, timeout_ms=1000):
        with self._lock:
            try:
                return self._client.get_dummy_mode(timeout_ms)
            except Exception:
                return 'default'

    def last_seq_status(self, timeout_ms=1000):
        """Returns {available, name, file_id, captured_at, fallback_active, mode}.
        Returns None on wire failure so callers can decide whether to retry."""
        with self._lock:
            try:
                return self._client.last_seq_status(timeout_ms)
            except Exception:
                return None

    def clear_last_seq_meta(self, timeout_ms=2000):
        with self._lock:
            try:
                return self._client.clear_last_seq_meta(timeout_ms)
            except Exception:
                return None

    def shot_health(self, timeout_ms=1000):
        """Per-shot health rollup (pyctrl backend). Returns None on wire failure
        OR when the backend lacks the verb (MATLAB), so callers degrade to
        'no health info' rather than surfacing a false alarm."""
        with self._lock:
            try:
                return self._client.shot_health(timeout_ms)
            except Exception:
                return None

    # -------- Images / status / sequence control --------

    def grab_imgs(self):
        """Grab all queued images from the server.

        Returns
        -------
        dict with:
            imgs : list of ndarray, each shape (H, W, n_imgs_per_seq)
            scan_ids : ndarray of int64
            seq_ids : ndarray of int64
        """
        empty = {
            'imgs': [],
            'scan_ids': np.array([], dtype=np.int64),
            'seq_ids': np.array([], dtype=np.int64),
        }
        if time.monotonic() < self._grab_imgs_cooldown_until:
            return empty
        try:
            with self._lock:
                raw = self._client.get_imgs(timeout_ms=30000)
        except Exception:
            self._grab_imgs_cooldown_until = (
                time.monotonic() + _GRAB_IMGS_COOLDOWN_S)
            return empty
        if raw is None or len(raw) == 0:
            return empty
        info = _process_imgs(raw)
        return {
            'imgs': info['imgs'],
            'scan_ids': np.array(info['scan_ids'], dtype=np.int64) if len(info['scan_ids']) > 0 else np.array([], dtype=np.int64),
            'seq_ids': np.array(info['seq_ids'], dtype=np.int64) if len(info['seq_ids']) > 0 else np.array([], dtype=np.int64),
        }

    def get_status(self):
        """Get experiment status: 0=Stopped, 1=Running, 2=Paused, 3=Unknown."""
        with self._lock:
            try:
                s = self._client.get_status()
            except Exception as e:
                now = time.monotonic()
                if now - self._last_get_status_warn > _STATUS_WARN_THROTTLE_S:
                    logger.warning('get_status failed: %s', e)
                    self._last_get_status_warn = now
                return 3
        return _STATUS_STR_TO_INT.get(s, 3)

    def abort_seq(self):
        with self._lock:
            try:
                self._client.abort_seq()
            except Exception:
                pass

    def pause_seq(self):
        with self._lock:
            try:
                self._client.pause_seq()
            except Exception:
                pass

    def start_seq(self):
        with self._lock:
            try:
                self._client.start_seq()
            except Exception:
                pass

    def cleanup(self):
        """Drop the wire client. Safe to call multiple times."""
        try:
            self._client = None
        except Exception:
            pass
