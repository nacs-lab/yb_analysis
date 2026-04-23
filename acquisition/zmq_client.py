"""ZMQ client wrapper with numpy conversion.

Wraps existing AnalysisUser.py / AnalysisClient.py with numpy-aware interfaces.
The existing Python ZMQ layer returns raw array.array('d', ...) blobs; this
module converts them into structured numpy arrays.
"""

import json
import sys
import os
import array
import threading
import numpy as np
import zmq

# Add the YbExpServer directory to sys.path so we can import AnalysisUser /
# AnalysisClient / ExptClient. yb_analysis/acquisition/ → repo_root →
# matlab_new/YbExpServer.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'matlab_new', 'YbExpServer')),
    # Legacy canonical location on the lab PC, kept as a last-resort fallback
    r'C:\msys64\home\Ybtweezer-PC2\projects\experiment-control\matlab_new\YbExpServer',
]
_EXPSERVER_DIR = next(
    (p for p in _CANDIDATES if os.path.isfile(os.path.join(p, 'AnalysisUser.py'))),
    None)
if _EXPSERVER_DIR is None:
    raise ImportError(
        "Could not locate AnalysisUser.py. Checked: " + ", ".join(_CANDIDATES))
if _EXPSERVER_DIR not in sys.path:
    sys.path.insert(0, _EXPSERVER_DIR)

from AnalysisUser import AnalysisUser  # noqa: E402


def _process_imgs(raw_data):
    """Parse the flat double array returned by AnalysisClient.get_imgs().

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

    if isinstance(raw_data, array.array):
        res = np.array(raw_data, dtype=np.float64)
    else:
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

    Wraps AnalysisUser with numpy conversions and a clean API.

    Parameters
    ----------
    url : str
        ZMQ server URL (default: tcp://127.0.0.1:8889).
    refresh_rate : float
        How often the background worker polls for new images (seconds).
    """

    def __init__(self, url='tcp://127.0.0.1:1312', refresh_rate=2.0):
        self._url = url
        self._au = AnalysisUser(url)
        self._au.set_refresh_rate(refresh_rate)
        # Patch AnalysisUser's worker to survive decode errors (ZMQ framing race)
        _orig_update_status = self._au._AnalysisUser__update_status
        def _safe_update_status():
            try:
                return _orig_update_status()
            except (UnicodeDecodeError, Exception):
                return self._au.SeqStatus.Unknown
        self._au._AnalysisUser__update_status = _safe_update_status
        # Dedicated queue-ops socket guarded by a lock (GUI reads concurrently)
        self._q_lock = threading.Lock()
        self._q_ctx = zmq.Context.instance()
        self._q_sock = None
        self._open_queue_sock()

    def _open_queue_sock(self):
        if self._q_sock is not None:
            try:
                self._q_sock.close(linger=0)
            except Exception:
                pass
        self._q_sock = self._q_ctx.socket(zmq.REQ)
        self._q_sock.setsockopt(zmq.LINGER, 0)
        self._q_sock.connect(self._url)

    def _q_call(self, *frames, reply='string', timeout_ms=2000):
        """Send a multi-frame REQ, receive a single reply. On failure the
        socket is recreated so callers can retry without leaving it in a
        broken REQ-state."""
        with self._q_lock:
            try:
                for i, f in enumerate(frames):
                    flags = zmq.SNDMORE if i < len(frames) - 1 else 0
                    if isinstance(f, str):
                        self._q_sock.send_string(f, flags)
                    else:
                        self._q_sock.send(bytes(f), flags)
                if self._q_sock.poll(timeout_ms) == 0:
                    self._open_queue_sock()
                    raise TimeoutError(f"{frames[0]}: no reply")
                if reply == 'string':
                    return self._q_sock.recv_string()
                elif reply == 'json':
                    return json.loads(self._q_sock.recv())
                elif reply == 'int':
                    return int.from_bytes(self._q_sock.recv(), 'little')
                else:
                    return self._q_sock.recv()
            except TimeoutError:
                raise
            except Exception:
                self._open_queue_sock()
                raise

    def ping(self, timeout_ms=500):
        try:
            return self._q_call('ping', timeout_ms=timeout_ms) == 'pong'
        except Exception:
            return False

    def submit_job(self, payload):
        return self._q_call('submit_job', payload, reply='int')

    def queue_list(self, timeout_ms=400):
        # Short timeout: called ~1 Hz from the UI; when the runner is absent
        # we want to fail fast rather than pile up outstanding REQs.
        return self._q_call('queue_list', reply='json', timeout_ms=timeout_ms)

    def queue_remove(self, job_id):
        return self._q_call(
            'queue_remove', int(job_id).to_bytes(8, 'little'))

    def queue_move(self, job_id, direction):
        return self._q_call(
            'queue_move', int(job_id).to_bytes(8, 'little'), direction)

    def shutdown_runner(self):
        try:
            return self._q_call('shutdown')
        except Exception:
            return ''

    def grab_imgs(self):
        """Grab all queued images from the server.

        Returns
        -------
        dict with:
            imgs : list of ndarray, each shape (H, W, n_imgs_per_seq)
            scan_ids : ndarray of int64
            seq_ids : ndarray of int64
        """
        raw_batches = self._au.grab_imgs()

        all_imgs = []
        all_scan_ids = []
        all_seq_ids = []

        for raw in raw_batches:
            if raw is None:
                continue
            # raw is an array.array('d', ...) from AnalysisClient.get_imgs()
            info = _process_imgs(raw)
            all_imgs.extend(info['imgs'])
            if len(info['scan_ids']) > 0:
                all_scan_ids.extend(info['scan_ids'])
                all_seq_ids.extend(info['seq_ids'])

        return {
            'imgs': all_imgs,
            'scan_ids': np.array(all_scan_ids, dtype=np.int64) if all_scan_ids else np.array([], dtype=np.int64),
            'seq_ids': np.array(all_seq_ids, dtype=np.int64) if all_seq_ids else np.array([], dtype=np.int64),
        }

    def get_status(self):
        """Get experiment status: 0=Stopped, 1=Running, 2=Paused, 3=Unknown."""
        return self._au.get_status()

    def abort_seq(self):
        """Send abort signal."""
        self._au.abort_seq()

    def pause_seq(self):
        """Send pause signal."""
        self._au.pause_seq()

    def start_seq(self):
        """Send start/continue signal."""
        self._au.start_seq()

    def set_refresh_rate(self, val):
        """Set background polling rate in seconds."""
        self._au.set_refresh_rate(val)

    def get_refresh_rate(self):
        return self._au.get_refresh_rate()

    def cleanup(self):
        """Stop the background worker thread."""
        self._au.stop_worker()
        try:
            self._q_sock.close(linger=0)
        except Exception:
            pass
