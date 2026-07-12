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

# One-time image-format probe: the first grab_imgs tries the new
# "get_imgs_uint16" verb with this SHORT timeout. A backend that supports it
# replies (near-instantly) with a parseable header; a backend that does not
# (the retired MATLAB ExptServer, or a pre-WP1 pyctrl) replies with an empty
# string, so the short timeout only bites when the backend is entirely down --
# in which case we leave the format undecided and re-probe on the next call.
_IMG_PROBE_TIMEOUT_MS = 2000


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


def _looks_like_uint16_reply(raw):
    """Cheap header sanity check for a get_imgs_uint16 reply.

    True iff `raw` is at least the 8-byte nseqs header and that leading f8 is a
    finite, non-negative, integral, not-absurd sequence count. An empty-string
    reply (unsupported verb) is 0 bytes -> False. nseqs == 0 (a supported
    backend with nothing buffered) is a VALID reply -> True, so we can still
    lock onto the new format when the first batch happens to be empty.
    """
    if raw is None or len(raw) < 8:
        return False
    try:
        nseqs = float(np.frombuffer(raw, dtype='<f8', count=1, offset=0)[0])
    except Exception:
        return False
    return (np.isfinite(nseqs) and nseqs >= 0.0
            and nseqs == int(nseqs) and nseqs < 1e7)


def _process_imgs_uint16(raw_bytes):
    """Parse the uint16 image wire format returned by ExptClient.get_imgs_uint16().

    Wire format (little-endian, single ZMQ frame):
        [nseqs: f8]
        per sequence:
            [scan_id: f8] [seq_id: f8]
            per image:
                [s1: f8] [s2: f8] [s3: f8]
                <s1*s2*s3 pixels: u2, C-ORDER flat of the (s1, s2, s3) array>
            [0.0: f8]   (sequence separator)

    Disambiguation: at each image slot read one f8; 0.0 ends the sequence,
    anything else is s1 of the next image (s1 is never 0). The pixel blocks are
    2-byte elements, so the f8 fields that follow them are generally NOT
    8-byte-aligned; every field is therefore read with an explicit
    np.frombuffer(..., count=, offset=) rather than one big f8 view.

    Returns the SAME dict shape as _process_imgs -- imgs is a list of (H, W, n)
    arrays and the images are reconstructed as the identical logical arrays the
    old F-order path yields -- EXCEPT the pixel dtype stays uint16 (no upcast;
    downstream .astype handles it).
    """
    if raw_bytes is None or len(raw_bytes) < 8:
        return {'imgs': [], 'scan_ids': [], 'seq_ids': []}

    # A writable copy so the frombuffer views (and single-image stacks) are
    # writable, matching _process_imgs's arrays (views into a writable buffer).
    buf = bytearray(raw_bytes) if not isinstance(raw_bytes, bytearray) else raw_bytes
    n = len(buf)

    def _f8(off):
        return float(np.frombuffer(buf, dtype='<f8', count=1, offset=off)[0])

    num_seqs = int(_f8(0))
    if num_seqs == 0:
        return {'imgs': [], 'scan_ids': [], 'seq_ids': []}

    imgs = []
    scan_ids = []
    seq_ids = []

    off = 8
    seq_count = 0
    while seq_count < num_seqs and off + 16 <= n:
        scan_id = int(_f8(off))
        seq_id = int(_f8(off + 8))
        off += 16
        scan_ids.append(scan_id)
        seq_ids.append(seq_id)

        cur_img_stack = None
        # Read images until the 0.0 separator (or the buffer is exhausted).
        while off + 8 <= n:
            first = _f8(off)
            if first == 0.0:
                off += 8            # consume the sequence separator
                break
            # `first` is s1; need s2, s3 too.
            if off + 24 > n:
                off = n             # truncated header -> bail cleanly
                break
            s1 = int(first)
            s2 = int(_f8(off + 8))
            s3 = int(_f8(off + 16))
            off += 24
            n_pixels = s1 * s2 * s3
            if off + n_pixels * 2 > n:
                off = n             # truncated pixel block -> bail cleanly
                break
            # C-order flat of the (s1, s2, s3) array (the old format was F-order;
            # for identical logical images both paths recover the same array).
            img_data = np.frombuffer(
                buf, dtype='<u2', count=n_pixels, offset=off
            ).reshape(s1, s2, s3)
            off += n_pixels * 2
            if cur_img_stack is None:
                cur_img_stack = img_data
            else:
                cur_img_stack = np.concatenate([cur_img_stack, img_data], axis=2)

        if cur_img_stack is not None:
            imgs.append(cur_img_stack)
        seq_count += 1

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
        # Image wire format, decided once (lazily) on the first grab_imgs that
        # reaches a live backend: None = not yet probed, 'uint16' = backend
        # supports the get_imgs_uint16 verb, 'legacy' = it does not (MATLAB
        # backend / pre-WP1 pyctrl -> stay on the float64 get_imgs stream).
        # A pure timeout during the probe leaves this None so a later call
        # re-probes (the backend may just not be up yet); a definitive reply
        # (parseable => uint16, empty/garbage => legacy) commits it for good.
        self._img_format = None

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

    def set_background_enabled(self, enabled, timeout_ms=2000):
        """Global toggle for the background (calibration) lane."""
        with self._lock:
            return self._client.set_background_enabled(enabled, timeout_ms)

    def get_background_enabled(self, timeout_ms=1000):
        with self._lock:
            try:
                return self._client.get_background_enabled(timeout_ms)
            except Exception:
                return True

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
        t0 = time.monotonic()
        t_lock = t0
        try:
            with self._lock:
                t_lock = time.monotonic()
                # Only the wire call(s) run under the shared REQ lock; parsing
                # (CPU-bound) happens after release, as before.
                raw, fmt = self._grab_raw_locked()
        except Exception as e:
            # A silent swallow here once hid a full 30 s get_imgs timeout that
            # stalled the whole frame pipeline at a scan boundary — always say
            # what happened and where the time went (lock wait vs request).
            now = time.monotonic()
            logger.warning(
                'grab_imgs failed after %.1fs (lock wait %.1fs): %r — '
                'cooling down %.1fs', now - t0, t_lock - t0, e,
                _GRAB_IMGS_COOLDOWN_S)
            self._grab_imgs_cooldown_until = (
                time.monotonic() + _GRAB_IMGS_COOLDOWN_S)
            return empty
        if raw is None or len(raw) == 0:
            return empty
        if fmt == 'uint16':
            try:
                info = _process_imgs_uint16(raw)
            except Exception as e:
                # A malformed uint16 batch shouldn't wedge or flip us back to
                # legacy (the backend still speaks uint16) -- drop this batch.
                logger.warning(
                    'grab_imgs: could not parse uint16 reply (%d bytes): %r',
                    len(raw), e)
                return empty
        else:
            info = _process_imgs(raw)
        return {
            'imgs': info['imgs'],
            'scan_ids': np.array(info['scan_ids'], dtype=np.int64) if len(info['scan_ids']) > 0 else np.array([], dtype=np.int64),
            'seq_ids': np.array(info['seq_ids'], dtype=np.int64) if len(info['seq_ids']) > 0 else np.array([], dtype=np.int64),
        }

    def _grab_raw_locked(self):
        """Fetch the raw image reply. Caller MUST hold self._lock.

        Returns (raw, fmt) where fmt is 'uint16' or 'legacy'. Once probed,
        exactly ONE wire request is issued per call (no per-call double
        requests). The one-time probe (first call against a live backend) may
        issue a second request only to fetch this call's data via the legacy
        verb after discovering the new verb is unsupported -- the unsupported
        path consumes no server-side images, so that fallback loses nothing.
        """
        if self._img_format == 'uint16':
            return self._client.get_imgs_uint16(timeout_ms=30000), 'uint16'
        if self._img_format == 'legacy':
            return self._client.get_imgs(timeout_ms=30000), 'legacy'

        # First contact with a live backend: probe the new verb (short timeout).
        raw = self._client.get_imgs_uint16(timeout_ms=_IMG_PROBE_TIMEOUT_MS)
        if _looks_like_uint16_reply(raw):
            self._img_format = 'uint16'
            logger.info('grab_imgs: backend supports get_imgs_uint16; '
                        'using the uint16 image wire format')
            return raw, 'uint16'
        # Empty/garbage reply => verb unsupported (MATLAB backend or a pyctrl
        # predating the producer change). handle_msg's else branch replied
        # without touching the image deque, so fetch this call's batch via the
        # legacy verb now and lock onto legacy for the rest of the process.
        self._img_format = 'legacy'
        logger.info('grab_imgs: backend lacks get_imgs_uint16 (%d-byte reply); '
                    'using the legacy float64 image wire format',
                    0 if raw is None else len(raw))
        return self._client.get_imgs(timeout_ms=30000), 'legacy'

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
