"""ZMQ REQ client for the ExptServer protocol.

Vendored into yb_analysis so the package owns its protocol client and does
not reach into the (retired) matlab_new tree at runtime. This is the
backend-agnostic client: it speaks the same ZMQ verbs to whichever backend
hosts the ExptServer (pyctrl run loop or the legacy MATLAB SequenceRunner).
Pure protocol — depends only on array/json/zmq.
"""
import array
import json
import zmq


class ExptClient(object):
    # -------- Lifecycle --------

    def __init__(self, url):
        self.__url = url
        self.__ctx = zmq.Context()
        self.__sock = None
        self.recreate_sock()

    def __del__(self):
        try:
            self.__sock.close(linger=0)
        except Exception:
            pass
        try:
            self.__ctx.destroy(linger=0)
        except Exception:
            pass

    def recreate_sock(self):
        if self.__sock is not None:
            try:
                self.__sock.close(linger=0)
            except Exception:
                pass
        self.__sock = self.__ctx.socket(zmq.REQ)
        self.__sock.setsockopt(zmq.LINGER, 0)
        self.__sock.connect(self.__url)

    # -------- REQ helpers --------
    #
    # ZMQ REQ sockets have a strict SEND -> RECV -> SEND -> RECV state machine.
    # If poll() times out mid-request the socket is wedged — the next send
    # will throw zmq.ZMQError with EFSM. Every method below catches
    # timeouts/errors and recreates the socket so the next call starts clean.

    def _reset_on_failure(self):
        try:
            self.recreate_sock()
        except Exception:
            pass

    def _str_call(self, cmd: str, timeout_ms: int) -> str:
        try:
            self.__sock.send_string(cmd)
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError(f"{cmd}: no reply")
            return self.__sock.recv_string()
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    def _str_call_with_payload(self, cmd: str, payload_str: str, timeout_ms: int) -> str:
        try:
            self.__sock.send_string(cmd, zmq.SNDMORE)
            self.__sock.send_string(payload_str)
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError(f"{cmd}: no reply")
            return self.__sock.recv_string()
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    # -------- Liveness --------

    def ping(self, timeout_ms: int = 1000) -> bool:
        try:
            self.__sock.send_string("ping")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                return False
            return self.__sock.recv_string() == "pong"
        except Exception:
            self._reset_on_failure()
            return False

    # -------- Job queue --------

    def submit_job(self, payload, summary_json="", timeout_ms=2000):
        """Submit a job. `summary_json` is a JSON-encoded scan summary built
        by ybScanSummary; stored on the queue entry so the GUI can show
        scan variable / range / key parameters without decoding the MATLAB
        byte-stream payload. Pass "" to omit."""
        try:
            self.__sock.send_string("submit_job", zmq.SNDMORE)
            self.__sock.send(bytes(payload), zmq.SNDMORE)
            self.__sock.send_string(summary_json or "")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("submit_job: no reply")
            return int.from_bytes(self.__sock.recv(), byteorder='little')
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    def queue_list(self, timeout_ms: int = 2000) -> dict:
        try:
            self.__sock.send_string("queue_list")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("queue_list: no reply")
            return json.loads(self.__sock.recv())
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    def queue_remove(self, job_id: int, timeout_ms: int = 2000) -> str:
        try:
            self.__sock.send_string("queue_remove", zmq.SNDMORE)
            self.__sock.send(int(job_id).to_bytes(8, 'little'))
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("queue_remove: no reply")
            return self.__sock.recv_string()
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    def queue_move(self, job_id: int, direction: str, timeout_ms: int = 2000) -> str:
        try:
            self.__sock.send_string("queue_move", zmq.SNDMORE)
            self.__sock.send(int(job_id).to_bytes(8, 'little'), zmq.SNDMORE)
            self.__sock.send_string(direction)
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("queue_move: no reply")
            return self.__sock.recv_string()
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    # -------- Descriptor queue (Phase 3) --------

    def submit_scan_descriptor(self, descriptor_json: str, label: str = '',
                               timeout_ms: int = 2000) -> int:
        """Submit a scan descriptor (JSON string conforming to
        yb_analysis/scans/descriptor.schema.json). The SequenceRunner
        pops it between jobs, dispatch_descriptor.m builds a fresh
        ScanGroup, and the resulting job appears in the queue. Returns
        the descriptor's queue id."""
        try:
            self.__sock.send_string("submit_scan_descriptor", zmq.SNDMORE)
            self.__sock.send_string(descriptor_json or "", zmq.SNDMORE)
            self.__sock.send_string(label or "")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("submit_scan_descriptor: no reply")
            return int.from_bytes(self.__sock.recv(), byteorder='little')
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    def descriptor_remove(self, desc_id: int, timeout_ms: int = 2000) -> str:
        """Cancel a queued descriptor. Only valid while state=='queued'
        -- once the dispatcher pops it (state='building'), removal is no
        longer possible. Returns 'ok' on success, 'error: ...' otherwise."""
        try:
            self.__sock.send_string("descriptor_remove", zmq.SNDMORE)
            self.__sock.send(int(desc_id).to_bytes(8, 'little'))
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("descriptor_remove: no reply")
            return self.__sock.recv_string()
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    # -------- Sequence control / images --------

    def pause_seq(self, timeout_ms: int = 2000) -> str:
        return self._str_call("pause_seq", timeout_ms)

    def abort_seq(self, timeout_ms: int = 2000) -> str:
        return self._str_call("abort_seq", timeout_ms)

    def start_seq(self, timeout_ms: int = 2000) -> str:
        return self._str_call("start_seq", timeout_ms)

    def get_status(self, timeout_ms: int = 2000) -> str:
        return self._str_call("get_status", timeout_ms)

    def get_imgs(self, timeout_ms: int = 10000):
        """Pull the buffered image deque from the runner.

        Returns array.array('d') with format [num_seqs, <per-sequence blocks>]
        as documented in ExptServer.get_imgs / zmq_client._process_imgs.
        """
        try:
            self.__sock.send_string("get_imgs")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("get_imgs: no reply")
            return array.array('d', self.__sock.recv())
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    # -------- Camera --------

    def camera_init(self, roi, exposure_time=None, timeout_ms: int = 10000) -> str:
        """Submit a camera-init request. Note: the runner returns 'ok' as a
        queue ack; the camera is not yet connected. Caller should poll
        camera_status() until connected=True (see ZmqClient.camera_init for
        the standard wait-for-connected wrapper)."""
        payload = {'roi': list(roi)}
        if exposure_time is not None:
            payload['exposure_time'] = float(exposure_time)
        return self._str_call_with_payload(
            "camera_init", json.dumps(payload), timeout_ms)

    def camera_apply_settings(self, roi, exposure_time, timeout_ms: int = 5000) -> str:
        """Apply ROI + exposure atomically."""
        payload = {'roi': list(roi), 'exposure_time': float(exposure_time)}
        return self._str_call_with_payload(
            "camera_apply_settings", json.dumps(payload), timeout_ms)

    def camera_close(self, timeout_ms: int = 5000) -> str:
        return self._str_call("camera_close", timeout_ms)

    def camera_status(self, timeout_ms: int = 2000) -> dict:
        """Return camera state: {connected, roi, error}."""
        try:
            self.__sock.send_string("camera_status")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("camera_status: no reply")
            return json.loads(self.__sock.recv())
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    # -------- Dummy keep-alive --------

    def set_dummy_enabled(self, enabled: bool, timeout_ms: int = 2000) -> str:
        return self._str_call_with_payload(
            "set_dummy_enabled", "1" if enabled else "0", timeout_ms)

    def get_dummy_enabled(self, timeout_ms: int = 1000) -> bool:
        return self._str_call("get_dummy_enabled", timeout_ms) == "1"

    def set_dummy_mode(self, mode: str, timeout_ms: int = 2000) -> str:
        """Set the runner's idle-loop mode: 'off' | 'default' | 'last'."""
        return self._str_call_with_payload(
            "set_dummy_mode", str(mode), timeout_ms)

    def get_dummy_mode(self, timeout_ms: int = 1000) -> str:
        return self._str_call("get_dummy_mode", timeout_ms)

    def set_background_enabled(self, enabled: bool, timeout_ms: int = 2000) -> str:
        """Global toggle for the background (calibration) lane: when off the runner skips
        background scans (queued ones stay, untouched, and resume when re-enabled)."""
        return self._str_call_with_payload(
            "set_background_enabled", "1" if enabled else "0", timeout_ms)

    def get_background_enabled(self, timeout_ms: int = 1000) -> bool:
        return self._str_call("get_background_enabled", timeout_ms) == "1"

    def last_seq_status(self, timeout_ms: int = 1000) -> dict:
        """Return {available, name, file_id, captured_at, fallback_active, mode}."""
        try:
            self.__sock.send_string("last_seq_status")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("last_seq_status: no reply")
            return json.loads(self.__sock.recv())
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise

    def clear_last_seq_meta(self, timeout_ms: int = 2000) -> str:
        return self._str_call("clear_last_seq_meta", timeout_ms)

    def shot_health(self, timeout_ms: int = 1000) -> dict:
        """Return the backend's per-shot health rollup (pyctrl only):
        {total, last_message, last_kind, seconds_since_last, seconds_since_ok,
         scan_id, errors}. A MATLAB backend lacks the verb (replies empty) ->
        json.loads raises -> the caller treats it as 'no health info'."""
        try:
            self.__sock.send_string("shot_health")
            if self.__sock.poll(timeout_ms) == 0:
                self._reset_on_failure()
                raise TimeoutError("shot_health: no reply")
            return json.loads(self.__sock.recv())
        except TimeoutError:
            raise
        except Exception:
            self._reset_on_failure()
            raise
