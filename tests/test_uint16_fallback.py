"""Fallback tests for the uint16 image-format probe (WP2 consumer side).

An in-process fake ZMQ ROUTER server stands in for a backend. Two flavours:
  * WITH the new verb -> the client locks onto uint16 after one probe and
    never issues the legacy verb again.
  * WITHOUT the new verb -> it replies to get_imgs_uint16 with an empty string,
    exactly as ExptServer.handle_msg's else branch does for any unknown verb
    (matlab_new AND pyctrl copies). The client must fall back to get_imgs
    cleanly, real data must flow, and subsequent calls must not wedge the REQ
    socket or re-probe.

The server counts requests per verb so we can assert the "no per-call double
requests after the probe" contract directly.
"""
import collections
import itertools
import threading

import numpy as np
import pytest
import zmq

from yb_analysis.acquisition.zmq_client import ZmqClient

_port_counter = itertools.count(15230)


def _next_url():
    return f"tcp://127.0.0.1:{next(_port_counter)}"


# --- wire encoders (single image (H, W, n_imgs) per sequence) --------------- #

def _encode_uint16(seqs):
    parts = [np.array([len(seqs)], dtype='<f8').tobytes()]
    for scan_id, seq_id, img in seqs:
        s1, s2, s3 = img.shape
        parts.append(np.array([scan_id, seq_id], dtype='<f8').tobytes())
        parts.append(np.array([s1, s2, s3], dtype='<f8').tobytes())
        parts.append(np.ascontiguousarray(img)
                     .astype('<u2').ravel(order='C').tobytes())
        parts.append(np.array([0.0], dtype='<f8').tobytes())
    return b''.join(parts)


def _encode_legacy_bytes(seqs):
    chunks = [np.array([len(seqs)], dtype=np.float64)]
    for scan_id, seq_id, img in seqs:
        s1, s2, s3 = img.shape
        chunks.append(np.array([scan_id, seq_id], dtype=np.float64))
        chunks.append(np.array([s1, s2, s3], dtype=np.float64))
        chunks.append(img.astype(np.float64).ravel(order='F'))
        chunks.append(np.array([0.0], dtype=np.float64))
    return np.concatenate(chunks).astype('<f8').tobytes()


class _FakeServer:
    """Minimal ROUTER server mimicking ExptServer's framing + else branch."""

    def __init__(self, url, support_uint16, seqs):
        self.url = url
        self.support_uint16 = support_uint16
        self.seqs = seqs
        self.counts = collections.Counter()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(url)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if self._sock.poll(50) == 0:
                continue
            try:
                addr = self._sock.recv()
                self._sock.recv()               # empty delimiter
                msg = self._sock.recv_string()
            except Exception:
                continue
            self.counts[msg] += 1
            if msg == 'ping':
                self._reply_str(addr, 'pong')
            elif msg == 'get_imgs':
                self._reply(addr, _encode_legacy_bytes(self.seqs))
            elif msg == 'get_imgs_uint16' and self.support_uint16:
                self._reply(addr, _encode_uint16(self.seqs))
            else:
                # Unknown/unsupported verb -> empty string, per ExptServer.
                self._reply_str(addr, '')

    def _reply(self, addr, data):
        self._sock.send(addr, zmq.SNDMORE)
        self._sock.send(b'', zmq.SNDMORE)
        self._sock.send(data)

    def _reply_str(self, addr, s):
        self._sock.send(addr, zmq.SNDMORE)
        self._sock.send(b'', zmq.SNDMORE)
        self._sock.send_string(s)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close(linger=0)
        self._ctx.destroy(linger=0)


@pytest.fixture
def make_server():
    servers = []

    def _make(support_uint16, seqs):
        srv = _FakeServer(_next_url(), support_uint16, seqs)
        servers.append(srv)
        return srv

    yield _make
    for s in servers:
        s.close()


def _sample_seqs():
    rng = np.random.default_rng(7)
    img0 = rng.integers(200, 500, size=(4, 5, 2), dtype=np.uint16)
    img1 = rng.integers(200, 500, size=(4, 5, 2), dtype=np.uint16)
    return [(9001, 1, img0), (9001, 2, img1)], (img0, img1)


def test_backend_with_new_verb_locks_uint16(make_server):
    seqs, (img0, img1) = _sample_seqs()
    srv = make_server(support_uint16=True, seqs=seqs)
    client = ZmqClient(url=srv.url)
    try:
        out = client.grab_imgs()
        assert len(out['imgs']) == 2
        assert out['imgs'][0].dtype == np.uint16
        np.testing.assert_array_equal(out['imgs'][0], img0)
        np.testing.assert_array_equal(out['imgs'][1], img1)
        assert list(out['scan_ids']) == [9001, 9001]
        assert list(out['seq_ids']) == [1, 2]
        assert client._img_format == 'uint16'

        # A second grab must issue exactly one more uint16 request, never legacy.
        out2 = client.grab_imgs()
        assert len(out2['imgs']) == 2
        assert srv.counts['get_imgs_uint16'] == 2
        assert srv.counts['get_imgs'] == 0
    finally:
        client.cleanup()


def test_backend_without_new_verb_falls_back(make_server):
    """MATLAB-style backend: no uint16 verb -> clean fallback, no wedge."""
    seqs, (img0, img1) = _sample_seqs()
    srv = make_server(support_uint16=False, seqs=seqs)
    client = ZmqClient(url=srv.url)
    try:
        out = client.grab_imgs()
        # Legacy data flowed through end to end.
        assert len(out['imgs']) == 2
        np.testing.assert_array_equal(
            out['imgs'][0].astype(np.float64), img0.astype(np.float64))
        assert list(out['scan_ids']) == [9001, 9001]
        assert client._img_format == 'legacy'
        # The probe issued get_imgs_uint16 once, then fetched via get_imgs.
        assert srv.counts['get_imgs_uint16'] == 1
        assert srv.counts['get_imgs'] == 1

        # Subsequent calls go straight to legacy: no re-probe, no EFSM wedge.
        out2 = client.grab_imgs()
        assert len(out2['imgs']) == 2
        out3 = client.grab_imgs()
        assert len(out3['imgs']) == 2
        assert srv.counts['get_imgs_uint16'] == 1   # never re-probed
        assert srv.counts['get_imgs'] == 3          # 1 (probe fallback) + 2
    finally:
        client.cleanup()


def test_probe_timeout_leaves_format_undecided():
    """No server up -> probe times out, format stays None, socket survives.

    A later call against a now-live uint16 backend must still lock onto
    uint16 (a transient startup timeout does not permanently cripple the
    fast path)."""
    url = _next_url()
    client = ZmqClient(url=url)
    try:
        out = client.grab_imgs()               # nothing listening
        assert out['imgs'] == []
        assert client._img_format is None      # undecided, will re-probe

        # Clear the post-failure cooldown so the next grab actually probes.
        client._grab_imgs_cooldown_until = 0.0

        seqs, (img0, _img1) = _sample_seqs()
        srv = _FakeServer(url, support_uint16=True, seqs=seqs)
        try:
            import time
            time.sleep(0.2)                    # let it bind
            out2 = client.grab_imgs()
            assert client._img_format == 'uint16'
            assert len(out2['imgs']) == 2
            np.testing.assert_array_equal(out2['imgs'][0], img0)
        finally:
            srv.close()
    finally:
        client.cleanup()
