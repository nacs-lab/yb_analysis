"""Unit tests for the uint16 image wire format parser (WP2 consumer side).

Builds reference byte streams by hand and checks that
zmq_client._process_imgs_uint16 reconstructs the expected numpy arrays, and
that the SAME logical images encoded in the OLD float64/F-order format (through
_process_imgs) and the NEW uint16/C-order format (through _process_imgs_uint16)
come out identical up to dtype. Odd shapes (3x3, 5x7) are included on purpose:
their pixel blocks are an odd number of 2-byte elements, so the f8 fields that
follow land on non-8-byte-aligned offsets -- the exact case the offset-based
parser must handle.
"""
import numpy as np
import pytest

from yb_analysis.acquisition.zmq_client import (
    _process_imgs, _process_imgs_uint16, _looks_like_uint16_reply,
)


# --------------------------------------------------------------------------- #
# Hand-rolled reference encoders (mirror the wire spec exactly).
# `seqs` is a list of (scan_id, seq_id, [img, img, ...]); each img is a
# (s1, s2, s3) ndarray. Multiple images per sequence are separate wire blocks
# that the parser concatenates along axis 2.
# --------------------------------------------------------------------------- #

def _encode_uint16(seqs):
    parts = [np.array([len(seqs)], dtype='<f8').tobytes()]
    for scan_id, seq_id, imgs in seqs:
        parts.append(np.array([scan_id, seq_id], dtype='<f8').tobytes())
        for img in imgs:
            s1, s2, s3 = img.shape
            parts.append(np.array([s1, s2, s3], dtype='<f8').tobytes())
            parts.append(np.ascontiguousarray(img)
                         .astype('<u2').ravel(order='C').tobytes())
        parts.append(np.array([0.0], dtype='<f8').tobytes())
    return b''.join(parts)


def _encode_legacy_f64(seqs):
    """The old flat-float64/F-order stream _process_imgs consumes."""
    chunks = [np.array([len(seqs)], dtype=np.float64)]
    for scan_id, seq_id, imgs in seqs:
        chunks.append(np.array([scan_id, seq_id], dtype=np.float64))
        for img in imgs:
            s1, s2, s3 = img.shape
            chunks.append(np.array([s1, s2, s3], dtype=np.float64))
            chunks.append(img.astype(np.float64).ravel(order='F'))
        chunks.append(np.array([0.0], dtype=np.float64))
    return np.concatenate(chunks)


def _mk(shape, base, rng):
    """A deterministic uint16-valued image of the given (s1, s2, s3) shape."""
    return rng.integers(base, base + 300, size=shape, dtype=np.uint16)


# --------------------------------------------------------------------------- #
# Parser unit tests
# --------------------------------------------------------------------------- #

def test_empty_nseqs_zero():
    raw = np.array([0], dtype='<f8').tobytes()
    out = _process_imgs_uint16(raw)
    assert out['imgs'] == []
    assert list(out['scan_ids']) == []
    assert list(out['seq_ids']) == []


def test_empty_and_short_inputs():
    assert _process_imgs_uint16(None)['imgs'] == []
    assert _process_imgs_uint16(b'')['imgs'] == []
    assert _process_imgs_uint16(b'\x00\x01\x02')['imgs'] == []  # < 8 bytes


def test_single_seq_single_image():
    rng = np.random.default_rng(1)
    img = _mk((4, 6, 2), 200, rng)  # (H, W, n_imgs)
    raw = _encode_uint16([(111, 7, [img])])
    out = _process_imgs_uint16(raw)

    assert len(out['imgs']) == 1
    assert out['imgs'][0].shape == (4, 6, 2)
    assert out['imgs'][0].dtype == np.uint16
    np.testing.assert_array_equal(out['imgs'][0], img)
    assert list(out['scan_ids']) == [111]
    assert list(out['seq_ids']) == [7]


def test_multi_seq_multi_image():
    rng = np.random.default_rng(2)
    # seq 0: two images (H, W, 1) each -> concatenated to (H, W, 2)
    a0 = _mk((5, 3, 1), 200, rng)
    a1 = _mk((5, 3, 1), 500, rng)
    # seq 1: one image (H, W, 2)
    b0 = _mk((5, 3, 2), 800, rng)
    seqs = [(1000, 1, [a0, a1]), (1000, 2, [b0])]
    raw = _encode_uint16(seqs)
    out = _process_imgs_uint16(raw)

    assert len(out['imgs']) == 2
    np.testing.assert_array_equal(out['imgs'][0], np.concatenate([a0, a1], axis=2))
    np.testing.assert_array_equal(out['imgs'][1], b0)
    assert list(out['scan_ids']) == [1000, 1000]
    assert list(out['seq_ids']) == [1, 2]
    assert all(im.dtype == np.uint16 for im in out['imgs'])


@pytest.mark.parametrize('shape', [(3, 3, 1), (5, 7, 1), (3, 3, 2), (7, 5, 3)])
def test_odd_shapes_unaligned_offsets(shape):
    """Odd pixel counts push the trailing f8 fields off 8-byte alignment."""
    rng = np.random.default_rng(hash(shape) & 0xFFFF)
    img = _mk(shape, 210, rng)
    raw = _encode_uint16([(42, 9, [img])])
    out = _process_imgs_uint16(raw)
    assert len(out['imgs']) == 1
    assert out['imgs'][0].shape == shape
    np.testing.assert_array_equal(out['imgs'][0], img)


def test_large_scan_id_exact():
    """A 14-digit scan_id must survive the f8 round-trip exactly."""
    rng = np.random.default_rng(3)
    img = _mk((3, 3, 1), 200, rng)
    scan_id = 20260712120000
    out = _process_imgs_uint16(_encode_uint16([(scan_id, 5, [img])]))
    assert out['scan_ids'][0] == scan_id


# --------------------------------------------------------------------------- #
# Cross-check: OLD format vs NEW format yield identical images (up to dtype)
# --------------------------------------------------------------------------- #

def test_cross_check_old_vs_new_identical():
    rng = np.random.default_rng(4)
    a0 = _mk((6, 4, 1), 200, rng)
    a1 = _mk((6, 4, 1), 400, rng)
    b0 = _mk((5, 7, 2), 600, rng)
    seqs = [(123, 1, [a0, a1]), (123, 2, [b0])]

    old = _process_imgs(_encode_legacy_f64(seqs))
    new = _process_imgs_uint16(_encode_uint16(seqs))

    assert len(old['imgs']) == len(new['imgs']) == 2
    assert list(old['scan_ids']) == list(new['scan_ids'])
    assert list(old['seq_ids']) == list(new['seq_ids'])
    for o, ncur in zip(old['imgs'], new['imgs']):
        assert ncur.dtype == np.uint16
        assert o.shape == ncur.shape
        # Same logical values; old is float64, new is uint16.
        np.testing.assert_array_equal(o, ncur.astype(np.float64))


# --------------------------------------------------------------------------- #
# Header sanity check helper
# --------------------------------------------------------------------------- #

def test_looks_like_uint16_reply():
    assert _looks_like_uint16_reply(None) is False
    assert _looks_like_uint16_reply(b'') is False              # empty-verb reply
    assert _looks_like_uint16_reply(b'\x00\x01\x02') is False  # < 8 bytes
    assert _looks_like_uint16_reply(np.array([0], dtype='<f8').tobytes()) is True
    assert _looks_like_uint16_reply(np.array([3], dtype='<f8').tobytes()) is True
    # negative / non-integral / absurd counts are rejected
    assert _looks_like_uint16_reply(np.array([-1], dtype='<f8').tobytes()) is False
    assert _looks_like_uint16_reply(np.array([2.5], dtype='<f8').tobytes()) is False
    assert _looks_like_uint16_reply(np.array([1e9], dtype='<f8').tobytes()) is False
