"""Parity + micro-benchmark for the vectorised ``detect_atom``.

The vectorised (cached sparse-matvec) ``detect_atom`` must reproduce the original
per-site loop, kept as ``_detect_atom_loop``, EXACTLY: intensities to ~machine
precision (the sparse matvec reassociates the masked sum, so ~1e-12 relative, not
bit-identical) and atom_status bit-for-bit (thresholds are placed +-1 ADU off the
computed intensities, >> the reassociation error, so no tie can flip a bit).

Sweep: uint16 frames fed as both uint16 and float64; grids fully interior, at and
just over every edge/corner; odd AND even box sizes; 1 site and 1000 sites. The
old loop broadcast-errors on the even-box configs whose clipped region and
sub-mask disagree in shape (never a production case -- boxes are odd: 11 default,
9 for the notebook); those configs are skipped and the vectorised path is only
asserted to not raise there. Even boxes that the loop DOES evaluate (e.g. the
bottom/right edges, where the mask offset is 0) are compared normally.
"""

import time

import numpy as np
import pytest

from yb_analysis.detection.detect_atom import (
    detect_atom, detect_atom_batch, _detect_atom_loop, _W_cache)


def _gaussian_mask(box_size, sigma=2.0):
    """Matches acquisition.data_manager._gaussian_mask (centre-normalised)."""
    mask = np.zeros((box_size, box_size))
    center = box_size // 2
    for i in range(box_size):
        for j in range(box_size):
            mask[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2)
                                / (2 * sigma ** 2))
    return mask


def _frame_uint16(H, W, rng):
    """Realistic-range camera frame: pedestal + occasional bright sites, plus a
    few values above 2**15 so the uint16 input path is exercised where int16
    would have wrapped."""
    img = rng.integers(180, 260, size=(H, W), dtype=np.uint16)
    # sprinkle bright spots and a handful of >32768 values
    ys = rng.integers(0, H, size=50)
    xs = rng.integers(0, W, size=50)
    img[ys, xs] = rng.integers(30000, 65535, size=50, dtype=np.uint16)
    return img


def _thresholds_from(loop_int, rng):
    """Per-site cut placed +-1.0 ADU from the loop intensity: far above the
    ~1e-12-relative matvec reassociation, so atom_status is unambiguous, while
    still exercising both True and False sites."""
    sign = np.where(rng.random(loop_int.shape[0]) < 0.5, -1.0, 1.0)
    return loop_int + sign


def _compare(grid, box_size, frame_shape, rng, sigma=2.0):
    """Returns 'compared' or 'skipped_loop_raises'. Asserts parity when the loop
    evaluates; asserts the vectorised path never raises."""
    H, W = frame_shape
    mask = _gaussian_mask(box_size, sigma)
    img_u16 = _frame_uint16(H, W, rng)
    img_f64 = img_u16.astype(np.float64)

    try:
        loop_status, loop_int = _detect_atom_loop(img_f64, grid,
                                                  np.zeros(grid.shape[0]), mask)
    except ValueError:
        # Even-box clipped config the old loop cannot evaluate (region vs
        # sub-mask shape mismatch). The vectorised path must still not raise.
        detect_atom(img_f64, grid, np.zeros(grid.shape[0]), mask)
        detect_atom(img_u16, grid, np.zeros(grid.shape[0]), mask)
        return 'skipped_loop_raises'

    thr = _thresholds_from(loop_int, rng)
    loop_status = loop_int > thr

    for img in (img_u16, img_f64):
        status, inten = detect_atom(img, grid, thr, mask)
        assert inten.dtype == np.float64
        assert status.dtype == np.bool_
        assert np.allclose(inten, loop_int, rtol=1e-12, atol=0.0), (
            'intensity mismatch box=%d shape=%s max|d|=%g'
            % (box_size, frame_shape, np.max(np.abs(inten - loop_int))))
        assert np.array_equal(status, loop_status), (
            'atom_status mismatch box=%d shape=%s' % (box_size, frame_shape))
    return 'compared'


def _interior_grid(H, W, n_side, box_size):
    """n_side x n_side lattice of sites comfortably inside the frame."""
    half = box_size // 2 + 2
    ys = np.linspace(half, H - 1 - half, n_side)
    xs = np.linspace(half, W - 1 - half, n_side)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return np.column_stack([yy.ravel(), xx.ravel()]).astype(np.float64)


ODD_BOXES = (7, 9, 11)
EVEN_BOXES = (8, 10)


@pytest.mark.parametrize('box_size', ODD_BOXES + EVEN_BOXES)
def test_parity_interior(box_size):
    rng = np.random.default_rng(1234 + box_size)
    H, W = 90, 110
    grid = _interior_grid(H, W, 6, box_size)          # 36 interior sites
    res = _compare(grid, box_size, (H, W), rng)
    if box_size in ODD_BOXES:
        assert res == 'compared'


@pytest.mark.parametrize('box_size', ODD_BOXES + EVEN_BOXES)
def test_parity_edges_and_corners(box_size):
    """Sites AT and just OVER every edge and corner."""
    rng = np.random.default_rng(555 + box_size)
    H, W = 80, 96
    edge_y = [-2, -1, 0, 1, H - 2, H - 1, H, H + 1]
    edge_x = [-2, -1, 0, 1, W - 2, W - 1, W, W + 1]
    pts = []
    for y in edge_y:
        for x in edge_x:
            pts.append((y, x))
    # add some interior sites for a mixed grid
    pts += [(H // 2, W // 2), (H // 3, W // 4), (2 * H // 3, 3 * W // 4)]
    grid = np.asarray(pts, dtype=np.float64)
    res = _compare(grid, box_size, (H, W), rng)
    assert res in ('compared', 'skipped_loop_raises')


@pytest.mark.parametrize('box_size', ODD_BOXES)
def test_parity_edges_odd_box_always_compares(box_size):
    """Odd boxes (the production case) evaluate for every edge/corner config."""
    rng = np.random.default_rng(99 + box_size)
    H, W = 70, 88
    pts = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1),
           (-1, -1), (H, W), (-1, W), (H, -1),
           (0, W // 2), (H - 1, W // 2), (H // 2, 0), (H // 2, W - 1),
           (H // 2, W // 2)]
    grid = np.asarray(pts, dtype=np.float64)
    assert _compare(grid, box_size, (H, W), rng) == 'compared'


def test_even_box_bottom_right_edge_is_compared():
    """At least one even-box edge config must actually be COMPARED (not skipped),
    so the even-box sweep is meaningful. The bottom/right edges keep the mask
    offset at 0, so the old loop evaluates them."""
    rng = np.random.default_rng(4242)
    H, W = 60, 60
    box_size = 8
    # ONLY bottom/right-edge sites: there the mask offset stays 0, so the clipped
    # region and sub-mask agree in shape and the old loop evaluates. (An interior
    # even-box site has a 9-row region vs an 8-row mask and would raise.)
    grid = np.asarray([(H - 1, W - 1), (H - 1, W - 2), (H - 2, W - 1),
                       (H - 1, W - 3), (H - 3, W - 1)], dtype=np.float64)
    assert _compare(grid, box_size, (H, W), rng) == 'compared'


def test_parity_single_site():
    rng = np.random.default_rng(7)
    H, W = 40, 50
    grid = np.asarray([(20.0, 25.0)], dtype=np.float64)
    assert _compare(grid, 11, (H, W), rng) == 'compared'


def test_parity_1000_sites():
    rng = np.random.default_rng(2026)
    H, W = 260, 260
    grid = _interior_grid(H, W, 32, 11)[:1000]        # 1000 interior sites
    assert grid.shape[0] == 1000
    assert _compare(grid, 11, (H, W), rng) == 'compared'


def test_grid_change_rebuilds_and_caches():
    """A changed grid must rebuild W (not reuse a stale one); an unchanged grid
    reuses it (last-4 LRU)."""
    rng = np.random.default_rng(11)
    H, W = 80, 80
    mask = _gaussian_mask(11)
    img = _frame_uint16(H, W, rng).astype(np.float64)
    g1 = _interior_grid(H, W, 5, 11)
    g2 = g1 + np.array([2.0, -1.0])                    # shifted grid (locate_atom)
    _, i1 = detect_atom(img, g1, np.zeros(g1.shape[0]), mask)
    _, i2 = detect_atom(img, g2, np.zeros(g2.shape[0]), mask)
    _, i1b = detect_atom(img, g1, np.zeros(g1.shape[0]), mask)
    assert np.array_equal(i1, i1b)                     # cached path identical
    assert not np.allclose(i1, i2)                     # different grid -> different
    # cross-check the shifted grid against the loop
    _, li2 = _detect_atom_loop(img, g2, np.zeros(g2.shape[0]), mask)
    assert np.allclose(i2, li2, rtol=1e-12, atol=0.0)
    assert len(_W_cache) <= 4


def test_batch_matches_per_frame_loop():
    rng = np.random.default_rng(321)
    H, W = 90, 90
    mask = _gaussian_mask(9)
    grid = _interior_grid(H, W, 6, 9)
    thr = np.full(grid.shape[0], 1e9)                  # exercise all-False
    imgs = np.stack([_frame_uint16(H, W, rng) for _ in range(8)])
    st, it = detect_atom_batch(imgs, grid, thr, mask)
    ls, li = np.zeros_like(st), np.zeros_like(it)
    for k in range(imgs.shape[0]):
        ls[k], li[k] = _detect_atom_loop(imgs[k].astype(np.float64), grid, thr, mask)
    assert np.allclose(it, li, rtol=1e-12, atol=0.0)
    assert np.array_equal(st, ls)


def test_benchmark_old_vs_new(capsys):
    """Not asserted -- prints old-loop vs cached-matvec per-frame time at 1000
    sites on a realistic detection frame."""
    rng = np.random.default_rng(0)
    H, W = 2304, 4096                                  # full Orca sensor
    mask = _gaussian_mask(11)
    grid = _interior_grid(H, W, 32, 11)[:1000]
    thr = np.zeros(grid.shape[0])
    img = _frame_uint16(H, W, rng).astype(np.float64)

    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        _detect_atom_loop(img, grid, thr, mask)
    t_old = (time.perf_counter() - t0) / reps

    _W_cache.clear()
    t0 = time.perf_counter()
    detect_atom(img, grid, thr, mask)                  # cold: builds W
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(reps):
        detect_atom(img, grid, thr, mask)              # warm: matvec only
    t_new = (time.perf_counter() - t0) / reps

    with capsys.disabled():
        print('\n[detect_atom benchmark] 1000 sites, frame %dx%d' % (H, W))
        print('  old per-site loop : %.3f ms/frame' % (t_old * 1e3))
        print('  new (W cached)    : %.3f ms/frame' % (t_new * 1e3))
        print('  W build (once)    : %.3f ms' % (t_build * 1e3))
        print('  speedup           : %.1fx' % (t_old / max(t_new, 1e-9)))
