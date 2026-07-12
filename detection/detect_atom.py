"""Atom detection via masked-intensity thresholding.

Port of YbDataAnalysis/AtomDetection/DetectAtom.m

The per-site masked sum is a sparse linear operator on the flattened frame:
``intensities = W . img.ravel()`` where ``W`` (M, H*W) holds each site's clipped
mask weights at that site's pixel indices. ``W`` depends only on the grid, the
mask, and the frame shape -- NOT on the pixel values -- so it is built once and
cached, and every frame is then a single sparse matvec (the pattern used by
pyctrl ``rearrange_runtime._Detector`` and ``spot_shape_model.detect_frame``).
``_detect_atom_loop`` keeps the original per-site loop as the parity reference.
"""

import threading
from collections import OrderedDict

import numpy as np


def _detect_atom_loop(img, grid_locations, thresholds, mask_mat):
    """Original per-site loop. Retained as the exact parity reference for the
    vectorised ``detect_atom`` (see tests/test_detect_atom.py). Not exported."""
    img = np.asarray(img, dtype=np.float64)
    grid_locations = np.asarray(grid_locations)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    mask_mat = np.asarray(mask_mat, dtype=np.float64)

    num_sites = grid_locations.shape[0]
    box_size = mask_mat.shape[0]
    half_box = box_size // 2
    H, W = img.shape

    intensities = np.zeros(num_sites, dtype=np.float64)

    for i in range(num_sites):
        y0 = int(round(grid_locations[i, 0]))
        x0 = int(round(grid_locations[i, 1]))

        y_min = max(y0 - half_box, 0)
        y_max = min(y0 + half_box, H - 1)
        x_min = max(x0 - half_box, 0)
        x_max = min(x0 + half_box, W - 1)

        region = img[y_min:y_max + 1, x_min:x_max + 1]

        # Handle edge clipping: use matching sub-mask
        my_min = y_min - (y0 - half_box)
        my_max = my_min + region.shape[0]
        mx_min = x_min - (x0 - half_box)
        mx_max = mx_min + region.shape[1]
        sub_mask = mask_mat[my_min:my_max, mx_min:mx_max]

        intensities[i] = np.sum(region * sub_mask)

    atom_status = intensities > thresholds
    return atom_status, intensities


# --- Cached sparse weight matrix -------------------------------------------
# W is keyed by (grid bytes/shape/dtype, mask bytes/shape/dtype, frame shape).
# The grid changes every ~50 shots (locate_atom_update), so W is rebuilt then;
# keep only the last few entries (grid + its img2 counterpart, plus a spare).
_W_CACHE_MAX = 4
_W_cache = OrderedDict()
_cache_lock = threading.Lock()


def _make_key(grid, mask_mat, H, W):
    return (grid.shape, grid.dtype.str, grid.tobytes(),
            mask_mat.shape, mask_mat.dtype.str, mask_mat.tobytes(),
            int(H), int(W))


def _build_W(grid, mask_mat, H, W, box_size, half_box):
    """Build the (M, H*W) CSR weight matrix that reproduces ``_detect_atom_loop``
    exactly: for each site the same int(round()) centre, the same edge clipping
    (``max(.,0)`` / ``min(.,H-1)``), and the same sub-mask offset arithmetic.
    Row-major (C-order) linear pixel index ``yy*W + xx`` so that ``W.dot(
    img.ravel())`` (numpy default C-order ravel) matches ``sum(region*sub_mask)``,
    and column-ascending accumulation matches the region's C-order traversal."""
    from scipy import sparse

    m = grid.shape[0]
    rows, cols, vals = [], [], []
    for i in range(m):
        y0 = int(round(grid[i, 0]))
        x0 = int(round(grid[i, 1]))

        y_min = max(y0 - half_box, 0)
        y_max = min(y0 + half_box, H - 1)
        x_min = max(x0 - half_box, 0)
        x_max = min(x0 + half_box, W - 1)

        rr = y_max - y_min + 1                 # region rows (as img slicing gives)
        cc = x_max - x_min + 1                 # region cols
        if rr <= 0 or cc <= 0:
            continue                            # box entirely off-frame -> intensity 0

        my_min = y_min - (y0 - half_box)        # 0-based offset into the mask
        mx_min = x_min - (x0 - half_box)
        # Available mask extent. For every config the loop evaluates without a
        # broadcast error (all production odd-box cases) this equals rr/cc, so W
        # is bit-identical to region*sub_mask; the clamp only guards the even-box
        # / far-clip configs where the loop itself raises (never compared).
        rr = min(rr, box_size - my_min)
        cc = min(cc, box_size - mx_min)
        if rr <= 0 or cc <= 0:
            continue

        for dy in range(rr):
            yy = y_min + dy
            base = yy * W + x_min
            mrow = my_min + dy
            for dx in range(cc):
                rows.append(i)
                cols.append(base + dx)
                vals.append(mask_mat[mrow, mx_min + dx])

    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp))),
        shape=(m, H * W))


def _get_W(grid, mask_mat, H, W):
    box_size = mask_mat.shape[0]
    half_box = box_size // 2
    key = _make_key(grid, mask_mat, H, W)
    with _cache_lock:
        Wm = _W_cache.get(key)
        if Wm is not None:
            _W_cache.move_to_end(key)
            return Wm
    # Build outside the lock so a rebuild (per grid change) never stalls a
    # concurrent detector thread; a rare double-build is idempotent.
    Wm = _build_W(grid, mask_mat, H, W, box_size, half_box)
    with _cache_lock:
        _W_cache[key] = Wm
        _W_cache.move_to_end(key)
        while len(_W_cache) > _W_CACHE_MAX:
            _W_cache.popitem(last=False)
    return Wm


def detect_atom(img, grid_locations, thresholds, mask_mat):
    """Detect atoms at tweezer sites using weighted intensity vs threshold.

    Parameters
    ----------
    img : ndarray, shape (H, W)
        Single camera image (float64).
    grid_locations : ndarray, shape (M, 2)
        Site positions as [y, x] per row.
    thresholds : ndarray, shape (M,)
        Per-site detection threshold.
    mask_mat : ndarray, shape (B, B)
        Weighting mask (e.g. Gaussian).

    Returns
    -------
    atom_status : ndarray, shape (M,), bool
        True where atom detected.
    intensities : ndarray, shape (M,), float64
        Weighted intensity at each site.
    """
    img = np.asarray(img, dtype=np.float64)
    grid_locations = np.asarray(grid_locations)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    mask_mat = np.asarray(mask_mat, dtype=np.float64)

    H, W = img.shape
    Wm = _get_W(grid_locations, mask_mat, H, W)
    intensities = Wm.dot(img.ravel())          # C-order ravel matches row-major W
    atom_status = intensities > thresholds
    return atom_status, intensities


def detect_atom_batch(images, grid_locations, thresholds, mask_mat):
    """Run detect_atom on a stack of images.

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
        Stack of camera images.
    grid_locations : ndarray, shape (M, 2)
    thresholds : ndarray, shape (M,)
    mask_mat : ndarray, shape (B, B)

    Returns
    -------
    all_status : ndarray, shape (N, M), bool
    all_intensities : ndarray, shape (N, M), float64
    """
    N = images.shape[0]
    M = grid_locations.shape[0]
    all_status = np.zeros((N, M), dtype=bool)
    all_intensities = np.zeros((N, M), dtype=np.float64)

    # The sparse W is cached across frames (same grid/mask/shape), so only the
    # first frame builds it; the rest are one matvec each.
    for k in range(N):
        all_status[k], all_intensities[k] = detect_atom(
            images[k], grid_locations, thresholds, mask_mat
        )

    return all_status, all_intensities
