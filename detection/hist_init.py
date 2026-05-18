"""Offline initialization of tweezer grid locations and detection thresholds.

Port of matlab_new/AtomDetection/YbHistInit.m + LocateAtom.m (initial mode).

Two ways to use this module:

1. One-shot (legacy): ``hist_init(scan_dir, num_tweezers=N)`` runs the full
   pipeline — load → average → detect → edit → threshold → save → plot.

2. Composable (recommended for iteration in a notebook): call the public
   functions individually so you can re-edit, re-fit, or re-detect without
   reloading thousands of images:

       ctx        = load_scan_context(scan_dir)
       avg_image  = compute_avg_image(ctx, n_avg=200)
       mask_mat   = make_mask(box_size=9, sigma=2.0)
       grid       = detect_grid(avg_image, num_tweezers=112)
       grid       = edit_grid(avg_image, grid)
       images     = load_threshold_images(ctx, n_thresh=2000)
       hist_data, thresholds, gauss_fits, infidelities = compute_thresholds(
           images, grid, mask_mat, num_bins=50)
       plot_grid_overlay(avg_image, grid, infidelities=infidelities)
       plot_histograms(hist_data, thresholds, gauss_fits, infidelities)
       save_calibration(ctx['scan_dir'], grid, thresholds, infidelities,
                        gauss_fits, hist_data)
"""

import os
import glob
import logging

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, label, center_of_mass
from scipy.io import savemat

from yb_analysis.analysis.load_data import load_scan_from_path, load_images
from yb_analysis.detection.dynamical_threshold import dynamical_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Scan context and image loading
# ---------------------------------------------------------------------------

def load_scan_context(scan_dir):
    """Load scan metadata and the indices of the first image of each sequence.

    Returns a dict with everything needed by the later steps so they don't
    each have to re-parse the scan.

    Returns
    -------
    dict with:
        scan_dir : str
        data_path : str — path to the .h5 (or .mat) image file
        config : dict — Scan config
        num_images : int — images per sequence
        total_frames : int
        num_seq : int
        all_first_indices : list[int] — frame index of image 1 of each seq
    """
    scan_data = load_scan_from_path(scan_dir)
    config = scan_data['Scan']
    data_path = scan_data['path']
    imgs_shape = scan_data.get('imgs_shape')

    if imgs_shape is None or imgs_shape[0] == 0:
        raise ValueError(f'No images found in {data_path}')

    total_frames = imgs_shape[0]
    num_images = int(np.asarray(config.get('NumImages', 1)).flat[0])
    all_first_indices = list(range(0, total_frames, num_images))

    return {
        'scan_dir': scan_dir,
        'data_path': data_path,
        'config': config,
        'num_images': num_images,
        'total_frames': total_frames,
        'num_seq': len(all_first_indices),
        'all_first_indices': all_first_indices,
    }


def compute_avg_image(ctx, n_avg=200):
    """Average the first ``n_avg`` sequence-first images for spot detection.

    Parameters
    ----------
    ctx : dict — from load_scan_context
    n_avg : int — number of images to average (clamped to available)

    Returns
    -------
    avg_image : ndarray (H, W)
    """
    n = min(ctx['num_seq'], n_avg)
    indices = ctx['all_first_indices'][:n]
    logger.info('Averaging %d of %d sequence-first images', n, ctx['num_seq'])
    imgs = load_images(ctx['data_path'], indices).astype(np.float64)
    avg = imgs.mean(axis=0)
    del imgs
    return avg


def load_threshold_images(ctx, n_thresh=2000):
    """Load sequence-first images for threshold computation.

    Returns
    -------
    images : ndarray (N, H, W) float64
    """
    n = min(ctx['num_seq'], n_thresh)
    indices = ctx['all_first_indices'][:n]
    logger.info('Loading %d images for threshold computation', n)
    return load_images(ctx['data_path'], indices).astype(np.float64)


# ---------------------------------------------------------------------------
# Step 2 — Mask
# ---------------------------------------------------------------------------

def make_mask(box_size=9, sigma=2.0):
    """Create the Gaussian weighting mask used for masked-intensity readout."""
    mask = np.zeros((box_size, box_size))
    center = box_size // 2
    mask[center, center] = 1.0
    return gaussian_filter(mask, sigma)


# ---------------------------------------------------------------------------
# Sub-pixel peak refinement (used by detect_grid and the interactive editor)
# ---------------------------------------------------------------------------

def _refine_subpixel(field, y_int, x_int):
    """Parabolic sub-pixel refinement at integer peak (y_int, x_int) of `field`.

    Returns (y_float, x_float). Falls back to the integer value at edges or if
    the curvature is degenerate.
    """
    H, W = field.shape
    if 0 < y_int < H - 1:
        v_m, v_0, v_p = field[y_int - 1, x_int], field[y_int, x_int], field[y_int + 1, x_int]
        denom = v_m - 2 * v_0 + v_p
        dy = 0.5 * (v_m - v_p) / denom if abs(denom) > 1e-12 else 0.0
        dy = max(-0.5, min(0.5, dy))
    else:
        dy = 0.0
    if 0 < x_int < W - 1:
        v_m, v_0, v_p = field[y_int, x_int - 1], field[y_int, x_int], field[y_int, x_int + 1]
        denom = v_m - 2 * v_0 + v_p
        dx = 0.5 * (v_m - v_p) / denom if abs(denom) > 1e-12 else 0.0
        dx = max(-0.5, min(0.5, dx))
    else:
        dx = 0.0
    return y_int + dy, x_int + dx


def _snap_to_peak(field, y, x, window=8):
    """Find the local max of ``field`` within ±``window`` of (y, x), sub-pixel."""
    H, W = field.shape
    y0, x0 = int(round(y)), int(round(x))
    ymin = max(0, y0 - window); ymax = min(H, y0 + window + 1)
    xmin = max(0, x0 - window); xmax = min(W, x0 + window + 1)
    sub = field[ymin:ymax, xmin:xmax]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return _refine_subpixel(field, ymin + iy, xmin + ix)


# ---------------------------------------------------------------------------
# CSV registration — load tweezer geometry, fit affine to image
# ---------------------------------------------------------------------------

def find_tweezer_csv(scan_dir, pattern='*coords.csv'):
    """Return path to the tweezer-coords CSV in ``scan_dir`` if exactly one
    matches ``pattern``. Returns None if no match. Raises if multiple.
    """
    matches = sorted(glob.glob(os.path.join(scan_dir, pattern)))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f'Multiple files match {pattern} in {scan_dir}: '
            f'{[os.path.basename(m) for m in matches]}. '
            'Pass an explicit `csv_pattern` to disambiguate.'
        )
    return matches[0]


def load_tweezer_csv(scan_dir_or_path, pattern='*coords.csv'):
    """Load a tweezer-coordinates CSV by header name (extra columns OK).

    The CSV must have a header row containing ``index``, ``x_um``, and
    ``y_um`` (case- and whitespace-tolerant). Any other columns are ignored.

    Parameters
    ----------
    scan_dir_or_path : str
        Either a scan directory (in which case ``pattern`` is used to find
        the file) or a direct path to the CSV.

    Returns
    -------
    csv_indices : ndarray (N,) int
    csv_xy_um   : ndarray (N, 2) — columns are [x_um, y_um]
    csv_path    : str
    """
    if os.path.isdir(scan_dir_or_path):
        path = find_tweezer_csv(scan_dir_or_path, pattern)
        if path is None:
            raise FileNotFoundError(
                f'No file matching {pattern} in {scan_dir_or_path}'
            )
    else:
        path = scan_dir_or_path

    with open(path, 'r') as f:
        header_line = f.readline().strip()
    cols = [c.strip().lower() for c in header_line.split(',')]
    required = ('index', 'x_um', 'y_um')
    try:
        idx_col = cols.index('index')
        x_col = cols.index('x_um')
        y_col = cols.index('y_um')
    except ValueError:
        raise ValueError(
            f"CSV {path} missing one of required columns {required}. "
            f"Got: {cols}"
        )

    data = np.loadtxt(path, delimiter=',', skiprows=1,
                      usecols=(idx_col, x_col, y_col))
    data = np.atleast_2d(data)
    csv_indices = data[:, 0].astype(int)
    csv_xy_um = data[:, 1:3].astype(np.float64)
    return csv_indices, csv_xy_um, path


def fit_affine_csv_to_image(csv_xy_um_anchors, image_yx_px_anchors):
    """Fit a 2x3 affine `A` such that [y_px, x_px]^T = A @ [x_um, y_um, 1]^T.

    Parameters
    ----------
    csv_xy_um_anchors : ndarray (N, 2) — [x_um, y_um] of anchor CSV points
    image_yx_px_anchors : ndarray (N, 2) — [y_px, x_px] of anchor image points

    Requires N >= 3. Uses least squares for N >= 4.

    Returns
    -------
    A : ndarray (2, 3)
    residuals_px : ndarray (N,) — Euclidean per-anchor fit residual in pixels
    """
    csv = np.asarray(csv_xy_um_anchors, dtype=np.float64).reshape(-1, 2)
    img = np.asarray(image_yx_px_anchors, dtype=np.float64).reshape(-1, 2)
    if csv.shape != img.shape or csv.shape[0] < 3:
        raise ValueError(
            f'Need >= 3 anchor pairs (got csv={csv.shape}, image={img.shape}).'
        )
    # Build design matrix M (N, 3) with [x_um, y_um, 1] rows
    M = np.column_stack([csv, np.ones(csv.shape[0])])
    # Solve M @ row = img column for each output coordinate
    # row_y (3,) such that M @ row_y ≈ img[:, 0]  (y_px)
    # row_x (3,) such that M @ row_x ≈ img[:, 1]  (x_px)
    row_y, *_ = np.linalg.lstsq(M, img[:, 0], rcond=None)
    row_x, *_ = np.linalg.lstsq(M, img[:, 1], rcond=None)
    A = np.vstack([row_y, row_x])  # (2, 3)

    predicted = (M @ A.T)  # (N, 2) → [y, x]
    residuals_px = np.linalg.norm(predicted - img, axis=1)
    return A, residuals_px


def project_csv(csv_xy_um, A):
    """Apply affine ``A`` (2,3) to all CSV points.

    Returns ndarray (N, 2) of [y_px, x_px] in image pixels.
    """
    csv = np.asarray(csv_xy_um, dtype=np.float64).reshape(-1, 2)
    M = np.column_stack([csv, np.ones(csv.shape[0])])
    return M @ A.T


# ---------------------------------------------------------------------------
# Step 3 — Spot detection
# ---------------------------------------------------------------------------

def detect_grid(avg_image, num_tweezers, spot_sigma=2.0, min_distance=10,
                sort=True, refine_subpixel=True):
    """Find the top ``num_tweezers`` brightest spots via LoG filtering.

    Computes the -LoG (Laplacian-of-Gaussian) response of the image, finds
    all local maxima, and returns the ``num_tweezers`` with the strongest
    response.  No threshold tuning required — the expected count is the
    only knob.

    Parameters
    ----------
    avg_image : ndarray (H, W)
    num_tweezers : int
        Expected number of tweezer spots.
    spot_sigma : float
        Expected spot radius in pixels (LoG kernel sigma).
    min_distance : int
        Minimum pixel separation between peaks.
    sort : bool
        If True, sort the grid into rows/columns (default).
    refine_subpixel : bool
        If True, refine each detected peak to sub-pixel precision via parabolic
        fit on the LoG response (recommended).

    Returns
    -------
    positions : ndarray (N, 2) — [y, x] pairs, N <= num_tweezers.
    """
    img = avg_image.astype(np.float64)
    neg_log = -gaussian_laplace(img, sigma=spot_sigma)

    n_candidates = num_tweezers * 3

    try:
        from skimage.feature import peak_local_max
        coords = peak_local_max(neg_log, min_distance=min_distance,
                                num_peaks=n_candidates)
    except ImportError:
        med = np.median(neg_log)
        mad = np.median(np.abs(neg_log - med))
        thr = med + 3.0 * 1.4826 * mad
        binary = neg_log >= thr
        labeled, n_feat = label(binary)
        if n_feat == 0:
            return np.zeros((0, 2), dtype=np.float64)
        centroids = center_of_mass(neg_log, labeled, range(1, n_feat + 1))
        coords = np.array(centroids, dtype=np.float64).round().astype(int)

    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    responses = neg_log[coords[:, 0], coords[:, 1]]
    top_idx = np.argsort(responses)[::-1][:num_tweezers]
    positions = coords[top_idx].astype(np.float64)

    if refine_subpixel:
        positions = np.array(
            [_refine_subpixel(neg_log, int(p[0]), int(p[1])) for p in positions],
            dtype=np.float64,
        )

    if sort:
        positions = sort_grid(positions)
    return positions


def sort_grid(positions):
    """Sort grid locations by X-clusters, then Y within each cluster.

    Port of LocateAtom.m lines 65-96.
    """
    if len(positions) <= 1:
        return positions.copy()

    X = positions[:, 1]
    order = np.argsort(X)
    sorted_pos = positions[order]
    sorted_X = X[order]

    dX = np.diff(sorted_X)
    if len(dX) == 0 or dX.max() == dX.min():
        return sorted_pos[np.argsort(sorted_pos[:, 0])]

    thr = (dX.min() + dX.max()) / 2.0
    breaks = np.where(dX > thr)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(sorted_pos)]])

    result = []
    for s, e in zip(starts, ends):
        chunk = sorted_pos[s:e]
        chunk = chunk[np.argsort(chunk[:, 0])]
        result.append(chunk)
    return np.vstack(result)


def sort_grid_rotated(positions, rotation_deg=None, spacing=None,
                      return_info=False, verbose=False):
    """Column-major, bottom-up sort for a (possibly rotated) lattice.

    Order: index 0 is the bottom-left site (smallest column, largest image
    y within that column), then up the leftmost column (decreasing image y),
    then step right to the next column, repeat.

    Algorithm:
      1. Estimate lattice rotation θ from nearest-neighbor bearings (folded
         by 4× angle doubling so the four orthogonal lattice directions
         collapse to one circular mean).
      2. Estimate lattice spacing as the median NN distance.
      3. Project sites into the rotated frame (x_r along the more-horizontal
         lattice axis, y_r along the more-vertical one).
      4. Snap each site's x_r to an integer column index by rounding
         ``(x_r - x_r.min()) / spacing``. Robust to missing columns.
      5. Sort by (column index asc, y_r desc).

    Parameters
    ----------
    positions : ndarray (N, 2)
        Sites in image coords [y, x] (image y increases downward).
    rotation_deg : float or None
        Lattice rotation, CCW from image +x, in degrees. If None, auto-
        estimated. For grids close to 45° tilt the auto-pick can flip row
        vs column by 90° — pass manually in that case.
    spacing : float or None
        Lattice pitch in px. If None, taken as the median NN distance.
    return_info : bool
        If True, return (sorted_pos, info_dict) with keys
        ``rotation_deg``, ``spacing``, ``n_columns``, ``column_sizes``.
    verbose : bool
        Print the diagnostics.
    """
    from scipy.spatial import cKDTree
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if len(pos) <= 1:
        info = {'rotation_deg': 0.0, 'spacing': 0.0,
                'n_columns': len(pos), 'column_sizes': np.array([len(pos)])}
        return (pos.copy(), info) if return_info else pos.copy()

    xy = pos[:, [1, 0]]
    tree = cKDTree(xy)
    kq = min(5, len(pos))
    nn_d, nn_i = tree.query(xy, k=kq)

    if rotation_deg is None:
        src = np.repeat(np.arange(len(pos)), kq - 1)
        dst = nn_i[:, 1:].ravel()
        v = xy[dst] - xy[src]
        norms = np.hypot(v[:, 0], v[:, 1])
        keep = norms > 0
        ang = np.arctan2(v[keep, 1], v[keep, 0])
        mean4 = np.arctan2(np.sin(4 * ang).mean(), np.cos(4 * ang).mean())
        rotation_deg = float(np.degrees(mean4 / 4.0))

    if spacing is None:
        # NN distance (column 1 of nn_d, since column 0 is self → 0)
        spacing = float(np.median(nn_d[:, 1]))

    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    y_im, x_im = pos[:, 0], pos[:, 1]
    x_r =  x_im * c + y_im * s
    y_r = -x_im * s + y_im * c

    col_idx = np.round((x_r - x_r.min()) / spacing).astype(np.int64)
    # Lexsort: primary key (last) = column index ascending,
    # secondary key (first) = -y_r ascending (== y_r descending → image-bottom first)
    order = np.lexsort((-y_r, col_idx))
    sorted_pos = pos[order]

    uniq, counts = np.unique(col_idx, return_counts=True)
    info = {
        'rotation_deg': float(rotation_deg),
        'spacing': float(spacing),
        'n_columns': int(len(uniq)),
        'column_sizes': counts,
        'order': order,
    }
    if verbose:
        logger.info(
            'sort_grid_rotated: rotation=%+.2f°, spacing=%.2f px, '
            'columns=%d (sizes min/median/max = %d/%d/%d)',
            info['rotation_deg'], info['spacing'], info['n_columns'],
            int(counts.min()), int(np.median(counts)), int(counts.max()),
        )

    if return_info:
        return sorted_pos, info
    return sorted_pos


# ---------------------------------------------------------------------------
# Step 4 — Interactive grid editor
# ---------------------------------------------------------------------------

class _GridEditor:
    """Internal state for the matplotlib-based interactive grid editor."""

    HELP = ("Left-click: add  |  Right-click: toggle alive/dead nearest  "
            "|  d: delete near cursor  |  u: undo  |  Enter/q: done")

    def __init__(self, avg_image, initial_grid, *, spot_sigma, snap, snap_window,
                 delete_radius, infidelities, is_detected, is_csv,
                 csv_indices, is_alive, box_size):
        from scipy.ndimage import gaussian_laplace
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RangeSlider, CheckButtons
        from matplotlib.patches import Rectangle
        from matplotlib import cm, colors

        self._plt = plt
        self._Rectangle = Rectangle

        self.avg = avg_image.astype(np.float64)
        self.neg_log = -gaussian_laplace(self.avg, sigma=spot_sigma)

        self.initial = [list(map(float, p)) for p in np.asarray(initial_grid).tolist()]
        self.grid = [list(p) for p in self.initial]
        self.site_origin = list(range(len(self.initial)))  # -1 = newly added

        # Per-row metadata (tombstoning support)
        n_init = len(self.initial)
        self.csv_indices = (
            list(map(int, np.asarray(csv_indices).tolist()))
            if csv_indices is not None else [-1] * n_init
        )
        if len(self.csv_indices) != n_init:
            raise ValueError('csv_indices length mismatch')
        self.initial_csv_indices = list(self.csv_indices)
        self.is_alive = (
            list(map(bool, np.asarray(is_alive).tolist()))
            if is_alive is not None else [True] * n_init
        )
        if len(self.is_alive) != n_init:
            raise ValueError('is_alive length mismatch')
        self.initial_is_alive = list(self.is_alive)

        self.history = []  # list[(grid, csv_idx, is_alive, origin)]

        self.snap = snap
        self.snap_window = snap_window
        self.delete_radius = delete_radius
        self.box_size = box_size
        self.cursor = (None, None)

        self.is_detected = (np.asarray(is_detected, dtype=bool)
                            if is_detected is not None else None)
        if self.is_detected is not None and len(self.is_detected) != len(self.initial):
            raise ValueError(
                f'is_detected length {len(self.is_detected)} does not match '
                f'initial_grid length {len(self.initial)}'
            )
        self.is_csv = (np.asarray(is_csv, dtype=bool)
                       if is_csv is not None else None)
        if self.is_csv is not None and len(self.is_csv) != len(self.initial):
            raise ValueError(
                f'is_csv length {len(self.is_csv)} does not match '
                f'initial_grid length {len(self.initial)}'
            )

        self.infidelities = (np.asarray(infidelities, dtype=float)
                             if infidelities is not None else None)
        if self.infidelities is not None and self.infidelities.size:
            finite = np.isfinite(self.infidelities)
            vmax_inf = (max(np.nanpercentile(self.infidelities[finite], 95), 1e-3)
                        if finite.any() else 1.0)
            self._inf_norm = colors.Normalize(vmin=0, vmax=vmax_inf)
            self._inf_cmap = cm.RdYlGn_r
        else:
            self._inf_norm = None
            self._inf_cmap = None

        # ---- Layout ----
        self.fig = plt.figure(figsize=(14, 10))
        self.ax_image = self.fig.add_axes([0.06, 0.18, 0.82, 0.78])
        self.ax_status = self.fig.add_axes([0.06, 0.135, 0.82, 0.03]); self.ax_status.axis('off')
        self.ax_slider = self.fig.add_axes([0.10, 0.075, 0.50, 0.025])
        self.ax_btn_done = self.fig.add_axes([0.64, 0.07, 0.07, 0.04])
        self.ax_btn_reset = self.fig.add_axes([0.72, 0.07, 0.07, 0.04])
        self.ax_check = self.fig.add_axes([0.80, 0.06, 0.07, 0.06]); self.ax_check.axis('off')

        vmin, vmax = np.percentile(self.avg, [1, 99.5])
        self._im = self.ax_image.imshow(self.avg, cmap='gray',
                                        vmin=vmin, vmax=vmax, aspect='equal')
        self.ax_image.set_title(self.HELP, fontsize=10)

        v_lo, v_hi = float(self.avg.min()), float(self.avg.max())
        self.slider = RangeSlider(self.ax_slider, 'Contrast', v_lo, v_hi,
                                  valinit=(vmin, vmax))
        self.slider.on_changed(self._on_contrast)

        self.btn_done = Button(self.ax_btn_done, 'Done')
        self.btn_done.on_clicked(self._on_done)
        self.btn_reset = Button(self.ax_btn_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)
        self.checks = CheckButtons(self.ax_check, ['Snap'], [self.snap])
        self.checks.on_clicked(self._on_check)

        self.artists = []
        self.status_text = self.ax_status.text(
            0.5, 0.5, '', ha='center', va='center', fontsize=10,
            transform=self.ax_status.transAxes,
        )

        # Optional infidelity colorbar (second-pass mode)
        if self._inf_norm is not None:
            sm = cm.ScalarMappable(norm=self._inf_norm, cmap=self._inf_cmap)
            sm.set_array([])
            self._cbar = self.fig.colorbar(sm, ax=self.ax_image,
                                           fraction=0.025, pad=0.01)
            self._cbar.set_label('Infidelity')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._redraw()

    # ---- state ops ----
    def _push_history(self):
        self.history.append((
            [list(p) for p in self.grid],
            list(self.csv_indices),
            list(self.is_alive),
            list(self.site_origin),
        ))
        if len(self.history) > 200:
            self.history.pop(0)

    def _undo(self):
        if not self.history:
            return
        grid, csv_idx, alive, origin = self.history.pop()
        self.grid = [list(p) for p in grid]
        self.csv_indices = list(csv_idx)
        self.is_alive = list(alive)
        self.site_origin = list(origin)
        self._redraw()

    def _add(self, y, x):
        if self.snap:
            y, x = _snap_to_peak(self.neg_log, y, x, window=self.snap_window)
        self._push_history()
        self.grid.append([float(y), float(x)])
        self.csv_indices.append(-1)   # new clicks are non-CSV
        self.is_alive.append(True)
        self.site_origin.append(-1)
        self._redraw()

    def _toggle_nearest(self, y, x):
        """Right-click: toggle alive↔dead on nearest site within delete_radius."""
        if not self.grid:
            return
        ds = [((p[0] - y) ** 2 + (p[1] - x) ** 2) ** 0.5 for p in self.grid]
        idx = int(np.argmin(ds))
        if ds[idx] > self.delete_radius:
            return
        self._push_history()
        self.is_alive[idx] = not self.is_alive[idx]
        self._redraw()

    def _delete_nearest_alive(self, y, x):
        """`d` key: kill nearest *alive* site. No-op on already-dead sites."""
        alive_idx = [i for i, a in enumerate(self.is_alive) if a]
        if not alive_idx:
            return
        ds = [((self.grid[i][0] - y) ** 2 + (self.grid[i][1] - x) ** 2) ** 0.5
              for i in alive_idx]
        k = int(np.argmin(ds))
        if ds[k] > self.delete_radius:
            return
        self._push_history()
        self.is_alive[alive_idx[k]] = False
        self._redraw()

    # ---- event handlers ----
    def _on_click(self, event):
        if event.inaxes != self.ax_image or event.xdata is None:
            return
        if event.button == 1:
            self._add(event.ydata, event.xdata)
        elif event.button == 3:
            self._toggle_nearest(event.ydata, event.xdata)

    def _on_motion(self, event):
        if event.inaxes == self.ax_image and event.xdata is not None:
            self.cursor = (event.ydata, event.xdata)

    def _on_key(self, event):
        if event.key == 'd':
            y, x = self.cursor
            if y is not None:
                self._delete_nearest_alive(y, x)
        elif event.key == 'u':
            self._undo()
        elif event.key in ('enter',):
            self._plt.close(self.fig)

    def _on_done(self, _event):
        self._plt.close(self.fig)

    def _on_reset(self, _event):
        self._push_history()
        self.grid = [list(p) for p in self.initial]
        self.csv_indices = list(self.initial_csv_indices)
        self.is_alive = list(self.initial_is_alive)
        self.site_origin = list(range(len(self.initial)))
        self._redraw()

    def _on_check(self, _label):
        self.snap = not self.snap
        self._update_status()

    def _on_contrast(self, vals):
        self._im.set_clim(vals[0], vals[1])
        self.fig.canvas.draw_idle()

    # ---- drawing ----
    def _color_for(self, idx):
        # Dead sites trump everything
        if not self.is_alive[idx]:
            return 'dimgray'
        origin = self.site_origin[idx]
        # Infidelity coloring takes priority (existing original site)
        if origin >= 0 and self._inf_norm is not None and origin < len(self.infidelities):
            v = self.infidelities[origin]
            if np.isfinite(v):
                return self._inf_cmap(self._inf_norm(v))
            return 'magenta'
        # Categorical coloring for original sites (no infidelity yet)
        if origin >= 0 and self.is_detected is not None and origin < len(self.is_detected):
            is_det = bool(self.is_detected[origin])
            is_csv_site = (bool(self.is_csv[origin])
                           if self.is_csv is not None and origin < len(self.is_csv)
                           else True)
            if not is_csv_site:
                return 'cyan'           # ghost: detected, not in CSV
            return 'lime' if is_det else 'orange'  # csv-detected vs csv-inferred
        # Newly added or default
        return 'red'

    def _redraw(self):
        for rect, txt in self.artists:
            rect.remove()
            txt.remove()
        self.artists.clear()
        half = self.box_size / 2.0
        for i, (y, x) in enumerate(self.grid):
            alive = self.is_alive[i]
            ls = '-' if alive else ':'
            r = self._Rectangle((x - half, y - half), self.box_size, self.box_size,
                                linewidth=1.4, edgecolor=self._color_for(i),
                                facecolor='none', linestyle=ls)
            self.ax_image.add_patch(r)
            label = self._label_for(i)
            text_color = 'yellow' if alive else 'gray'
            text_alpha = 1.0 if alive else 0.6
            t = self.ax_image.text(x, y, label, color=text_color,
                                   fontsize=7, ha='center', va='center',
                                   alpha=text_alpha)
            self.artists.append((r, t))
        self._update_status()
        self.fig.canvas.draw_idle()

    def _label_for(self, idx):
        """Show CSV index for CSV sites, else row position."""
        ci = self.csv_indices[idx]
        return str(ci) if ci > 0 else f'+{idx + 1}'

    def _update_status(self):
        n = len(self.grid)
        n_alive = sum(self.is_alive)
        n_dead = n - n_alive
        msg = (f'{n_alive} alive | {n_dead} deleted | '
               f'snap: {"on" if self.snap else "off"} | undo: {len(self.history)}')
        # Restrict downstream stats to alive original sites
        kept = [self.site_origin[i] for i in range(n)
                if self.is_alive[i] and self.site_origin[i] >= 0]
        if self.infidelities is not None and self._inf_norm is not None and kept:
            msg += f'  |  mean infidelity: {np.nanmean(self.infidelities[kept]):.2e}'
        elif self.is_detected is not None and kept:
            kept_arr = np.asarray(kept)
            if self.is_csv is not None:
                csv_mask = self.is_csv[kept_arr]
                n_csv_det = int((csv_mask & self.is_detected[kept_arr]).sum())
                n_csv_inf = int((csv_mask & ~self.is_detected[kept_arr]).sum())
                n_ghosts = int((~csv_mask).sum())
                msg += (f'  |  csv-detected: {n_csv_det}  '
                        f'|  csv-inferred: {n_csv_inf}  |  ghosts: {n_ghosts}')
            else:
                det = int(self.is_detected[kept_arr].sum())
                msg += f'  |  detected: {det}/{len(kept)}  |  inferred: {len(kept) - det}/{len(kept)}'
        self.status_text.set_text(msg)

    def run(self):
        self._plt.show(block=True)
        # Build per-row metadata. site_origin maps current row → original row;
        # values >= 0 index into the input is_csv / is_detected arrays.
        n = len(self.grid)
        is_csv_out = np.zeros(n, dtype=bool)
        is_detected_out = np.zeros(n, dtype=bool)
        for i, origin in enumerate(self.site_origin):
            if origin >= 0:
                if self.is_csv is not None and origin < len(self.is_csv):
                    is_csv_out[i] = bool(self.is_csv[origin])
                if self.is_detected is not None and origin < len(self.is_detected):
                    is_detected_out[i] = bool(self.is_detected[origin])
            else:
                # newly clicked sites: not from CSV, but anchored to a peak via snap
                is_detected_out[i] = True
        return {
            'grid': np.array(self.grid, dtype=np.float64),
            'csv_indices': np.array(self.csv_indices, dtype=int),
            'is_alive': np.array(self.is_alive, dtype=bool),
            'is_csv': is_csv_out,
            'is_detected': is_detected_out,
        }


def edit_grid(avg_image, initial_grid, *, spot_sigma=2.0,
              snap=True, snap_window=8, delete_radius=12,
              infidelities=None, is_detected=None, is_csv=None,
              csv_indices=None, is_alive=None,
              box_size=10, sort=False):
    """Launch matplotlib editor for tweezer sites with tombstone deletion.

    Mouse:
        Left-click   — add a new site (snapped to nearest LoG peak if ``snap``)
        Right-click  — toggle alive/dead on the nearest site (within ``delete_radius``)
    Keyboard (cursor must be over the image):
        d            — mark nearest *alive* site as dead
        u            — undo last add/toggle/reset
        Enter        — finish (same as the Done button)
    Buttons:
        Done         — finish editing
        Reset        — restore the initial grid (revives all dead sites,
                       drops user-clicked additions)
        Snap         — toggle click-to-peak snapping
        Contrast     — adjust display range

    Tombstoned (dead) sites stay in the returned arrays so that downstream
    consumers can preserve identity (e.g. CSV index → row mapping).

    Parameters
    ----------
    avg_image : ndarray (H, W)
    initial_grid : ndarray (N, 2) — [y, x] starting points
    csv_indices : ndarray (N,) int or None
        CSV index per row (-1 for non-CSV / ghost / user-added). If None,
        all rows are treated as -1 and labeled by row position.
    is_alive : ndarray (N,) bool or None
        Initial alive/dead state. Defaults to all True.
    infidelities, is_detected, is_csv :
        Coloring hints — see source. If supplied, lengths must match
        ``initial_grid``.
    sort : bool
        If True, the returned ``grid`` is filtered to alive sites only and
        sorted into rows/columns. The returned ``csv_indices`` and other
        metadata are filtered to match. Default False — preserve identity.

    Returns
    -------
    dict with:
        grid          : ndarray (M, 2) [y, x]
        csv_indices   : ndarray (M,) int   — -1 for non-CSV
        is_alive      : ndarray (M,) bool
        is_csv        : ndarray (M,) bool
        is_detected   : ndarray (M,) bool

    Requires a live matplotlib backend (``%matplotlib qt`` or ``widget``).
    """
    import matplotlib
    backend = matplotlib.get_backend()
    if 'inline' in backend.lower():
        raise RuntimeError(
            'Interactive grid editing requires a live matplotlib backend.\n'
            "Run '%matplotlib qt' (or '%matplotlib widget' if ipympl is "
            'installed) before calling edit_grid().'
        )

    initial = np.asarray(initial_grid, dtype=np.float64).reshape(-1, 2)
    editor = _GridEditor(
        avg_image, initial,
        spot_sigma=spot_sigma, snap=snap, snap_window=snap_window,
        delete_radius=delete_radius, infidelities=infidelities,
        is_detected=is_detected, is_csv=is_csv,
        csv_indices=csv_indices, is_alive=is_alive, box_size=box_size,
    )
    result = editor.run()

    if len(result['grid']) == 0:
        raise ValueError('No grid locations selected. Cannot compute thresholds.')

    if sort:
        # Drop dead rows, sort the alive grid, and align metadata to the sort order.
        alive = result['is_alive']
        kept = result['grid'][alive]
        kept_csv = result['csv_indices'][alive]
        kept_iscsv = result['is_csv'][alive]
        kept_isdet = result['is_detected'][alive]
        sorted_grid = sort_grid(kept)
        # Reindex: for each row of sorted_grid, find its index in kept
        order = []
        used = np.zeros(len(kept), dtype=bool)
        for row in sorted_grid:
            d = np.hypot(kept[:, 0] - row[0], kept[:, 1] - row[1])
            d[used] = np.inf
            j = int(np.argmin(d))
            order.append(j)
            used[j] = True
        order = np.asarray(order)
        result = {
            'grid': sorted_grid,
            'csv_indices': kept_csv[order],
            'is_alive': np.ones(len(sorted_grid), dtype=bool),
            'is_csv': kept_iscsv[order],
            'is_detected': kept_isdet[order],
        }
    return result


# ---------------------------------------------------------------------------
# Step 4b — CSV anchor picker (interactive) + register_csv_grid
# ---------------------------------------------------------------------------

class _AnchorPicker:
    """Side-by-side picker for CSV→image anchor correspondences."""

    HELP = ("Click a detected spot in the IMAGE (left, cyan ring), then click "
            "the matching indexed point in the CSV panel (right). Repeat "
            "until you have enough anchors, then press Done.")

    def __init__(self, avg_image, csv_indices, csv_xy_um, *, min_anchors,
                 detected_spots=None, neg_log_field=None,
                 snap_window=8, match_tolerance=5.0):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._plt = plt
        self.avg = avg_image.astype(np.float64)
        self.csv_indices = np.asarray(csv_indices)
        self.csv_xy_um = np.asarray(csv_xy_um, dtype=np.float64)
        self.min_anchors = int(min_anchors)
        self.snap_window = snap_window
        self.match_tolerance = float(match_tolerance)
        self.neg_log = neg_log_field
        self.detected_spots = (np.asarray(detected_spots, dtype=np.float64)
                               if detected_spots is not None else None)

        # State
        self.anchors = []  # list of (csv_index, (y_px, x_px))
        self.pending_image = None  # (y_px, x_px) waiting for CSV pair
        self.pending_csv = None    # csv_index waiting for image pair

        # ---- Layout: image left, CSV right ----
        self.fig = plt.figure(figsize=(16, 9))
        self.ax_img = self.fig.add_axes([0.04, 0.18, 0.45, 0.78])
        self.ax_csv = self.fig.add_axes([0.53, 0.18, 0.42, 0.78])
        self.ax_status = self.fig.add_axes([0.04, 0.10, 0.92, 0.05])
        self.ax_status.axis('off')
        self.ax_btn_done = self.fig.add_axes([0.42, 0.03, 0.08, 0.05])
        self.ax_btn_reset = self.fig.add_axes([0.51, 0.03, 0.08, 0.05])

        # Image panel
        vmin, vmax = np.percentile(self.avg, [1, 99.5])
        self.ax_img.imshow(self.avg, cmap='gray', vmin=vmin, vmax=vmax,
                           aspect='equal')
        if self.detected_spots is not None and len(self.detected_spots):
            self.ax_img.scatter(self.detected_spots[:, 1],
                                self.detected_spots[:, 0],
                                s=40, facecolor='none', edgecolor='cyan',
                                linewidths=0.8)
            self.ax_img.set_title(
                f'Average image — click a cyan ring '
                f'({len(self.detected_spots)} auto-detected spots)',
                fontsize=10)
        else:
            self.ax_img.set_title('Average image — click an anchor spot',
                                  fontsize=10)

        # CSV panel — camera-like (y inverted so layout matches image)
        self.ax_csv.scatter(self.csv_xy_um[:, 0], self.csv_xy_um[:, 1],
                            s=20, c='steelblue')
        for idx, (xu, yu) in zip(self.csv_indices, self.csv_xy_um):
            self.ax_csv.text(xu, yu, str(int(idx)), fontsize=7,
                             ha='center', va='center', color='black')
        self.ax_csv.set_aspect('equal')
        self.ax_csv.invert_yaxis()
        self.ax_csv.set_xlabel('x_um')
        self.ax_csv.set_ylabel('y_um')
        self.ax_csv.set_title(
            f'CSV layout ({len(self.csv_indices)} sites) — '
            f'click the matching index here', fontsize=10)

        # Buttons
        self.btn_done = Button(self.ax_btn_done, 'Done')
        self.btn_done.on_clicked(self._on_done)
        self.btn_reset = Button(self.ax_btn_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        # Status text + dynamic markers
        self.status_text = self.ax_status.text(
            0.5, 0.5, '', ha='center', va='center', fontsize=10,
            transform=self.ax_status.transAxes,
        )
        self._img_markers = []
        self._csv_markers = []

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.suptitle(self.HELP, fontsize=10, y=0.99)
        self._update_status()

    def _on_click(self, event):
        if event.button != 1 or event.xdata is None:
            return
        if event.inaxes is self.ax_img:
            self._click_image(event.ydata, event.xdata)
        elif event.inaxes is self.ax_csv:
            self._click_csv(event.xdata, event.ydata)

    def _click_image(self, y_px, x_px):
        # Prefer snapping to the nearest auto-detected spot (more precise
        # than a fresh LoG max search). Fall back to LoG peak if no
        # detected spot is within match_tolerance.
        if self.detected_spots is not None and len(self.detected_spots):
            ds = np.hypot(self.detected_spots[:, 0] - y_px,
                          self.detected_spots[:, 1] - x_px)
            j = int(np.argmin(ds))
            if ds[j] <= self.match_tolerance:
                y_px, x_px = float(self.detected_spots[j, 0]), \
                             float(self.detected_spots[j, 1])
            elif self.neg_log is not None:
                y_px, x_px = _snap_to_peak(self.neg_log, y_px, x_px,
                                           window=self.snap_window)
        elif self.neg_log is not None:
            y_px, x_px = _snap_to_peak(self.neg_log, y_px, x_px,
                                       window=self.snap_window)
        if self.pending_csv is not None:
            # Pair with pending CSV index
            self.anchors.append((int(self.pending_csv),
                                 (float(y_px), float(x_px))))
            self.pending_csv = None
            self._refresh_markers()
        else:
            self.pending_image = (float(y_px), float(x_px))
            self._refresh_markers()
        self._update_status()

    def _click_csv(self, xu, yu):
        # Snap to nearest CSV index
        d = np.hypot(self.csv_xy_um[:, 0] - xu, self.csv_xy_um[:, 1] - yu)
        idx = int(np.argmin(d))
        csv_index = int(self.csv_indices[idx])
        if self.pending_image is not None:
            self.anchors.append((csv_index, self.pending_image))
            self.pending_image = None
            self._refresh_markers()
        else:
            self.pending_csv = csv_index
            self._refresh_markers()
        self._update_status()

    def _refresh_markers(self):
        for art in self._img_markers + self._csv_markers:
            try:
                art.remove()
            except Exception:
                pass
        self._img_markers.clear()
        self._csv_markers.clear()

        # Confirmed anchors — green
        for idx, (yp, xp) in self.anchors:
            m = self.ax_img.plot(xp, yp, 'o', mfc='none', mec='lime',
                                 mew=2, ms=14)[0]
            t = self.ax_img.text(xp + 7, yp - 7, str(idx),
                                 color='lime', fontsize=9, fontweight='bold')
            self._img_markers.extend([m, t])
            row = np.where(self.csv_indices == idx)[0]
            if len(row):
                xu, yu = self.csv_xy_um[row[0]]
                cm = self.ax_csv.plot(xu, yu, 'o', mfc='none', mec='lime',
                                      mew=2, ms=14)[0]
                self._csv_markers.append(cm)

        # Pending markers — yellow ring
        if self.pending_image is not None:
            yp, xp = self.pending_image
            m = self.ax_img.plot(xp, yp, 'o', mfc='none', mec='yellow',
                                 mew=2, ms=14)[0]
            self._img_markers.append(m)
        if self.pending_csv is not None:
            row = np.where(self.csv_indices == self.pending_csv)[0]
            if len(row):
                xu, yu = self.csv_xy_um[row[0]]
                cm = self.ax_csv.plot(xu, yu, 'o', mfc='none', mec='yellow',
                                      mew=2, ms=14)[0]
                self._csv_markers.append(cm)

        self.fig.canvas.draw_idle()

    def _update_status(self):
        n = len(self.anchors)
        need = max(0, self.min_anchors - n)
        if self.pending_image is not None:
            cue = '— now click the matching CSV index'
        elif self.pending_csv is not None:
            cue = f'— now click the matching spot in the IMAGE (CSV idx {self.pending_csv})'
        else:
            cue = ''
        msg = f'{n} anchors picked (need ≥ {self.min_anchors}, {need} to go) {cue}'
        self.status_text.set_text(msg)

    def _on_done(self, _event):
        if len(self.anchors) < self.min_anchors:
            return  # Done is a no-op until min reached
        self._plt.close(self.fig)

    def _on_reset(self, _event):
        self.anchors.clear()
        self.pending_image = None
        self.pending_csv = None
        self._refresh_markers()
        self._update_status()

    def run(self):
        self._plt.show(block=True)
        return list(self.anchors)


def pick_csv_anchors(avg_image, csv_indices, csv_xy_um, *,
                     min_anchors=3, detected_spots=None,
                     neg_log_field=None, snap_window=8,
                     match_tolerance=5.0):
    """Open a side-by-side picker for CSV→image anchor correspondences.

    Image-click snap behavior (in priority order):
      1. If ``detected_spots`` given: snap to the nearest detected spot
         within ``match_tolerance`` px of the click. This is the precise
         path — recommended.
      2. Else if ``neg_log_field`` given: snap to the nearest LoG max
         within ``snap_window`` (looser, may snap to noise peaks).
      3. Else: use the raw click position.

    Returns
    -------
    anchors : list of (csv_index, (y_px, x_px))
    """
    import matplotlib
    if 'inline' in matplotlib.get_backend().lower():
        raise RuntimeError(
            'Anchor picker requires a live matplotlib backend.\n'
            "Run '%matplotlib qt' before calling pick_csv_anchors()."
        )
    picker = _AnchorPicker(avg_image, csv_indices, csv_xy_um,
                           min_anchors=min_anchors,
                           detected_spots=detected_spots,
                           neg_log_field=neg_log_field,
                           snap_window=snap_window,
                           match_tolerance=match_tolerance)
    anchors = picker.run()
    if len(anchors) < min_anchors:
        raise ValueError(
            f'Got only {len(anchors)} anchors, need >= {min_anchors}.'
        )
    return anchors


def register_csv_grid(avg_image, scan_dir, *, detected_spots=None,
                      include_ghosts=True,
                      spot_sigma=2.0, snap_window=8, match_tolerance=5.0,
                      min_anchors=3, csv_pattern='*coords.csv'):
    """Register a tweezer-coords CSV onto an averaged atom image.

    Returns ``None`` if no CSV matching ``csv_pattern`` is found in
    ``scan_dir`` — caller should fall back to the auto-detect-only flow.

    Algorithm:
      1. Load CSV (header-driven; extra columns ignored).
      2. If ``detected_spots`` is None, run :func:`detect_grid` to find
         bright peaks. Otherwise use the supplied spots.
      3. Open the side-by-side picker; user clicks ≥ ``min_anchors``
         correspondences (image clicks snap to nearest detected spot).
      4. Fit a 2×3 affine from CSV (µm) → image (px) anchor pairs.
      5. Project ALL CSV points through the affine.
      6. For each projected position, find the nearest detected spot.
         If within ``match_tolerance`` px, mark ``is_detected=True`` and
         use the detected spot's position. Otherwise keep the projected
         position and mark ``is_detected=False`` (CSV-inferred only).
      7. If ``include_ghosts`` is True (default), any detected spot that
         has NO CSV neighbor within ``match_tolerance`` is appended to
         the grid as a "ghost" trap. These show up in the editor as cyan
         and can be deleted manually if you don't want them.

    Parameters
    ----------
    detected_spots : ndarray (M, 2) [y, x] or None
        Output of :func:`detect_grid` (sub-pixel peak positions). Used
        both as click targets in the picker and as ground truth for the
        ``is_detected`` flag. If None, computed internally.
    include_ghosts : bool
        If True (default), append detected spots that don't match any
        CSV position to the grid. Set False to keep only CSV sites.
    match_tolerance : float
        Max distance (px) from a projected CSV position to a detected
        spot for it to count as "detected". Should be smaller than half
        the tweezer pitch in pixels.

    Returns
    -------
    None  (no CSV in scan_dir)
    or dict with:
        grid           : ndarray (N+G, 2) [y, x] in image px
                         (N CSV sites + G ghost traps if include_ghosts)
        csv_indices    : ndarray (N+G,) — ghost rows are -1
        csv_xy_um      : ndarray (N+G, 2) — ghost rows are NaN
        is_detected    : ndarray (N+G,) bool
        is_csv         : ndarray (N+G,) bool — True for CSV rows, False for ghosts
        A              : ndarray (2, 3) affine matrix
        residuals_px   : ndarray (n_anchors,)
        match_distances: ndarray (N+G,) — px to nearest detected spot;
                         NaN for ghosts
        csv_path       : str
    """
    csv_path = find_tweezer_csv(scan_dir, csv_pattern)
    if csv_path is None:
        logger.info('No CSV matching %s in %s — skipping registration.',
                    csv_pattern, scan_dir)
        return None

    csv_indices, csv_xy_um, _ = load_tweezer_csv(csv_path)
    logger.info('Loaded %d tweezers from %s', len(csv_indices),
                os.path.basename(csv_path))

    if detected_spots is None:
        logger.info('No detected_spots passed — running detect_grid internally.')
        detected_spots = detect_grid(avg_image, len(csv_indices),
                                     spot_sigma=spot_sigma, sort=False)
    detected_spots = np.asarray(detected_spots, dtype=np.float64).reshape(-1, 2)

    neg_log = -gaussian_laplace(avg_image.astype(np.float64), sigma=spot_sigma)

    anchors = pick_csv_anchors(avg_image, csv_indices, csv_xy_um,
                               min_anchors=min_anchors,
                               detected_spots=detected_spots,
                               neg_log_field=neg_log,
                               snap_window=snap_window,
                               match_tolerance=match_tolerance)

    # Build anchor arrays in CSV-order for the fit
    idx_to_xy = {int(i): xy for i, xy in zip(csv_indices, csv_xy_um)}
    anchor_csv = np.array([idx_to_xy[i] for i, _ in anchors])
    anchor_img = np.array([yx for _, yx in anchors])

    A, residuals_px = fit_affine_csv_to_image(anchor_csv, anchor_img)
    projected = project_csv(csv_xy_um, A)  # (N, 2) [y, x]

    # For each projected CSV position, find the nearest detected spot.
    # If within match_tolerance, use the detected position and mark detected.
    grid = projected.copy()
    is_detected = np.zeros(len(projected), dtype=bool)
    match_distances = np.full(len(projected), np.inf)

    if len(detected_spots):
        for i, (y, x) in enumerate(projected):
            ds = np.hypot(detected_spots[:, 0] - y, detected_spots[:, 1] - x)
            j = int(np.argmin(ds))
            match_distances[i] = ds[j]
            if ds[j] <= match_tolerance:
                grid[i] = detected_spots[j]
                is_detected[i] = True

    # ---- Ghost traps: detected spots with no CSV neighbor ----
    is_csv = np.ones(len(grid), dtype=bool)
    n_ghosts = 0
    if include_ghosts and len(detected_spots):
        spot_match = np.zeros(len(detected_spots), dtype=bool)
        for j, (sy, sx) in enumerate(detected_spots):
            ds = np.hypot(projected[:, 0] - sy, projected[:, 1] - sx)
            if ds.min() <= match_tolerance:
                spot_match[j] = True
        ghosts = detected_spots[~spot_match]
        n_ghosts = len(ghosts)
        if n_ghosts:
            grid = np.vstack([grid, ghosts])
            is_detected = np.concatenate(
                [is_detected, np.ones(n_ghosts, dtype=bool)]
            )
            is_csv = np.concatenate(
                [is_csv, np.zeros(n_ghosts, dtype=bool)]
            )
            csv_indices = np.concatenate(
                [csv_indices, np.full(n_ghosts, -1, dtype=int)]
            )
            csv_xy_um = np.vstack(
                [csv_xy_um, np.full((n_ghosts, 2), np.nan)]
            )
            match_distances = np.concatenate(
                [match_distances, np.full(n_ghosts, np.nan)]
            )

    n_csv_detected = int((is_csv & is_detected).sum())
    n_csv_inferred = int((is_csv & ~is_detected).sum())
    logger.info(
        'Registration: %d anchors, residual max/mean=%.2f/%.2f px, '
        'csv-detected=%d, csv-inferred=%d, ghosts=%d (match_tolerance=%.1f px)',
        len(anchors), residuals_px.max(), residuals_px.mean(),
        n_csv_detected, n_csv_inferred, n_ghosts, match_tolerance,
    )

    return {
        'grid': grid,
        'csv_indices': csv_indices,
        'csv_xy_um': csv_xy_um,
        'is_detected': is_detected,
        'is_csv': is_csv,
        'A': A,
        'residuals_px': residuals_px,
        'match_distances': match_distances,
        'csv_path': csv_path,
    }


# ---------------------------------------------------------------------------
# Step 5 — Threshold computation
# ---------------------------------------------------------------------------

def compute_thresholds(images, grid, mask_mat, num_bins=50, is_alive=None):
    """Fit double-Gaussian histograms and compute per-site thresholds.

    Thin wrapper around ``dynamical_threshold``. If ``is_alive`` is given,
    only alive rows are fit; dead rows get NaN thresholds, NaN infidelities,
    None Gaussian-fit params, and empty histogram bins. The output arrays
    keep the full length of ``grid`` so identity is preserved.

    Returns
    -------
    hist_data, thresholds, gauss_fits, infidelities
    """
    grid = np.asarray(grid, dtype=np.float64)
    M = len(grid)
    if is_alive is None:
        return dynamical_threshold(images, grid, mask_mat, num_bins=num_bins)

    is_alive = np.asarray(is_alive, dtype=bool)
    alive_idx = np.where(is_alive)[0]
    if len(alive_idx) == 0:
        # Nothing to fit — return all-NaN arrays.
        hist_data = [{'counts': np.array([]), 'bin_centers': np.array([])}
                     for _ in range(M)]
        return (hist_data, np.full(M, np.nan),
                [{'params': None} for _ in range(M)], np.full(M, np.nan))

    alive_grid = grid[alive_idx]
    hd_a, thr_a, gf_a, inf_a = dynamical_threshold(
        images, alive_grid, mask_mat, num_bins=num_bins
    )

    # Expand back to full length M.
    hist_data = [{'counts': np.array([]), 'bin_centers': np.array([])}
                 for _ in range(M)]
    thresholds = np.full(M, np.nan)
    gauss_fits = [{'params': None} for _ in range(M)]
    infidelities = np.full(M, np.nan)
    for k, i in enumerate(alive_idx):
        hist_data[i] = hd_a[k]
        thresholds[i] = thr_a[k]
        gauss_fits[i] = gf_a[k]
        infidelities[i] = inf_a[k]
    return hist_data, thresholds, gauss_fits, infidelities


# ---------------------------------------------------------------------------
# Step 6 — Visualization
# ---------------------------------------------------------------------------

def plot_grid_overlay(avg_image, grid, infidelities=None, thresholds=None,
                      labels=None, box_size=10, ax=None):
    """Show the average image with grid sites overlaid.

    If ``infidelities`` is given, sites are colored by infidelity (green =
    good, red = bad) — useful as a sanity check after threshold computation.

    Parameters
    ----------
    labels : sequence or None
        Per-site label. If given (length matches ``grid``), the text drawn
        inside each box is ``str(labels[i])``. Default is the 1-indexed row
        position. Use this to preserve CSV identity when ``grid`` was
        filtered by ``is_alive`` — pass ``csv_indices[is_alive]``.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import cm, colors

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    else:
        fig = ax.figure

    vmin, vmax = np.percentile(avg_image, [1, 99.5])
    ax.imshow(avg_image, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')

    half = box_size / 2

    def label_for(i):
        if labels is not None and i < len(labels):
            return str(labels[i])
        return str(i + 1)

    if infidelities is not None:
        finite = np.isfinite(infidelities)
        if finite.any():
            vmax_inf = max(np.nanpercentile(infidelities[finite], 95), 1e-3)
            norm = colors.Normalize(vmin=0, vmax=vmax_inf)
        else:
            norm = colors.Normalize(vmin=0, vmax=1)
        cmap = cm.RdYlGn_r

        for i, (y, x) in enumerate(grid):
            inf = infidelities[i] if i < len(infidelities) else np.nan
            color = cmap(norm(inf)) if np.isfinite(inf) else 'magenta'
            ax.add_patch(Rectangle((x - half, y - half), box_size, box_size,
                                   linewidth=1.5, edgecolor=color,
                                   facecolor='none'))
            ax.text(x, y, label_for(i), color='yellow', fontsize=7,
                    ha='center', va='center')

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('Infidelity')
        ax.set_title(f'Grid overlay — {len(grid)} sites, '
                     f'mean infidelity {np.nanmean(infidelities):.2e}')
    else:
        for i, (y, x) in enumerate(grid):
            ax.add_patch(Rectangle((x - half, y - half), box_size, box_size,
                                   linewidth=1.5, edgecolor='red',
                                   facecolor='none'))
            ax.text(x, y, label_for(i), color='yellow', fontsize=7,
                    ha='center', va='center')
        ax.set_title(f'Grid overlay — {len(grid)} sites')

    return fig, ax


def plot_histograms(hist_data, thresholds, gauss_fits, infidelities,
                    num_cols=5, num_rows=4, sort_by=None,
                    csv_indices=None, is_alive=None):
    """Single-figure paginated histogram viewer.

    Use **Prev / Next** buttons or the **Left / Right arrow** keys to flip
    pages. **PageUp / PageDown** also work. The figure title shows the
    current page and the site range displayed.

    Parameters
    ----------
    hist_data, thresholds, gauss_fits, infidelities : per-site outputs of
        :func:`compute_thresholds`.
    num_cols, num_rows : int
        Tile layout per page.
    sort_by : {None, 'infidelity', 'site'}
        If 'infidelity', show sites in descending infidelity order
        (worst first) — handy when scanning thousands of sites. NaN
        infidelities (e.g. for deleted sites) are placed last.
    csv_indices : ndarray (M,) int or None
        If given, tiles are labeled "Site k" using the CSV index instead
        of the row position. Rows with -1 are labeled by row position.
    is_alive : ndarray (M,) bool or None
        If given, dead rows render as a "DELETED" placeholder tile.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from scipy.stats import norm

    num_sites = len(hist_data)
    per_page = num_cols * num_rows
    num_pages = max(1, (num_sites + per_page - 1) // per_page)

    if sort_by == 'infidelity':
        # Place NaN infidelities (incl. deleted sites) at the end.
        inf = np.asarray(infidelities, dtype=float)
        order = np.lexsort((np.arange(num_sites), -inf, np.isnan(inf)))
    else:
        order = np.arange(num_sites)

    csv_indices = (np.asarray(csv_indices, dtype=int)
                   if csv_indices is not None else None)
    is_alive = (np.asarray(is_alive, dtype=bool)
                if is_alive is not None else None)

    def label_for(s):
        if csv_indices is not None and 0 <= s < len(csv_indices) and csv_indices[s] > 0:
            return f'Site {int(csv_indices[s])}'
        return f'Row {s + 1}'

    fig = plt.figure(figsize=(3.5 * num_cols, 3 * num_rows + 0.8))
    gs = fig.add_gridspec(num_rows, num_cols,
                          top=0.92, bottom=0.10,
                          left=0.05, right=0.97,
                          hspace=0.55, wspace=0.30)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(num_cols)]
                     for r in range(num_rows)])

    state = {'page': 0}

    def render():
        page = state['page']
        start = page * per_page
        end = min(start + per_page, num_sites)
        for idx in range(per_page):
            r, c = divmod(idx, num_cols)
            ax = axes[r, c]
            ax.clear()
            slot = start + idx
            if slot >= end:
                ax.set_visible(False)
                continue
            ax.set_visible(True)

            s = int(order[slot])
            tag = label_for(s)
            dead = is_alive is not None and not bool(is_alive[s])

            if dead:
                ax.set_title(f'{tag} — DELETED', fontsize=9, color='gray')
                ax.text(0.5, 0.5, 'DELETED', transform=ax.transAxes,
                        ha='center', va='center', fontsize=18,
                        color='gray', alpha=0.5)
                ax.set_facecolor('#f4f4f4')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            hd = hist_data[s]
            bc = hd['bin_centers']
            ct = hd['counts']
            if len(bc) == 0:
                # No fit produced (e.g. site dropped by dynamical_threshold)
                ax.set_title(f'{tag} — no data', fontsize=9, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            width = np.diff(bc).mean() if len(bc) > 1 else 1
            ax.bar(bc, ct, width=width, alpha=0.6, color='steelblue')
            ax.axvline(thresholds[s], color='red', ls='--', lw=1.5)

            params = gauss_fits[s].get('params') if gauss_fits[s] else None
            if params is not None and len(params) == 6:
                mu1, s1, A1, mu2, s2, A2 = params
                x_fine = np.linspace(bc[0], bc[-1], 200)
                ax.plot(x_fine, A1 * norm.pdf(x_fine, mu1, s1), 'g-', lw=1.2)
                ax.plot(x_fine, A2 * norm.pdf(x_fine, mu2, s2), 'g-', lw=1.2)

            ax.set_title(tag, fontsize=9)
            inf_val = infidelities[s]
            if np.isfinite(inf_val):
                ax.text(0.97, 0.95, f'{inf_val:.2e}',
                        transform=ax.transAxes,
                        fontsize=7, ha='right', va='top', color='darkred')
            ax.tick_params(labelsize=7)

        order_tag = ' [worst-first]' if sort_by == 'infidelity' else ''
        fig.suptitle(
            f'Site histograms — page {page + 1}/{num_pages}  '
            f'(slots {start + 1}–{end} of {num_sites}){order_tag}',
            fontsize=12,
        )
        fig.canvas.draw_idle()

    # Page-control buttons
    ax_prev = fig.add_axes([0.42, 0.025, 0.07, 0.04])
    ax_next = fig.add_axes([0.51, 0.025, 0.07, 0.04])
    btn_prev = Button(ax_prev, '◀ Prev')
    btn_next = Button(ax_next, 'Next ▶')

    def on_prev(_event):
        state['page'] = (state['page'] - 1) % num_pages
        render()

    def on_next(_event):
        state['page'] = (state['page'] + 1) % num_pages
        render()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def on_key(event):
        if event.key in ('right', 'pagedown', 'down'):
            on_next(None)
        elif event.key in ('left', 'pageup', 'up'):
            on_prev(None)
        elif event.key == 'home':
            state['page'] = 0
            render()
        elif event.key == 'end':
            state['page'] = num_pages - 1
            render()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Keep button refs alive (matplotlib weak-refs button callbacks)
    fig._hist_buttons = (btn_prev, btn_next)

    render()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Step 7 — Save calibration files
# ---------------------------------------------------------------------------

def save_calibration(scan_dir, grid, thresholds, infidelities,
                     gauss_fits, hist_data, save_to_day_folder=False,
                     is_alive=None, csv_indices=None):
    """Write gridLocations.txt, threshold.mat, histData.mat to ``scan_dir``.

    File formats match data_manager.py _save_grid/_save_threshold/_save_histdata.

    Parameters
    ----------
    is_alive : ndarray (M,) bool or None
        If given, only alive rows are saved (dead rows are dropped).
    csv_indices : ndarray (M,) int or None
        If given, ``gridLocations.txt`` gains a third column ``Site_Index``
        carrying the CSV index for each row (-1 for non-CSV / ghost /
        user-added). When None, the file stays in the legacy 2-column
        ``Y\\tX`` form.
    save_to_day_folder : bool
        If True, also copy the three files to the parent (day) folder.
        Use with care — overwrites the day-level calibration.
    """
    grid = np.asarray(grid, dtype=np.float64)
    thresholds = np.asarray(thresholds)
    infidelities = np.asarray(infidelities)
    csv_indices_arr = (np.asarray(csv_indices, dtype=int)
                       if csv_indices is not None else None)

    if is_alive is not None:
        keep = np.asarray(is_alive, dtype=bool)
        grid = grid[keep]
        thresholds = thresholds[keep]
        infidelities = infidelities[keep]
        gauss_fits = [gf for gf, k in zip(gauss_fits, keep) if k]
        hist_data = [hd for hd, k in zip(hist_data, keep) if k]
        if csv_indices_arr is not None:
            csv_indices_arr = csv_indices_arr[keep]
        n_dropped = int((~keep).sum())
        if n_dropped:
            logger.info('save_calibration: dropping %d dead site(s)', n_dropped)

    M = len(thresholds)

    grid_path = os.path.join(scan_dir, 'gridLocations.txt')
    if csv_indices_arr is not None:
        out = np.column_stack([grid, csv_indices_arr.reshape(-1, 1).astype(np.int64)])
        np.savetxt(grid_path, out, header='Y\tX\tSite_Index',
                   fmt=['%.6f', '%.6f', '%d'],
                   delimiter='\t', comments='')
    else:
        np.savetxt(grid_path, grid, header='Y\tX', delimiter='\t', comments='')
    logger.info('Saved %s', grid_path)

    gs = np.empty(M, dtype=[('params', 'O')])
    for s in range(M):
        p = gauss_fits[s].get('params') if gauss_fits[s] else None
        gs[s]['params'] = p if p is not None else np.array([])
    thresh_path = os.path.join(scan_dir, 'threshold.mat')
    savemat(thresh_path, {
        'thresholds': thresholds,
        'infidelities': infidelities,
        'gaussFitsStruct': gs,
    })
    logger.info('Saved %s', thresh_path)

    hs = np.empty(M, dtype=[('counts', 'O'), ('bin_centers', 'O')])
    for s in range(M):
        hs[s]['counts'] = hist_data[s]['counts']
        hs[s]['bin_centers'] = hist_data[s]['bin_centers']
    hist_path = os.path.join(scan_dir, 'histData.mat')
    savemat(hist_path, {'histData': hs})
    logger.info('Saved %s', hist_path)

    if save_to_day_folder:
        day_folder = os.path.dirname(scan_dir)
        for src_name in ('gridLocations.txt', 'threshold.mat', 'histData.mat'):
            src = os.path.join(scan_dir, src_name)
            dst = os.path.join(day_folder, src_name)
            try:
                import shutil
                shutil.copy2(src, dst)
                logger.info('Copied to %s', dst)
            except Exception as e:
                logger.warning('Failed to copy %s to day folder: %s', src_name, e)


# ---------------------------------------------------------------------------
# One-shot wrapper (legacy)
# ---------------------------------------------------------------------------

def hist_init(scan_dir, num_tweezers, box_size=9, sigma=2.0, num_bins=50,
              n_avg=200, n_thresh=2000, save_to_day_folder=True):
    """Run the full hist-init pipeline in one call.

    Equivalent to calling ``load_scan_context`` → ``compute_avg_image`` →
    ``detect_grid`` → ``edit_grid`` → ``load_threshold_images`` →
    ``compute_thresholds`` → ``save_calibration`` → ``plot_histograms``.

    Prefer the composable API for iterative work in a notebook.
    """
    print(f'Loading scan from {scan_dir} ...')
    ctx = load_scan_context(scan_dir)

    print(f'Loading {min(ctx["num_seq"], n_avg)} images for averaging ...')
    avg_image = compute_avg_image(ctx, n_avg=n_avg)

    mask_mat = make_mask(box_size, sigma)

    print(f'Detecting top {num_tweezers} tweezer spots ...')
    spots = detect_grid(avg_image, num_tweezers, sort=False)
    print(f'Found {len(spots)} candidate spots')
    if len(spots) == 0:
        print('No spots auto-detected. The editor will open with an empty grid.')
        spots = np.zeros((0, 2), dtype=np.float64)

    print('Opening interactive grid editor ...')
    # Legacy path: filter dead and sort the alive grid for backward compat.
    edit_result = edit_grid(avg_image, spots, sort=True)
    grid = edit_result['grid']
    print(f'{len(grid)} sites after editing')

    print(f'Loading {min(ctx["num_seq"], n_thresh)} images for threshold computation ...')
    images = load_threshold_images(ctx, n_thresh=n_thresh)

    print('Computing per-site thresholds ...')
    hist_data, thresholds, gauss_fits, infidelities = compute_thresholds(
        images, grid, mask_mat, num_bins=num_bins
    )
    del images
    print(f'Mean infidelity: {np.nanmean(infidelities):.4e}')

    save_calibration(scan_dir, grid, thresholds, infidelities,
                     gauss_fits, hist_data,
                     save_to_day_folder=save_to_day_folder)
    print(f'Calibration saved to {scan_dir}')
    if save_to_day_folder:
        print(f'Also copied to {os.path.dirname(scan_dir)}')

    plot_histograms(hist_data, thresholds, gauss_fits, infidelities)

    return {
        'grid_locations': grid,
        'thresholds': thresholds,
        'infidelities': infidelities,
        'gauss_fits': gauss_fits,
        'hist_data': hist_data,
        'mask_mat': mask_mat,
        'avg_image': avg_image,
        'scan_dir': scan_dir,
        'ctx': ctx,
    }
