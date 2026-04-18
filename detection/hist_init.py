"""Offline initialization of tweezer grid locations and detection thresholds.

Port of matlab_new/AtomDetection/YbHistInit.m + LocateAtom.m (initial mode).

Typical workflow:
    1. Run a scan with isInit=1  ->  images saved, no processing
    2. Call hist_init(scan_dir)  ->  detect spots, edit interactively,
       compute thresholds, save calibration files
    3. Subsequent scans with isInit=0 pick up those calibration files
"""

import os
import logging

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, label, center_of_mass
from scipy.io import savemat

from yb_analysis.analysis.load_data import load_scan_from_path, load_images
from yb_analysis.detection.dynamical_threshold import dynamical_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_mask(box_size, sigma):
    """Create Gaussian weighting mask (matches MATLAB imgaussfilt approach)."""
    mask = np.zeros((box_size, box_size))
    center = box_size // 2
    mask[center, center] = 1.0
    return gaussian_filter(mask, sigma)


def _detect_spots(avg_image, num_tweezers, spot_sigma=2.0, min_distance=10):
    """Find the top *num_tweezers* brightest spots via LoG filtering.

    Computes the -LoG (Laplacian-of-Gaussian) response of the image, finds
    all local maxima, and returns the *num_tweezers* with the strongest
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

    Returns
    -------
    positions : ndarray (N, 2) — [y, x] pairs, N <= num_tweezers.
    """
    img = avg_image.astype(np.float64)

    # -LoG: positive at bright blobs matching spot_sigma
    neg_log = -gaussian_laplace(img, sigma=spot_sigma)

    # Find many candidate peaks (ask for more than needed, then rank)
    n_candidates = num_tweezers * 3

    try:
        from skimage.feature import peak_local_max
        coords = peak_local_max(neg_log, min_distance=min_distance,
                                num_peaks=n_candidates)
    except ImportError:
        # Fallback: low threshold + connected components
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

    # Rank by LoG response strength and keep the top num_tweezers
    responses = neg_log[coords[:, 0], coords[:, 1]]
    top_idx = np.argsort(responses)[::-1][:num_tweezers]
    return coords[top_idx].astype(np.float64)


def _sort_grid(positions):
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
        chunk = chunk[np.argsort(chunk[:, 0])]  # sort by Y within cluster
        result.append(chunk)
    return np.vstack(result)


def _interactive_grid_editor(avg_image, initial_grid):
    """Launch matplotlib click-to-add/remove editor for tweezer sites.

    Left-click near an existing point (<10 px) removes it.
    Left-click elsewhere adds a new point.
    Press "Done" or close the window to finish.

    Requires a live matplotlib backend (%matplotlib widget or qt).
    """
    import matplotlib
    backend = matplotlib.get_backend()
    if 'inline' in backend.lower():
        raise RuntimeError(
            'Interactive grid editing requires a live matplotlib backend.\n'
            "Run '%matplotlib qt' (or '%matplotlib widget' if ipympl is "
            'installed) before calling hist_init().'
        )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from matplotlib.patches import Rectangle

    grid = initial_grid.tolist()  # mutable list of [y, x]

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    vmin, vmax = np.percentile(avg_image, [1, 99.5])
    ax.imshow(avg_image, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title('Left-click: add/remove sites. Press Done when finished.')

    artists = []  # list of (rect, text) tuples

    def _redraw():
        for rect, txt in artists:
            rect.remove()
            txt.remove()
        artists.clear()
        for i, (y, x) in enumerate(grid):
            r = Rectangle((x - 5, y - 5), 10, 10, linewidth=2,
                           edgecolor='red', facecolor='none')
            ax.add_patch(r)
            t = ax.text(x, y, str(i + 1), color='yellow', fontsize=8,
                        ha='center', va='center')
            artists.append((r, t))
        fig.canvas.draw_idle()

    _redraw()

    def _on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        xc, yc = event.xdata, event.ydata
        if xc is None or yc is None:
            return

        # Check proximity to existing points
        min_dist = float('inf')
        min_idx = -1
        for i, (y, x) in enumerate(grid):
            d = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                min_idx = i
        if min_dist < 10:
            grid.pop(min_idx)
        else:
            grid.append([round(yc), round(xc)])
        _redraw()

    fig.canvas.mpl_connect('button_press_event', _on_click)

    ax_button = fig.add_axes([0.4, 0.01, 0.2, 0.05])
    btn = Button(ax_button, 'Done')

    def _done(_event):
        plt.close(fig)

    btn.on_clicked(_done)
    plt.show(block=True)

    if not grid:
        raise ValueError('No grid locations selected. Cannot compute thresholds.')
    return np.array(grid, dtype=np.float64)


def _save_calibration(scan_dir, grid, thresholds, infidelities,
                      gauss_fits, hist_data, day_folder=None):
    """Write gridLocations.txt, threshold.mat, histData.mat.

    File formats match data_manager.py _save_grid/_save_threshold/_save_histdata.
    """
    M = len(thresholds)

    # --- gridLocations.txt ---
    grid_path = os.path.join(scan_dir, 'gridLocations.txt')
    np.savetxt(grid_path, grid, header='Y\tX', delimiter='\t', comments='')
    logger.info('Saved %s', grid_path)

    # --- threshold.mat ---
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

    # --- histData.mat ---
    hs = np.empty(M, dtype=[('counts', 'O'), ('bin_centers', 'O')])
    for s in range(M):
        hs[s]['counts'] = hist_data[s]['counts']
        hs[s]['bin_centers'] = hist_data[s]['bin_centers']
    hist_path = os.path.join(scan_dir, 'histData.mat')
    savemat(hist_path, {'histData': hs})
    logger.info('Saved %s', hist_path)

    # Copy to day folder
    if day_folder:
        for src_name in ('gridLocations.txt', 'threshold.mat', 'histData.mat'):
            src = os.path.join(scan_dir, src_name)
            dst = os.path.join(day_folder, src_name)
            try:
                import shutil
                shutil.copy2(src, dst)
                logger.info('Copied to %s', dst)
            except Exception as e:
                logger.warning('Failed to copy %s to day folder: %s', src_name, e)


def _plot_histograms(hist_data, thresholds, gauss_fits, infidelities,
                     num_cols=5, num_rows=4):
    """Paginated histogram summary with fitted Gaussians and thresholds."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    num_sites = len(hist_data)
    per_page = num_cols * num_rows
    num_pages = max(1, (num_sites + per_page - 1) // per_page)

    for page in range(num_pages):
        start = page * per_page
        end = min(start + per_page, num_sites)
        n = end - start

        fig, axes = plt.subplots(num_rows, num_cols,
                                 figsize=(3.5 * num_cols, 3 * num_rows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f'Site histograms (page {page + 1}/{num_pages})',
                     fontsize=13)

        for idx in range(per_page):
            row, col = divmod(idx, num_cols)
            ax = axes[row, col]
            s = start + idx
            if s >= end:
                ax.set_visible(False)
                continue

            hd = hist_data[s]
            bc = hd['bin_centers']
            ct = hd['counts']
            ax.bar(bc, ct, width=np.diff(bc).mean() if len(bc) > 1 else 1,
                   alpha=0.6, color='steelblue')

            # Threshold line
            ax.axvline(thresholds[s], color='red', ls='--', lw=1.5,
                       label=f'thr={thresholds[s]:.0f}')

            # Gaussian fits
            params = gauss_fits[s].get('params') if gauss_fits[s] else None
            if params is not None and len(params) == 6:
                mu1, s1, A1, mu2, s2, A2 = params
                x_fine = np.linspace(bc[0], bc[-1], 200)
                ax.plot(x_fine, A1 * norm.pdf(x_fine, mu1, s1),
                        'g-', lw=1.2)
                ax.plot(x_fine, A2 * norm.pdf(x_fine, mu2, s2),
                        'g-', lw=1.2)

            inf_val = infidelities[s]
            ax.set_title(f'Site {s + 1}', fontsize=9)
            if not np.isnan(inf_val):
                ax.text(0.97, 0.95, f'{inf_val:.2e}', transform=ax.transAxes,
                        fontsize=7, ha='right', va='top', color='darkred')
            ax.tick_params(labelsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hist_init(scan_dir, num_tweezers, box_size=9, sigma=2.0, num_bins=50,
              save_to_day_folder=True):
    """Initialize tweezer grid locations and detection thresholds.

    Loads images from a scan directory (typically an isInit=1 scan), detects
    tweezer spots automatically, lets the user interactively add/remove sites,
    fits double-Gaussian histograms, and saves calibration files.

    Parameters
    ----------
    scan_dir : str
        Path to scan directory, e.g.
        r'D:\\...\\Data\\20260416\\data_20260416_182630'
    num_tweezers : int
        Expected number of tweezer spots.  The detector picks the top N
        brightest blob-like features from the averaged image.
    box_size : int
        Side length of the Gaussian weighting mask (pixels).
    sigma : float
        Gaussian sigma for the weighting mask.
    num_bins : int
        Number of histogram bins for the double-Gaussian fit.
    save_to_day_folder : bool
        If True, also save calibration files to the parent date folder.

    Returns
    -------
    dict with keys:
        grid_locations : ndarray (M, 2) — [Y, X]
        thresholds : ndarray (M,)
        infidelities : ndarray (M,)
        gauss_fits : list of dict
        hist_data : list of dict
        mask_mat : ndarray (box_size, box_size)
        avg_image : ndarray (H, W)
        scan_dir : str
    """
    # 1. Load scan metadata
    print(f'Loading scan from {scan_dir} ...')
    scan_data = load_scan_from_path(scan_dir)
    config = scan_data['Scan']
    data_path = scan_data['path']
    imgs_shape = scan_data.get('imgs_shape')

    if imgs_shape is None or imgs_shape[0] == 0:
        raise ValueError(f'No images found in {data_path}')

    total_frames = imgs_shape[0]
    num_images = int(np.asarray(config.get('NumImages', 1)).flat[0])

    # 2. Select first image of each sequence
    all_first_indices = list(range(0, total_frames, num_images))
    num_seq = len(all_first_indices)

    # 3. Load a subset for the average image (fast spot detection)
    n_avg = min(num_seq, 200)
    avg_indices = all_first_indices[:n_avg]
    print(f'Loading {n_avg} images (of {num_seq}) for averaging ...')
    avg_imgs = load_images(data_path, avg_indices).astype(np.float64)
    avg_image = avg_imgs.mean(axis=0)
    del avg_imgs

    # 4. Build mask
    mask_mat = _gaussian_mask(box_size, sigma)

    # 5. Detect spots — pick the top num_tweezers by LoG response
    print(f'Detecting top {num_tweezers} tweezer spots ...')
    spots = _detect_spots(avg_image, num_tweezers)
    print(f'Found {len(spots)} candidate spots')
    if len(spots) == 0:
        print('No spots auto-detected. The editor will open with an empty grid.')
        spots = np.zeros((0, 2), dtype=np.float64)

    # 6. Interactive editing
    print('Opening interactive grid editor ...')
    edited_grid = _interactive_grid_editor(avg_image, spots)
    print(f'{len(edited_grid)} sites after editing')

    # 7. Sort
    grid = _sort_grid(edited_grid)

    # 8. Load images for threshold computation (cap at 2000)
    n_thresh = min(num_seq, 2000)
    thresh_indices = all_first_indices[:n_thresh]
    print(f'Loading {n_thresh} images for threshold computation ...')
    images = load_images(data_path, thresh_indices).astype(np.float64)

    # 9. Compute thresholds (reuse dynamical_threshold)
    print('Computing per-site thresholds ...')
    hist_data, thresholds, gauss_fits, infidelities = dynamical_threshold(
        images, grid, mask_mat, num_bins=num_bins
    )
    del images
    print(f'Mean infidelity: {np.nanmean(infidelities):.4e}')

    # 10. Save calibration files
    day_folder = os.path.dirname(scan_dir) if save_to_day_folder else None
    _save_calibration(scan_dir, grid, thresholds, infidelities,
                      gauss_fits, hist_data, day_folder=day_folder)
    print(f'Calibration saved to {scan_dir}')
    if day_folder:
        print(f'Also copied to {day_folder}')

    # 11. Plot summary
    _plot_histograms(hist_data, thresholds, gauss_fits, infidelities)

    return {
        'grid_locations': grid,
        'thresholds': thresholds,
        'infidelities': infidelities,
        'gauss_fits': gauss_fits,
        'hist_data': hist_data,
        'mask_mat': mask_mat,
        'avg_image': avg_image,
        'scan_dir': scan_dir,
    }
