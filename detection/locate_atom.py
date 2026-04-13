"""Grid location tracking via global-shift cross-correlation.

Port of YbDataAnalysis/AtomDetection/LocateAtom.m (update mode only).
"""

import numpy as np
from scipy.optimize import least_squares


def _gaussian_2d(coords, A, x0, sigma_x, y0, sigma_y, offset):
    """2D Gaussian model for shift heatmap fitting."""
    dx = coords[:, 0] - x0
    dy = coords[:, 1] - y0
    return A * np.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2))) + offset


def _residuals_2d(params, coords, z_data):
    return _gaussian_2d(coords, *params) - z_data


def _custom_round(val):
    """Round: if fractional part <= 0.5, toward zero; else away from zero."""
    frac = abs(val - int(val))
    if frac <= 0.5:
        return int(val)  # truncate toward zero
    else:
        return int(np.ceil(val)) if val >= 0 else int(np.floor(val))


def locate_atom_update(images, grid_locations, search_range, mask_mat):
    """Refine grid locations by finding a global pixel shift.

    Computes a brute-force cross-correlation over candidate shifts, fits a 2D
    Gaussian to the resulting heatmap, and applies the rounded shift to all
    grid positions.

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
        Stack of recent images.
    grid_locations : ndarray, shape (M, 2)
        Current site positions [y, x].
    search_range : int
        Maximum shift to search (pixels).
    mask_mat : ndarray, shape (B, B)
        Gaussian weighting mask.

    Returns
    -------
    updated_grid : ndarray, shape (M, 2)
        Updated grid locations.
    avg_image : ndarray, shape (H, W)
        Mean image.
    shift_dy : int
        Applied y-shift.
    shift_dx : int
        Applied x-shift.
    heatmap : ndarray, shape (2*R+1, 2*R+1)
        Intensity heatmap over candidate shifts (for visualization).
    """
    images = np.asarray(images, dtype=np.float64)
    grid_locations = np.asarray(grid_locations, dtype=np.float64)
    mask_mat = np.asarray(mask_mat, dtype=np.float64)

    avg_image = images.mean(axis=0)
    R = int(round(search_range))
    H, W = avg_image.shape
    num_sites = grid_locations.shape[0]
    box_size = mask_mat.shape[0]
    half_box = box_size // 2

    heatmap = np.zeros((2 * R + 1, 2 * R + 1))

    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            total_intensity = 0.0
            valid_count = 0
            for s in range(num_sites):
                ny = grid_locations[s, 0] + dy
                nx = grid_locations[s, 1] + dx
                y_min = int(round(ny)) - half_box
                y_max = int(round(ny)) + half_box
                x_min = int(round(nx)) - half_box
                x_max = int(round(nx)) + half_box
                if y_min < 0 or x_min < 0 or y_max >= H or x_max >= W:
                    continue
                patch = avg_image[y_min:y_max + 1, x_min:x_max + 1]
                if patch.shape != mask_mat.shape:
                    continue
                total_intensity += np.sum(patch * mask_mat) / mask_mat.size
                valid_count += 1
            if valid_count > 0:
                heatmap[dy + R, dx + R] = total_intensity / valid_count

    # Find initial peak
    max_idx = np.argmax(heatmap)
    max_row, max_col = np.unravel_index(max_idx, heatmap.shape)
    initial_dy = max_row - R
    initial_dx = max_col - R

    # Fit 2D Gaussian to heatmap
    shift_coords = np.mgrid[-R:R + 1, -R:R + 1]
    X = shift_coords[1].ravel()  # dx
    Y = shift_coords[0].ravel()  # dy
    coords = np.column_stack([X, Y])
    z_data = heatmap.ravel()

    A_guess = z_data.max() - z_data.min()
    offset_guess = z_data.min()
    p0 = [A_guess, float(initial_dx), 2.0, float(initial_dy), 2.0, offset_guess]
    lb = [0, -R, 0.1, -R, 0.1, -np.inf]
    ub = [np.inf, R, R, R, R, np.inf]

    try:
        result = least_squares(
            _residuals_2d, p0, args=(coords, z_data),
            bounds=(lb, ub), method='trf'
        )
        fitted_dx = result.x[1]
        fitted_dy = result.x[3]
    except Exception:
        fitted_dx = float(initial_dx)
        fitted_dy = float(initial_dy)

    global_dx = _custom_round(fitted_dx)
    global_dy = _custom_round(fitted_dy)

    updated_grid = grid_locations + np.array([[global_dy, global_dx]])

    return updated_grid, avg_image, global_dy, global_dx, heatmap
