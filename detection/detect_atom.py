"""Atom detection via masked-intensity thresholding.

Port of YbDataAnalysis/AtomDetection/DetectAtom.m
"""

import numpy as np


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

    num_sites = grid_locations.shape[0]
    box_size = mask_mat.shape[0]
    half_box = box_size // 2
    H, W = img.shape
    mask_flat = mask_mat.ravel()

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

    for k in range(N):
        all_status[k], all_intensities[k] = detect_atom(
            images[k], grid_locations, thresholds, mask_mat
        )

    return all_status, all_intensities
