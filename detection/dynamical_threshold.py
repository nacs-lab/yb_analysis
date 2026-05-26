"""Dynamical per-site threshold computation via double-Gaussian fit.

Port of YbDataAnalysis/AtomDetection/DynamicalThreshold.m
"""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import norm


def _two_gaussians(x, params):
    """Evaluate sum of two Gaussian PDFs: A1*N(mu1,s1) + A2*N(mu2,s2)."""
    mu1, s1, A1, mu2, s2, A2 = params
    return A1 * norm.pdf(x, mu1, s1) + A2 * norm.pdf(x, mu2, s2)


def _residuals(params, x, y):
    return _two_gaussians(x, params) - y


def _compute_site_intensities(images, positions, mask_mat):
    """Compute masked intensity for each (image, site) pair.

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
    positions : ndarray, shape (M, 2) — [y, x]
    mask_mat : ndarray, shape (B, B)

    Returns
    -------
    intensities : ndarray, shape (N, M)
    """
    N = images.shape[0]
    M = positions.shape[0]
    box_size = mask_mat.shape[0]
    r = box_size // 2
    H, W = images.shape[1], images.shape[2]

    intensities = np.zeros((N, M), dtype=np.float64)

    for s in range(M):
        y0 = int(round(positions[s, 0]))
        x0 = int(round(positions[s, 1]))

        y1 = max(0, y0 - r)
        y2 = min(H, y0 + r + 1)
        x1 = max(0, x0 - r)
        x2 = min(W, x0 + r + 1)

        # Sub-mask for edge handling
        my1 = y1 - (y0 - r)
        my2 = my1 + (y2 - y1)
        mx1 = x1 - (x0 - r)
        mx2 = mx1 + (x2 - x1)
        sub_mask = mask_mat[my1:my2, mx1:mx2]
        sub_mask_vec = sub_mask.ravel()

        # Extract patches for all images at once: (N, patch_h, patch_w)
        patch_stack = images[:, y1:y2, x1:x2]
        # Reshape to (N, patch_pixels) and dot with mask
        P = patch_stack.reshape(N, -1).astype(np.float64)
        intensities[:, s] = P @ sub_mask_vec

    return intensities


def dynamical_threshold(images, positions, mask_mat, num_bins=50,
                        outlier_clip_mad=5.0):
    """Compute per-site detection thresholds from a stack of recent images.

    Fits a two-Gaussian mixture to each site's intensity histogram and finds
    the cutoff that minimizes the sum of the two tail integrals (infidelity).

    Parameters
    ----------
    images : ndarray, shape (N, H, W)
        Stack of recent images (float or int).
    positions : ndarray, shape (M, 2)
        Grid locations [y, x] per site.
    mask_mat : ndarray, shape (B, B)
        Gaussian weighting mask.
    num_bins : int
        Number of histogram bins.
    outlier_clip_mad : float or None
        Per-site, drop intensities above ``median + outlier_clip_mad * 1.4826
        * MAD`` before histogramming and fitting. Rejects bad-frame artifacts
        (e.g. cosmic rays / readout glitches) whose huge values otherwise
        blow out the histogram range and squash the bimodal no-atom/atom
        structure into what looks like a single peak. Set to None to disable.

    Returns
    -------
    hist_data : list of dict
        Per-site {'counts': ndarray, 'bin_centers': ndarray}.
    thresholds : ndarray, shape (M,)
        Optimal detection threshold per site.
    gauss_fits : list of dict
        Per-site {'params': ndarray[6] or None}.
    infidelities : ndarray, shape (M,)
        Discrimination infidelity per site.
    """
    images = np.asarray(images, dtype=np.float64)
    positions = np.asarray(positions)
    mask_mat = np.asarray(mask_mat, dtype=np.float64)

    num_sites = positions.shape[0]
    intensities = _compute_site_intensities(images, positions, mask_mat)

    thresholds = np.zeros(num_sites)
    infidelities = np.zeros(num_sites)
    hist_data = []
    gauss_fits = []

    for s in range(num_sites):
        site_data = intensities[:, s]
        if outlier_clip_mad is not None and outlier_clip_mad > 0:
            med = np.median(site_data)
            mad = np.median(np.abs(site_data - med))
            if mad > 0:
                upper = med + outlier_clip_mad * 1.4826 * mad
                site_data = site_data[site_data <= upper]
        counts, edges = np.histogram(site_data, bins=num_bins, density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        hist_data.append({'counts': counts, 'bin_centers': bin_centers})

        min_val = site_data.min()
        max_val = site_data.max()
        std_val = site_data.std()

        # Initial guess: [mu1, s1, A1, mu2, s2, A2]
        x0 = np.array([
            0.25 * max_val + 0.75 * min_val, std_val / 10, 0.2,
            0.75 * max_val + 0.25 * min_val, std_val / 2, 0.2,
        ])
        lb = np.array([
            min_val, std_val / 15, 0.05,
            0.5 * max_val + 0.5 * min_val, std_val / 15, 0.05,
        ])
        ub = np.array([
            0.5 * max_val + 0.5 * min_val, std_val, 1.0,
            max_val, std_val, 1.0,
        ])

        try:
            result = least_squares(
                _residuals, x0, args=(bin_centers, counts),
                bounds=(lb, ub), method='trf'
            )
            params = result.x
            mu1, s1 = params[0], params[1]
            mu2, s2 = params[3], params[4]

            # Ensure mu1 < mu2
            if mu1 > mu2:
                mu1, s1, mu2, s2 = mu2, s2, mu1, s1
                params = np.array([mu1, s1, params[3 + 2], mu2, s2, params[2]])

            # Minimize infidelity = P(false positive) + P(false negative)
            def infidelity_func(x_cut):
                return (1.0 - norm.cdf(x_cut, mu1, s1)) + norm.cdf(x_cut, mu2, s2)

            opt = minimize_scalar(infidelity_func, bounds=(mu1, mu2), method='bounded')
            thresholds[s] = opt.x
            infidelities[s] = opt.fun
            gauss_fits.append({'params': params})

        except Exception:
            thresholds[s] = np.median(site_data)
            infidelities[s] = np.nan
            gauss_fits.append({'params': None})

    return hist_data, thresholds, gauss_fits, infidelities
