"""Unpack scan logicals: group sequences by scan parameter.

Port of MATLAB's unpackScanLogicals.m.
"""

import numpy as np

from yb_analysis.detection.scan_analysis import (
    extract_scan_params, extract_scan_params_h5,
    extract_scan_dims, extract_scan_dims_h5,
)


def unpack_scan_logicals(scan, logicals, seq_ids=None, mat_path=None):
    """Unpack scanned parameter values and logical results.

    Supports 1D scans with 1 or 2 images per sequence.
    Handles scrambled sequence order via seq_ids + Scan.Params.

    Parameters
    ----------
    scan : dict
        Scan configuration with Params, ScanGroup, NumImages.
    logicals : ndarray (nFrames, nSites)
        All detection logicals (bool). nFrames = NumImages * nCapturedSequences.
    seq_ids : ndarray (nCapturedSequences,), optional
        1-indexed sequence IDs. If None, assumes sequential order.
    mat_path : str, optional
        Path to .mat file for HDF5-direct fallback when dict-based extraction
        fails (e.g. N-D scans with h5py object references).

    Returns
    -------
    scan_params : ndarray (nParams,)
        Unique parameter values (sorted).
    logic1 : ndarray (nSites, nParams, minReps) bool
        Image-1 logicals grouped by parameter.
    logic2 : ndarray (nSites, nParams, minReps) bool or None
        Image-2 logicals (None if NumImages=1).
    """
    if logicals is None:
        raise ValueError(
            'logicals is None — scan may still be running or data not yet saved. '
            'Try loading a completed scan or wait for the current scan to finish.')

    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0])
    params_arr = np.asarray(scan.get('Params', [])).ravel().astype(int)

    # Extract scan dimensions — try dict first, then HDF5-direct fallback
    scan_dims = extract_scan_dims(scan)
    if scan_dims is None and mat_path is not None:
        scan_dims = extract_scan_dims_h5(mat_path)

    if scan_dims is not None and len(scan_dims) >= 2:
        # Build Cartesian grid: shape (s0*s1, 2).
        # MATLAB column-major: dim0 varies fastest (idx0 = p % s0).
        d0, d1 = scan_dims[0], scan_dims[1]
        s0, s1 = d0['size'], d1['size']
        n_combos = s0 * s1
        scan_params = np.empty((n_combos, 2), dtype=float)
        for p in range(n_combos):
            scan_params[p, 0] = d0['values'][p % s0]
            scan_params[p, 1] = d1['values'][p // s0]
    else:
        # 1D scan: scan_params is a flat array
        scan_params = scan_dims[0]['values'] if scan_dims else None
        if scan_params is None:
            scan_params = extract_scan_params(scan)
        if scan_params is None and mat_path is not None:
            scan_params = extract_scan_params_h5(mat_path)
        if scan_params is None:
            n_unique = int(params_arr.max()) if len(params_arr) > 0 else 1
            scan_params = np.arange(1, n_unique + 1, dtype=float)

    n_params = len(scan_params)
    n_frames = logicals.shape[0]
    n_sites = logicals.shape[1]
    n_seqs = n_frames // num_images

    if seq_ids is None:
        seq_ids = np.arange(1, n_seqs + 1)
    seq_ids = np.asarray(seq_ids).ravel().astype(int)

    # Count reps per parameter
    reps_per_param = np.zeros(n_params, dtype=int)
    for k in range(len(seq_ids)):
        idx = seq_ids[k] - 1  # 0-indexed into Params
        if 0 <= idx < len(params_arr):
            p = params_arr[idx] - 1  # 0-indexed param
            if 0 <= p < n_params:
                reps_per_param[p] += 1
    max_reps = int(reps_per_param.max()) if np.any(reps_per_param > 0) else 0

    if max_reps == 0:
        return scan_params, np.zeros((n_sites, n_params, 0), dtype=bool), None

    # Fill logic arrays — sized to max_reps so every parameter uses all its
    # available repetitions.  Parameters with fewer reps leave trailing slots
    # as False; prob11_site_resolved's loaded>0 mask handles this correctly,
    # giving each parameter its own SEM based on its actual repetition count.
    logic1 = np.zeros((n_sites, n_params, max_reps), dtype=bool)
    logic2 = np.zeros((n_sites, n_params, max_reps), dtype=bool) if num_images >= 2 else None
    fill_count = np.zeros(n_params, dtype=int)

    for k in range(min(len(seq_ids), n_seqs)):
        idx = seq_ids[k] - 1
        if idx < 0 or idx >= len(params_arr):
            continue
        p = params_arr[idx] - 1
        if p < 0 or p >= n_params:
            continue
        if fill_count[p] >= max_reps:
            continue

        r = fill_count[p]
        base_row = k * num_images
        if base_row < n_frames:
            logic1[:, p, r] = logicals[base_row, :]
        if num_images >= 2 and base_row + 1 < n_frames:
            logic2[:, p, r] = logicals[base_row + 1, :]
        fill_count[p] += 1

    return scan_params, logic1, logic2
