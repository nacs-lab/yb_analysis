"""Unpack scan logicals: group sequences by scan parameter.

Port of MATLAB's unpackScanLogicals.m.
"""

import numpy as np

from yb_analysis.detection.scan_analysis import (
    extract_scan_params, extract_scan_params_h5,
    extract_scan_dims, extract_scan_dims_h5,
)


def unpack_scan_logicals(scan, logicals=None, seq_ids=None, mat_path=None,
                         logicals_img1=None, logicals_img2=None):
    """Unpack scanned parameter values and logical results.

    Two input modes:

    * **Single-array (legacy):** pass ``logicals`` of shape ``(nFrames, nSites)``
      where ``nFrames = NumImages * nCapturedSequences`` (image-1 and image-2
      interleaved). ``logic1`` and ``logic2`` come back with the same
      ``nSites``. ``logic2`` is ``None`` when NumImages=1.

    * **Two-array (isGrid2=1):** pass ``logicals_img1`` of shape
      ``(NSeqs, M1)`` and ``logicals_img2`` of shape ``(NSeqs, M2)`` (one row
      per captured sequence per image, already de-interleaved). ``logic1``
      has shape ``(M1, nParams, maxReps)`` and ``logic2`` has shape
      ``(M2, nParams, maxReps)`` — different first dimension.

    Handles scrambled sequence order via seq_ids + ``Scan.Params``.

    Returns
    -------
    scan_params : ndarray (nParams,) or (nParams, 2)
        Unique parameter values (sorted) — flat for 1-D scans, paired for 2-D.
    logic1 : ndarray (nSites_img1, nParams, maxReps) bool
    logic2 : ndarray (nSites_img2, nParams, maxReps) bool or None
    reps_per_param : ndarray (nParams,) int
        Actual number of repetitions recorded for each parameter point. With
        non-uniform reps (e.g. mid-scan abort, scrambled scans), trailing
        slots in the logic arrays are padded with ``False``; downstream code
        that computes loading rate per-param MUST divide by this rather than
        by ``logic1.shape[2] = max_reps``.
    """
    two_array = logicals_img1 is not None and logicals_img2 is not None
    if not two_array and logicals is None:
        raise ValueError(
            'logicals is None — scan may still be running or data not yet '
            'saved. Pass `logicals` (single-array) or `logicals_img1` + '
            '`logicals_img2` (two-array).')

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

    if two_array:
        n_seqs = logicals_img1.shape[0]
        n_sites_1 = logicals_img1.shape[1]
        n_sites_2 = logicals_img2.shape[1]
    else:
        n_frames = logicals.shape[0]
        n_sites_1 = n_sites_2 = logicals.shape[1]
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
        empty_l1 = np.zeros((n_sites_1, n_params, 0), dtype=bool)
        empty_l2 = (np.zeros((n_sites_2, n_params, 0), dtype=bool)
                    if two_array or num_images >= 2 else None)
        return scan_params, empty_l1, empty_l2, reps_per_param

    # Fill logic arrays — sized to max_reps so every parameter uses all its
    # available repetitions.  Parameters with fewer reps leave trailing slots
    # as False; downstream loaded>0 / reps masks handle this correctly,
    # giving each parameter its own SEM based on its actual repetition count.
    logic1 = np.zeros((n_sites_1, n_params, max_reps), dtype=bool)
    if two_array:
        logic2 = np.zeros((n_sites_2, n_params, max_reps), dtype=bool)
    else:
        logic2 = (np.zeros((n_sites_1, n_params, max_reps), dtype=bool)
                  if num_images >= 2 else None)
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
        if two_array:
            logic1[:, p, r] = logicals_img1[k, :]
            logic2[:, p, r] = logicals_img2[k, :]
        else:
            base_row = k * num_images
            if base_row < n_frames:
                logic1[:, p, r] = logicals[base_row, :]
            # logic2 = FINAL frame of the seq (== row + 1 for pSeq=2, ==
            # row + 2 for pSeq=3, etc). This keeps "survival" defined as
            # initial -> final regardless of how many intermediate
            # captures the seq took (e.g. multi-round rearrangement, where
            # the middle frame is just a diagnostic).
            last_row = base_row + num_images - 1
            if num_images >= 2 and last_row < n_frames:
                logic2[:, p, r] = logicals[last_row, :]
        fill_count[p] += 1

    return scan_params, logic1, logic2, reps_per_param
