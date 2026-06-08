"""Scan parameter analysis: group sequences by parameter, compute survival/loading curves.

Port of MATLAB's unpackScanLogicals + get_prob11 + get_loadingRate.
Supports both 1-D and N-D (e.g., 2-D) scans.
"""

import logging
import warnings
import numpy as np
import h5py

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Single-dimension helpers (backward compatible)
# ---------------------------------------------------------------------------

def extract_scan_params(config):
    """Extract scan parameter values from config's ScanGroup.base.vars (first dim).

    Returns ndarray (nParams,) or None.
    """
    dims = extract_scan_dims(config)
    if dims and len(dims) > 0:
        return dims[0]['values']
    return None


def extract_scan_name(config):
    """Extract the scan parameter name path for the first dimension.

    Returns str or None.
    """
    dims = extract_scan_dims(config)
    if dims and len(dims) > 0:
        return dims[0]['name']
    return None


# ---------------------------------------------------------------------------
#  Multi-dimension extraction
# ---------------------------------------------------------------------------

def extract_scan_dims(config):
    """Extract ALL scan dimensions from config's ScanGroup.base.vars.

    For a 1-D scan returns a single-element list; for a 2-D scan returns two, etc.

    Returns
    -------
    dims : list of dict  or  None
        Each dict has:
            'name'   : str    — dotted param path (e.g. 'Pushout.Green.Freq')
            'values' : ndarray (N,) — parameter values for that dimension
            'size'   : int    — number of points
        Ordered dim-0 first (dim-0 varies fastest in MATLAB column-major indexing).
    """
    sg = config.get('ScanGroup')
    if sg is None:
        return None
    base = sg.get('base') if isinstance(sg, dict) else None
    if base is None:
        return None
    vars_ = base.get('vars') if isinstance(base, dict) else None
    if not isinstance(vars_, dict):
        return None

    # --- Detect whether vars is a single struct (1-D) or list of structs (N-D) ---
    params = vars_.get('params')
    sizes = vars_.get('size')

    # If params is a list → each element is one dimension's param struct
    if isinstance(params, list):
        # N-D scan: params = [dim0_struct, dim1_struct, ...]
        dims = []
        for i, p in enumerate(params):
            val, path = _find_first_numeric(p, [])
            sz = int(np.asarray(sizes[i]).ravel()[0]) if sizes is not None and i < len(sizes) else (len(val) if val is not None else 0)
            if val is not None:
                dims.append({'name': '.'.join(path), 'values': val, 'size': sz})
        return dims if dims else None

    # If params is a dict → 1-D scan (inline struct)
    if isinstance(params, dict):
        val, path = _find_first_numeric(params, [])
        if val is None:
            return None
        sz = int(np.asarray(sizes).ravel()[0]) if sizes is not None else len(val)
        return [{'name': '.'.join(path), 'values': val, 'size': sz}]

    return None


def extract_scan_dims_h5(mat_path):
    """Extract scan dimensions directly from an HDF5 v7.3 .mat file.

    Handles the object-reference layout that MATLAB uses for struct arrays
    when ndims >= 2.

    Returns list of dict (same format as extract_scan_dims) or None.
    """
    try:
        with h5py.File(mat_path, 'r') as f:
            vars_params = f.get('Scan/ScanGroup/base/vars/params')
            vars_size = f.get('Scan/ScanGroup/base/vars/size')
            if vars_params is None:
                return None

            # 1-D scan: vars/params is an HDF5 Group (inline struct)
            if isinstance(vars_params, h5py.Group):
                val = _find_first_dataset_h5(vars_params)
                if val is None:
                    return None
                name = _find_first_dataset_path_h5(vars_params)
                sz = int(f['Scan/ScanGroup/base/vars/size'][()].ravel()[0]) if vars_size is not None else len(val)
                return [{'name': name, 'values': val, 'size': sz}]

            # N-D scan: vars/params is a Dataset of object references
            if isinstance(vars_params, h5py.Dataset) and vars_params.dtype == h5py.ref_dtype:
                refs = vars_params[()].ravel()
                size_refs = vars_size[()].ravel() if vars_size is not None else [None] * len(refs)
                dims = []
                for i, ref in enumerate(refs):
                    grp = f[ref]
                    val = _find_first_dataset_h5(grp) if isinstance(grp, h5py.Group) else grp[()].ravel().astype(np.float64)
                    name = _find_first_dataset_path_h5(grp) if isinstance(grp, h5py.Group) else f'dim{i}'
                    sz_ref = size_refs[i]
                    if sz_ref is not None:
                        sz_grp = f[sz_ref] if isinstance(sz_ref, h5py.Reference) else sz_ref
                        sz = int(np.asarray(sz_grp).ravel()[0]) if hasattr(sz_grp, '__array__') else int(sz_grp)
                    else:
                        sz = len(val) if val is not None else 0
                    if val is not None:
                        dims.append({'name': name, 'values': val, 'size': sz})
                return dims if dims else None

            return None
    except Exception as e:
        logger.debug('Could not extract scan dims from HDF5: %s', e)
        return None


# ---------------------------------------------------------------------------
#  Private tree-walk helpers
# ---------------------------------------------------------------------------

def _find_first_numeric(obj, path):
    """Recursively search a nested dict/array for the first numeric vector.

    Returns (array, path_list) or (None, []).
    """
    # A swept axis stored as a plain Python list/tuple (the pyctrl JSON
    # sidecar keeps swept values this way -- numeric like [50, 100] AND
    # boolean like [False, True] for bool kwargs e.g. model_bookend_pre).
    # Coerce to ndarray so bool/int/float sweeps all resolve their dotted
    # name instead of falling back to an "axisN" placeholder.
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
        except Exception:
            return None, []
        if arr.dtype.kind in ('b', 'i', 'u', 'f') and arr.ndim <= 2 and arr.size > 1:
            return arr.ravel().astype(np.float64), path
        return None, []
    if isinstance(obj, np.ndarray):
        # Skip arrays of object references or non-numeric dtypes (bool 'b'
        # is numeric here -- it's how boolean swept kwargs are stored).
        if obj.dtype.kind in ('O', 'V', 'U', 'S'):
            return None, []
        if obj.ndim <= 2 and obj.size > 1:
            try:
                return obj.ravel().astype(np.float64), path
            except (TypeError, ValueError):
                return None, []
        return None, []
    if isinstance(obj, dict):
        for k, v in obj.items():
            result, rpath = _find_first_numeric(v, path + [k])
            if result is not None:
                return result, rpath
    return None, []


def _find_first_dataset_h5(grp):
    """Recursively find the first dataset in an HDF5 group."""
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Dataset) and item.size > 1:
            return item[:].ravel().astype(np.float64)
        elif isinstance(item, h5py.Group):
            result = _find_first_dataset_h5(item)
            if result is not None:
                return result
    return None


def _find_first_dataset_path_h5(grp, prefix=''):
    """Recursively find the dotted path to the first dataset in an HDF5 group."""
    for key in grp:
        item = grp[key]
        p = f'{prefix}.{key}' if prefix else key
        if isinstance(item, h5py.Dataset) and item.size > 1:
            return p
        elif isinstance(item, h5py.Group):
            result = _find_first_dataset_path_h5(item, p)
            if result is not None:
                return result
    return prefix or 'unknown'


# ---------------------------------------------------------------------------
#  HDF5 backward-compat wrappers
# ---------------------------------------------------------------------------

def extract_scan_params_h5(mat_path):
    """Extract first-dimension scan parameters from an HDF5 .mat file."""
    dims = extract_scan_dims_h5(mat_path)
    if dims and len(dims) > 0:
        return dims[0]['values']
    return None


# ---------------------------------------------------------------------------
#  Compute scan curves (1-D and 2-D)
# ---------------------------------------------------------------------------

def _target_tp_per_flat(scan_logicals, param_indices, n_total, seq_targets):
    """Per-shot target-aware TP averaged per flat param index.

    Mirrors the offline ``_target_aware_from_lab_paths``: for each shot with a
    known target set (``seq_targets[seq_id]`` = lab-site indices from the diag),
    TP = (# target sites occupied in the final image) / (# target sites). Index
    into the logicals directly (no grid needed). Returns
    ``(mean, sem, n_reps)`` each length ``n_total``, or ``None`` when no shot
    had usable targets (→ caller falls back to per-site survival)."""
    sum_tp = np.zeros(n_total)
    sumsq = np.zeros(n_total)
    cnt = np.zeros(n_total, dtype=int)
    used = 0
    for seq_id, logic1, logic2 in scan_logicals:
        if logic2 is None:
            continue
        tgt = seq_targets.get(int(seq_id))
        if tgt is None:
            continue
        li = int(seq_id) - 1
        if li < 0 or li >= len(param_indices):
            continue
        p = int(param_indices[li]) - 1
        if p < 0 or p >= n_total:
            continue
        l2 = np.asarray(logic2)
        t = np.asarray(tgt, dtype=int)
        t = t[(t >= 0) & (t < l2.shape[0])]
        if t.size == 0:
            continue
        tp = float(l2[t].sum()) / t.size
        sum_tp[p] += tp
        sumsq[p] += tp * tp
        cnt[p] += 1
        used += 1
    if used == 0:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnt > 0, sum_tp / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 0, sumsq / np.maximum(cnt, 1) - mean ** 2, np.nan)
        sem = np.sqrt(np.maximum(var, 0.0) / np.maximum(cnt, 1))
    return mean, sem, cnt


def compute_scan_curve(scan_logicals, param_indices, scan_params, num_images,
                       scan_dims=None, is_two_array=False, recent_seq_ids=None,
                       seq_targets=None):
    """Compute survival, loading, or rearrangement curve from accumulated
    logicals.

    For 1-D scans (scan_dims is None or has 1 entry) returns the classic
    scatter-with-errorbars dict.

    For 2-D scans (scan_dims has 2 entries) returns a heatmap dict instead.

    Parameters
    ----------
    scan_logicals : list of (seq_id, logicals_img1, logicals_img2_or_None)
    param_indices : ndarray (nPlannedSequences,)
        Scan.Params: maps seq_id (1-indexed) → flat param index (1-indexed).
    scan_params : ndarray (nParams,)   [used for 1-D only]
    num_images : int
    scan_dims : list of dict or None
        From extract_scan_dims(); if len >= 2, produce a 2-D heatmap.
    is_two_array : bool
        Two-array mode (isGrid2=1). When True AND the two grids have
        **different** site counts, compute mean of ``logic2`` (rearrangement
        success rate). When the two grids have the **same** site count we
        assume grid-2 is just a translation of grid-1 (matched site indices)
        and fall back to the standard conditioned-survival calc. When
        False, behaves exactly as before (survival when NumImages=2,
        loading when NumImages=1).

    Returns
    -------
    dict
    """
    is_2d = scan_dims is not None and len(scan_dims) >= 2

    if is_2d:
        return _compute_2d(scan_logicals, param_indices, scan_dims, num_images,
                           is_two_array=is_two_array, recent_seq_ids=recent_seq_ids,
                           seq_targets=seq_targets)

    # --- 1-D path ---
    if not scan_logicals or scan_params is None or param_indices is None:
        return None

    n_params = len(scan_params)
    n_sites_1 = len(scan_logicals[0][1])
    has_logic2 = scan_logicals[0][2] is not None
    n_sites_2 = len(scan_logicals[0][2]) if has_logic2 else 0

    # Target-aware survival (matches the Analysis tab's TP): when the diag
    # supplied per-shot target sets, the live curve is per-shot TP, not
    # per-site survival. Falls through to per-site when no targets apply.
    if seq_targets and num_images >= 2 and has_logic2:
        ta = _target_tp_per_flat(scan_logicals, param_indices, n_params, seq_targets)
        if ta is not None:
            mean, sem, cnt = ta
            order = np.argsort(scan_params)
            return {
                'scan_x': scan_params[order],
                'y_mean': mean[order],
                'y_sem': sem[order],
                'y_mean_sr': mean[order][None, :],
                'n_reps': cnt[order],
                'mode': 'survival',
                'target_aware': True,
            }
    # In two-array mode with different-sized grids, the per-param metric is
    # the mean of logic2 over array-2 sites. With same-sized grids, treat as
    # matched-index survival like the legacy single-array case.
    different_arrays = is_two_array and has_logic2 and n_sites_1 != n_sites_2
    n_sites = n_sites_2 if different_arrays else n_sites_1

    buckets = [[] for _ in range(n_params)]
    for seq_id, logic1, logic2 in scan_logicals:
        idx = int(seq_id) - 1
        if idx < 0 or idx >= len(param_indices):
            continue
        p = int(param_indices[idx]) - 1
        if p < 0 or p >= n_params:
            continue
        buckets[p].append((logic1, logic2))

    # For num_images >= 2 the seq_logicals tuple stores (logic1, last_frame),
    # so "survival" here is the conditional probability of being loaded in
    # the FINAL frame given loaded in img1 — regardless of how many
    # intermediate captures (e.g. two-round SLM rearrangement) sat between.
    if different_arrays and num_images >= 2:
        mode = 'rearrangement'
        y_mean_sr, y_sem_sr, n_reps = _rearrangement_buckets(
            buckets, n_sites, n_params)
    elif num_images >= 2:
        mode = 'survival'
        y_mean_sr, y_sem_sr, n_reps = _survival_buckets(buckets, n_sites, n_params)
    else:
        mode = 'loading'
        y_mean_sr, y_sem_sr, n_reps = _loading_buckets(buckets, n_sites, n_params)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # nanmean on empty slices
        y_mean = np.nanmean(y_mean_sr, axis=0)
    y_sem = np.sqrt(np.nansum(y_sem_sr**2, axis=0)) / n_sites

    order = np.argsort(scan_params)
    return {
        'scan_x': scan_params[order],
        'y_mean': y_mean[order],
        'y_sem': y_sem[order],
        'y_mean_sr': y_mean_sr[:, order],
        'n_reps': n_reps[order],
        'mode': mode,
    }


def _compute_2d(scan_logicals, param_indices, scan_dims, num_images,
                is_two_array=False, recent_seq_ids=None, seq_targets=None):
    """Compute 2-D heatmap for multi-dimensional scans.

    The flat param_index decomposes column-major (dim-0 varies fastest):
        idx0 = (param_idx - 1) % size0
        idx1 = (param_idx - 1) // size0
    """
    if not scan_logicals or param_indices is None:
        return None

    d0, d1 = scan_dims[0], scan_dims[1]
    s0, s1 = d0['size'], d1['size']
    n_total = s0 * s1
    n_sites_1 = len(scan_logicals[0][1])
    has_logic2 = scan_logicals[0][2] is not None
    n_sites_2 = len(scan_logicals[0][2]) if has_logic2 else 0
    different_arrays = is_two_array and has_logic2 and n_sites_1 != n_sites_2
    n_sites = n_sites_2 if different_arrays else n_sites_1

    # Target-aware survival (matches Analysis TP) when the diag gave targets.
    if seq_targets and num_images >= 2 and has_logic2:
        ta = _target_tp_per_flat(scan_logicals, param_indices, n_total, seq_targets)
        if ta is not None:
            mean_flat, sem_flat, n_flat = ta
            sids = list(recent_seq_ids) if recent_seq_ids else (
                [int(scan_logicals[-1][0])] if scan_logicals else [])
            current = []
            seen = set()
            for sid in sids:
                li = int(sid) - 1
                if not (0 <= li < len(param_indices)):
                    continue
                p = int(param_indices[li]) - 1
                if not (0 <= p < n_total):
                    continue
                cell = (p % s0, p // s0)
                if cell not in seen:
                    seen.add(cell)
                    current.append({'x_idx': cell[0], 'y_idx': cell[1]})
            return {
                'mode': 'survival', 'ndim': 2, 'target_aware': True,
                'heatmap': mean_flat.reshape(s1, s0),
                'sem': sem_flat.reshape(s1, s0),
                'n_reps': n_flat.reshape(s1, s0),
                'x_values': d0['values'], 'y_values': d1['values'],
                'x_name': d0['name'], 'y_name': d1['name'],
                'x_size': s0, 'y_size': s1, 'current': current,
            }

    # Bucket by flat param index (0-based)
    buckets = [[] for _ in range(n_total)]
    for seq_id, logic1, logic2 in scan_logicals:
        idx = int(seq_id) - 1
        if idx < 0 or idx >= len(param_indices):
            continue
        p = int(param_indices[idx]) - 1
        if p < 0 or p >= n_total:
            continue
        buckets[p].append((logic1, logic2))

    if different_arrays and num_images >= 2:
        mode = 'rearrangement'
        y_mean_sr, y_sem_sr, n_reps = _rearrangement_buckets(
            buckets, n_sites, n_total)
    elif num_images >= 2:
        mode = 'survival'
        y_mean_sr, y_sem_sr, n_reps = _survival_buckets(buckets, n_sites, n_total)
    else:
        mode = 'loading'
        y_mean_sr, y_sem_sr, n_reps = _loading_buckets(buckets, n_sites, n_total)

    # Site-average
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # nanmean on empty slices
        y_mean_flat = np.nanmean(y_mean_sr, axis=0)  # (n_total,)

    # Per-cell error of the site-averaged value, propagated from the per-site
    # binomial standard errors exactly like the 1-D curve's error bars:
    #   value_cell = mean_s p_s,   SE_cell = sqrt(Σ_s SE_s²) / n_sites
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sem_flat = np.sqrt(np.nansum(y_sem_sr**2, axis=0)) / max(n_sites, 1)

    # Reshape into (s1, s0) grid → heatmap[dim1_idx, dim0_idx]
    # Column-major: dim0 varies fastest in the flat array
    heatmap = y_mean_flat.reshape(s1, s0)
    n_grid = n_reps.reshape(s1, s0)
    sem_grid = sem_flat.reshape(s1, s0)

    # Cells currently being scanned = every sequence completed in the most
    # recent batch (one dashboard update can carry several sequences). Map each
    # seq_id's flat param index back to (dim0, dim1) grid coords. Fall back to
    # the last measured sequence so a box still shows on a quiet refresh.
    sids = list(recent_seq_ids) if recent_seq_ids else (
        [int(scan_logicals[-1][0])] if scan_logicals else [])
    current = []
    seen = set()
    for sid in sids:
        li = int(sid) - 1
        if not (0 <= li < len(param_indices)):
            continue
        p = int(param_indices[li]) - 1
        if not (0 <= p < n_total):
            continue
        cell = (p % s0, p // s0)
        if cell not in seen:
            seen.add(cell)
            current.append({'x_idx': cell[0], 'y_idx': cell[1]})

    return {
        'mode': mode,
        'ndim': 2,
        'heatmap': heatmap,
        'sem': sem_grid,  # per-cell standard error of the site-averaged value
        'n_reps': n_grid,
        'x_values': d0['values'],  # dim0 → x-axis
        'y_values': d1['values'],  # dim1 → y-axis
        'x_name': d0['name'],
        'y_name': d1['name'],
        'x_size': s0,
        'y_size': s1,
        'current': current,  # list of {x_idx, y_idx} cells to highlight
    }


# ---------------------------------------------------------------------------
#  Shared bucket computation
# ---------------------------------------------------------------------------

def _survival_buckets(buckets, n_sites, n_params):
    """Compute survival (prob11) for each param bucket."""
    y_mean_sr = np.full((n_sites, n_params), np.nan)
    y_sem_sr = np.full((n_sites, n_params), np.nan)
    n_reps = np.zeros(n_params, dtype=int)

    for p in range(n_params):
        if not buckets[p]:
            continue
        reps = len(buckets[p])
        n_reps[p] = reps
        l1 = np.array([b[0] for b in buckets[p]])
        l2 = np.array([b[1] for b in buckets[p]])
        joint = (l1 & l2).sum(axis=0)
        loaded = l1.sum(axis=0)
        mask = loaded > 0
        p11 = np.where(mask, joint / np.maximum(loaded, 1), np.nan)
        se = np.where(mask, np.sqrt(p11 * (1 - p11) / np.maximum(loaded, 1)), np.nan)
        y_mean_sr[:, p] = p11
        y_sem_sr[:, p] = se

    return y_mean_sr, y_sem_sr, n_reps


def _loading_buckets(buckets, n_sites, n_params):
    """Compute loading rate for each param bucket."""
    y_mean_sr = np.full((n_sites, n_params), np.nan)
    y_sem_sr = np.full((n_sites, n_params), np.nan)
    n_reps = np.zeros(n_params, dtype=int)

    for p in range(n_params):
        if not buckets[p]:
            continue
        reps = len(buckets[p])
        n_reps[p] = reps
        l1 = np.array([b[0] for b in buckets[p]])
        prob = l1.mean(axis=0)
        se = np.sqrt(prob * (1 - prob) / reps)
        y_mean_sr[:, p] = prob
        y_sem_sr[:, p] = se

    return y_mean_sr, y_sem_sr, n_reps


def _rearrangement_buckets(buckets, n_sites, n_params):
    """Compute mean of logic2 (rearrangement success rate) per param bucket.

    Used in two-array mode (isGrid2=1) where image-2 captures a defect-free
    target array. n_sites is the img2 grid size.
    """
    y_mean_sr = np.full((n_sites, n_params), np.nan)
    y_sem_sr = np.full((n_sites, n_params), np.nan)
    n_reps = np.zeros(n_params, dtype=int)

    for p in range(n_params):
        if not buckets[p]:
            continue
        # Skip reps that don't have an image-2 logical (single-image seqs)
        l2_list = [b[1] for b in buckets[p] if b[1] is not None]
        if not l2_list:
            continue
        reps = len(l2_list)
        n_reps[p] = reps
        l2 = np.array(l2_list)
        prob = l2.mean(axis=0)
        se = np.sqrt(prob * (1 - prob) / reps)
        y_mean_sr[:, p] = prob
        y_sem_sr[:, p] = se

    return y_mean_sr, y_sem_sr, n_reps
