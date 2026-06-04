"""Read existing MATLAB .mat files produced by the MATLAB DataManager.

Handles both v5 (.mat) and v7.3 (.mat = HDF5) formats.
"""

import os
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


def load_mat_file(path):
    """Load a .mat file, auto-detecting format.

    Returns
    -------
    data : dict
        Keys depend on file contents. Common keys:
        'Scan' (config struct), 'imgs' (images), 'logicals', 'seq_ids'.
    """
    # Try v7.3 (HDF5) first
    if h5py is not None:
        try:
            with h5py.File(path, 'r') as f:
                return _load_hdf5_mat(f)
        except Exception:
            pass

    # Fall back to scipy.io for v5
    if loadmat is not None:
        try:
            return loadmat(path, squeeze_me=True, struct_as_record=True)
        except Exception:
            pass

    raise RuntimeError(f"Cannot load {path}: install h5py and/or scipy")


def _dataset_nbytes(ds):
    """Estimate dataset size in bytes without reading it."""
    return ds.size * ds.dtype.itemsize


# 500 MB — datasets larger than this are skipped during bulk loads
_MAX_DATASET_BYTES = 500_000_000


def _load_hdf5_mat(f, max_bytes=_MAX_DATASET_BYTES):
    """Load variables from an HDF5-based .mat (v7.3) file.

    Datasets larger than *max_bytes* are replaced by a shape/dtype stub
    to avoid blowing up memory on files with huge image arrays.
    """
    data = {}
    for key in f.keys():
        if key.startswith('#'):
            continue
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            if _dataset_nbytes(obj) > max_bytes:
                data[key] = {'_skipped': True, 'shape': obj.shape, 'dtype': str(obj.dtype)}
                continue
            arr = obj[:]
            data[key] = arr
        elif isinstance(obj, h5py.Group):
            data[key] = _load_group_as_dict(obj, max_bytes=max_bytes)
    return data


def _load_group_as_dict(group, max_bytes=_MAX_DATASET_BYTES):
    """Recursively load an HDF5 group into a nested dict.

    Datasets larger than *max_bytes* are skipped.
    """
    d = {}
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            if _dataset_nbytes(obj) > max_bytes:
                continue
            d[key] = obj[:]
        elif isinstance(obj, h5py.Group):
            d[key] = _load_group_as_dict(obj, max_bytes=max_bytes)
    return d


def load_scan_config_from_mat(path):
    """Load just the Scan configuration struct from a .mat file.

    For HDF5 files, reads ONLY the Scan group (avoids loading 64GB+ image data).
    Large datasets (>500 MB) are always skipped.

    Returns
    -------
    config : dict
        Scan fields: frameSize, NumImages, boxSize, maskSigma,
        initThresholds, initGridLocationsX, initGridLocationsY,
        initInfidelities, isInit, histData, gaussFits, etc.
    """
    file_size = os.path.getsize(path) if os.path.isfile(path) else 0

    # Try HDF5 first — read only the Scan group
    if h5py is not None:
        try:
            with h5py.File(path, 'r') as f:
                if 'Scan' in f:
                    return _load_hdf5_group(f['Scan'])
                # No Scan group — load with size guard
                return _load_hdf5_mat(f)
        except Exception as e:
            # For large files there's no scipy fallback — report the real error
            if file_size > _MAX_DATASET_BYTES:
                raise RuntimeError(
                    f"Failed to read {file_size / 1e9:.1f} GB .mat file: {path}\n"
                    f"The file may still be syncing or is truncated.\n"
                    f"h5py error: {e}"
                ) from e
            # Small file — might be scipy-v5 .mat, fall through

    # scipy v5 .mat — only for small files
    if loadmat is not None:
        if file_size > _MAX_DATASET_BYTES:
            raise RuntimeError(
                f"File too large for scipy.loadmat ({file_size / 1e9:.1f} GB): {path}. "
                f"Requires h5py for v7.3 .mat files this large."
            )
        try:
            raw = loadmat(path, squeeze_me=True, struct_as_record=True)
            if 'Scan' in raw:
                scan = raw['Scan']
                if isinstance(scan, np.void):
                    return {name: scan[name] for name in scan.dtype.names}
                elif isinstance(scan, dict):
                    return scan
            return raw
        except Exception:
            pass

    raise RuntimeError(f"Cannot load {path}: install h5py and/or scipy")


def load_scan_config(mat_fname):
    """Load a scan config, preferring a pyctrl JSON sidecar over the MATLAB ``.mat``.

    The pyctrl backend (scan_prep.write_scan_config) writes ``<dir>/data_<stamp>.json`` instead
    of a MATLAB ``.mat`` (a Python backend has no reason to emit MATLAB binary). If that JSON is
    present it is loaded directly; otherwise we fall back to the MATLAB ``.mat`` reader, so
    MATLAB-written scans are unaffected. Returns a dict of ``Scan`` fields either way.
    """
    import json
    json_fname = os.path.splitext(mat_fname)[0] + '.json'
    if os.path.exists(json_fname):
        with open(json_fname) as f:
            return _config_arrays(json.load(f))
    return load_scan_config_from_mat(mat_fname)


def _config_arrays(obj):
    """Coerce numeric JSON lists to float ndarrays so a JSON-sourced config reads like a
    scipy-loaded ``.mat``.

    The ``.mat`` path returns numpy arrays; the config consumers built for it
    (``extract_scan_dims`` / ``_find_first_numeric``, which only recognize ``np.ndarray``
    leaves) would silently miss a pyctrl JSON config's swept-axis values (plain lists) and
    report no scan dimensions. Recursively: an all-numeric list -> ``np.ndarray`` (float);
    a list with dicts/strings -> recurse element-wise but stay a list; dicts recurse;
    scalars/strings pass through. (``bool`` is excluded so flag lists aren't floated.)"""
    if isinstance(obj, dict):
        return {k: _config_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            return np.asarray(obj, dtype=float)
        return [_config_arrays(x) for x in obj]
    return obj


def _load_hdf5_group(grp, max_bytes=_MAX_DATASET_BYTES):
    """Load an HDF5 group into a nested dict (only datasets + subgroups)."""
    result = {}
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            if _dataset_nbytes(item) > max_bytes:
                continue
            # Skip object-reference datasets (can't convert to plain arrays)
            if item.dtype == h5py.ref_dtype:
                continue
            try:
                result[key] = item[:]
            except Exception:
                pass
        elif isinstance(item, h5py.Group):
            result[key] = _load_hdf5_group(item, max_bytes=max_bytes)
    return result
