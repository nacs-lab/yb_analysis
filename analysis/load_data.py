"""Load scan data from HDF5 files.

Port of MATLAB's loadLatestScanData.m and loadScanDataFromPath.m.
"""

import os
import glob
import numpy as np

from yb_analysis.config import DATA_DIR
from yb_analysis.io.mat_reader import load_scan_config_from_mat


def load_latest_scan(data_dir=None, date=None):
    """Load the most recent scan data.

    Parameters
    ----------
    data_dir : str, optional
        Override base data directory.
    date : str, optional
        Date string 'YYYYMMDD'. None = today.

    Returns
    -------
    dict with keys: Scan, logicals, imgs, seq_ids, path
    """
    data_dir = data_dir or DATA_DIR
    if date is None:
        # Find most recent date folder
        dates = sorted(glob.glob(os.path.join(data_dir, '2*')))
        if not dates:
            raise FileNotFoundError(f'No date folders in {data_dir}')
        date_dir = dates[-1]
    else:
        date_dir = os.path.join(data_dir, date)

    # Find most recent scan in date folder
    scans = sorted(glob.glob(os.path.join(date_dir, 'data_*')))
    if not scans:
        raise FileNotFoundError(f'No scans in {date_dir}')
    return load_scan_from_path(scans[-1])


def load_scan_from_path(scan_dir):
    """Load scan data from a specific scan directory.

    Reads from HDF5 (.h5) if available, falls back to .mat.

    Parameters
    ----------
    scan_dir : str
        Path to scan directory (e.g., data_20260406_101610/)

    Returns
    -------
    dict with keys:
        Scan : dict — scan config
        logicals : ndarray (nFrames, nSites) bool
        intensities : ndarray (nFrames, nSites) float64 (if available)
        imgs : ndarray (nFrames, H, W) int16 (if available, can be None for large files)
        seq_ids : ndarray (nSeqs,) int64
        path : str
    """
    base = os.path.basename(scan_dir)

    # Try HDF5 first (Python-generated)
    h5_path = os.path.join(scan_dir, base + '.h5')
    if os.path.isfile(h5_path):
        return _load_from_h5(h5_path, scan_dir, base)

    # Fall back to .mat (MATLAB-generated)
    mat_path = os.path.join(scan_dir, base + '.mat')
    if os.path.isfile(mat_path):
        return _load_from_mat(mat_path)

    raise FileNotFoundError(f'No .h5 or .mat file in {scan_dir}')


def _load_from_h5(h5_path, scan_dir, base):
    """Load from Python-generated HDF5 file.

    Detects two-array layout (``two_array=True`` file attr) and returns per-
    image logicals/intensities; legacy single-array files still produce the
    flat ``logicals`` / ``intensities`` arrays as before.
    """
    import h5py

    mat_path = os.path.join(scan_dir, base + '.mat')
    scan = load_scan_config_from_mat(mat_path) if os.path.isfile(mat_path) else {}

    logicals = intensities = None
    logicals_img1 = logicals_img2 = None
    intensities_img1 = intensities_img2 = None

    with h5py.File(h5_path, 'r') as f:
        two_array = bool(f.attrs.get('two_array', False))
        if two_array:
            logicals_img1 = f['logicals_img1'][:] if 'logicals_img1' in f else None
            logicals_img2 = f['logicals_img2'][:] if 'logicals_img2' in f else None
            if 'intensities_img1' in f:
                intensities_img1 = f['intensities_img1'][:]
            if 'intensities_img2' in f:
                intensities_img2 = f['intensities_img2'][:]
        else:
            logicals = f['logicals'][:] if 'logicals' in f else None
            intensities = f['intensities'][:] if 'intensities' in f else None
        seq_ids = f['seq_ids'][:] if 'seq_ids' in f else None
        # Don't load imgs by default (can be huge)
        imgs_shape = f['imgs'].shape if 'imgs' in f else None

        # Load scan_config attrs
        if 'scan_config' in f:
            for k, v in f['scan_config'].attrs.items():
                if k not in scan:
                    scan[k] = v

    return {
        'Scan': scan,
        'two_array': two_array,
        'logicals': logicals,
        'intensities': intensities,
        'logicals_img1': logicals_img1,
        'logicals_img2': logicals_img2,
        'intensities_img1': intensities_img1,
        'intensities_img2': intensities_img2,
        'seq_ids': seq_ids.ravel() if seq_ids is not None else None,
        'imgs_shape': imgs_shape,
        'path': h5_path,
        'mat_path': mat_path if os.path.isfile(mat_path) else None,
    }


def _load_from_mat(mat_path):
    """Load from MATLAB-generated .mat file (HDF5 v7.3)."""
    import h5py

    scan = load_scan_config_from_mat(mat_path)

    with h5py.File(mat_path, 'r') as f:
        logicals = f['logicals'][:].T if 'logicals' in f else None  # MATLAB: (nSites, nFrames) → (nFrames, nSites)
        seq_ids = f['seq_ids'][:].ravel().astype(int) if 'seq_ids' in f else None
        imgs_shape = f['imgs'].shape if 'imgs' in f else None

    return {
        'Scan': scan,
        'two_array': False,
        'logicals': logicals,
        'intensities': None,  # MATLAB doesn't save intensities
        'logicals_img1': None,
        'logicals_img2': None,
        'intensities_img1': None,
        'intensities_img2': None,
        'seq_ids': seq_ids,
        'imgs_shape': imgs_shape,
        'path': mat_path,
        'mat_path': mat_path,
    }


def load_images(data_path, frames=None):
    """Lazily load image frames from an HDF5 or .mat file.

    Use this instead of loading the full ``imgs`` array when the file is
    too large to fit in memory.

    Parameters
    ----------
    data_path : str
        Path returned by ``load_scan_from_path`` (the ``.h5`` or ``.mat`` file).
    frames : int, slice, list[int], or None
        Which frames to load.  Examples::

            load_images(path, 0)           # single frame → (H, W)
            load_images(path, slice(0,10)) # first 10 → (10, H, W)
            load_images(path, [0, 5, 10])  # specific frames → (3, H, W)
            load_images(path)              # ALL frames (careful!)

    Returns
    -------
    imgs : ndarray
    """
    import h5py

    with h5py.File(data_path, 'r') as f:
        ds = f['imgs']
        if frames is None:
            return ds[:]
        if isinstance(frames, int):
            return ds[frames]
        if isinstance(frames, slice):
            return ds[frames]
        # list / array of indices — h5py needs sorted fancy index
        idx = np.asarray(frames)
        order = np.argsort(idx)
        imgs_sorted = ds[idx[order]]
        # restore original order
        restore = np.argsort(order)
        return imgs_sorted[restore]


def get_images_shape(data_path):
    """Return the shape of the imgs dataset without loading it."""
    import h5py

    with h5py.File(data_path, 'r') as f:
        if 'imgs' in f:
            return f['imgs'].shape
    return None


def list_scans(data_dir=None, date=None):
    """List all scans for a given date.

    Returns list of (scan_dir, timestamp) tuples, sorted by time.
    """
    data_dir = data_dir or DATA_DIR
    if date is None:
        dates = sorted(glob.glob(os.path.join(data_dir, '2*')))
        if not dates:
            return []
        date_dir = dates[-1]
    else:
        date_dir = os.path.join(data_dir, date)

    scans = sorted(glob.glob(os.path.join(date_dir, 'data_*')))
    result = []
    for s in scans:
        base = os.path.basename(s)
        parts = base.split('_')
        timestamp = '_'.join(parts[1:]) if len(parts) >= 3 else base
        result.append((s, timestamp))
    return result
