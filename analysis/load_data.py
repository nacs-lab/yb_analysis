"""Load scan data from HDF5 files.

Port of MATLAB's loadLatestScanData.m and loadScanDataFromPath.m.
"""

import os
import glob
import numpy as np

from yb_analysis.config import DATA_DIR
from yb_analysis.io.mat_reader import load_scan_config, load_scan_config_from_mat
from yb_analysis.io.scan_files import (
    LAYOUT_COMBINED, LAYOUT_SPLIT, image_source, resolve_scan_files)


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
        imgs : ndarray (nFrames, H, W) uint16 (int16 in legacy files) (if available, can be None for large files)
        seq_ids : ndarray (nSeqs,) int64
        path : str - the DATA file (scan identity; never the image file)
        image_path : str - where /imgs lives (== path for a combined scan)
        layout : str - 'combined' | 'split'
    """
    base = os.path.basename(scan_dir)

    # The resolver keeps the historical precedence (.h5 by basename, then the
    # data_*.h5 glob, then .mat) and additionally reports where /imgs lives.
    # probe_attrs=False: the shape logic below reads the attrs from the handle
    # it opens anyway, and we must not open the (multi-GB) image file here.
    sf = resolve_scan_files(scan_dir, probe_attrs=False)

    # Try HDF5 first (Python-generated)
    h5_path = os.path.join(scan_dir, base + '.h5')
    if os.path.isfile(h5_path):
        return _load_from_h5(h5_path, scan_dir, base, sf)

    # Fall back to .mat (MATLAB-generated)
    mat_path = os.path.join(scan_dir, base + '.mat')
    if os.path.isfile(mat_path):
        return _load_from_mat(mat_path)

    raise FileNotFoundError(f'No .h5 or .mat file in {scan_dir}')


def _load_from_h5(h5_path, scan_dir, base, sf=None):
    """Load from Python-generated HDF5 file.

    Detects two-array layout (``two_array=True`` file attr) and returns per-
    image logicals/intensities; legacy single-array files still produce the
    flat ``logicals`` / ``intensities`` arrays as before.

    ``sf`` is the ``ScanFiles`` record from ``resolve_scan_files`` (optional, so
    direct callers keep working): it says whether ``/imgs`` lives in this file
    (combined) or in a sibling ``image_<stamp>.h5`` (split).
    """
    import h5py

    if sf is None:
        sf = resolve_scan_files(scan_dir, probe_attrs=False)
    split = (sf.layout == LAYOUT_SPLIT)
    image_path = sf.image_path or h5_path

    mat_path = os.path.join(scan_dir, base + '.mat')
    json_path = os.path.join(scan_dir, base + '.json')
    # Prefer the pyctrl .json sidecar, fall back to the MATLAB .mat. Without
    # this, a pyctrl scan (no .mat) loads an EMPTY config — so unpack finds no
    # Params/ScanGroup and the analysis pane renders blank curves.
    if os.path.isfile(mat_path) or os.path.isfile(json_path):
        scan = load_scan_config(mat_path)
    else:
        scan = {}

    logicals = intensities = None
    logicals_img1 = logicals_img2 = None
    intensities_img1 = intensities_img2 = None
    # img2 detector provenance + per-site posterior "% certainty" (only present
    # when img2 was detected by the spot-shape model, distinct-pattern runs).
    logicals_img2_source = None
    certainties_img2 = None
    # Middle (verify) frame, NumImages >= 3: post-rearrangement occupancy.
    logicals_mid = intensities_mid = None

    with h5py.File(h5_path, 'r') as f:
        two_array = bool(f.attrs.get('two_array', False))
        if two_array:
            logicals_img1 = f['logicals_img1'][:] if 'logicals_img1' in f else None
            logicals_img2 = f['logicals_img2'][:] if 'logicals_img2' in f else None
            if 'intensities_img1' in f:
                intensities_img1 = f['intensities_img1'][:]
            if 'intensities_img2' in f:
                intensities_img2 = f['intensities_img2'][:]
            src = f.attrs.get('logicals_img2_source')
            logicals_img2_source = src.decode() if isinstance(src, bytes) else (
                str(src) if src is not None else None)
            if 'certainties_img2' in f:
                certainties_img2 = f['certainties_img2'][:]
            if 'logicals_mid' in f:
                logicals_mid = f['logicals_mid'][:]
            if 'intensities_mid' in f:
                intensities_mid = f['intensities_mid'][:]
        else:
            logicals = f['logicals'][:] if 'logicals' in f else None
            intensities = f['intensities'][:] if 'intensities' in f else None
        seq_ids = f['seq_ids'][:] if 'seq_ids' in f else None
        # Don't load imgs by default (can be huge)
        if not split:
            imgs_shape = f['imgs'].shape if 'imgs' in f else None
        else:
            # SPLIT layout: the analysis path must NOT open the image file, so
            # derive the shape from the DATA file's own attrs
            # (num_images_per_seq + frame_size) times the recorded seq rows.
            imgs_shape = _split_imgs_shape_from_attrs(f)

        # Load scan_config attrs
        if 'scan_config' in f:
            for k, v in f['scan_config'].attrs.items():
                if k not in scan:
                    scan[k] = v

    if split and imgs_shape is None:
        # The attrs our writer stamps are missing (hand-made / pre-attr file):
        # only now is a header-only open of the image file justified. Still
        # cheap (no pixels read); imgs_shape=None if that fails too, which is
        # an already-supported state downstream.
        imgs_shape = _imgs_shape_from_header(image_path)

    return {
        'Scan': scan,
        'two_array': two_array,
        'logicals': logicals,
        'intensities': intensities,
        'logicals_img1': logicals_img1,
        'logicals_img2': logicals_img2,
        'intensities_img1': intensities_img1,
        'intensities_img2': intensities_img2,
        'logicals_img2_source': logicals_img2_source,
        'certainties_img2': certainties_img2,
        'logicals_mid': logicals_mid,
        'intensities_mid': intensities_mid,
        'seq_ids': seq_ids.ravel() if seq_ids is not None else None,
        'imgs_shape': imgs_shape,
        'path': h5_path,
        # Where /imgs lives: the same file for a combined scan, the sibling
        # image_<stamp>.h5 for a split one. `path` stays the DATA file (the
        # scan identity every caller keys on).
        'image_path': image_path,
        'layout': sf.layout,
        'mat_path': mat_path if os.path.isfile(mat_path) else None,
    }


def _split_imgs_shape_from_attrs(f):
    """Derive ``/imgs`` shape for a SPLIT scan from the DATA file alone.

    ``(n_seq_rows * num_images_per_seq, H, W)`` using the data file's own
    ``num_images_per_seq`` / ``frame_size`` attrs and the recorded sequence
    rows in the same open handle. Returns None when either attr is missing
    (the caller then falls back to an image-file header open).
    """
    try:
        num_images = int(f.attrs['num_images_per_seq'])
        frame_size = tuple(int(v) for v in f.attrs['frame_size'])
    except (KeyError, TypeError, ValueError):
        return None
    if num_images < 1 or len(frame_size) != 2:
        return None
    n_rows = None
    for k in ('seq_ids', 'logicals_img1', 'logicals'):
        if k in f:
            n_rows = int(f[k].shape[0])
            break
    if n_rows is None:
        return None
    return (n_rows * num_images,) + frame_size


def _imgs_shape_from_header(image_path):
    """Header-only ``/imgs`` shape from the image file. None on any failure."""
    import h5py

    try:
        with h5py.File(image_path, 'r') as f:
            return f['imgs'].shape if 'imgs' in f else None
    except (OSError, KeyError):
        return None


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
        'logicals_mid': None,
        'intensities_mid': None,
        'seq_ids': seq_ids,
        'imgs_shape': imgs_shape,
        'path': mat_path,
        # A .mat scan is always the combined layout (no image sibling exists).
        'image_path': mat_path,
        'layout': LAYOUT_COMBINED,
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

    # Split layout: pixels live in the sibling image_<stamp>.h5. Redirecting
    # here keeps every existing caller (hist_init, the notebook, ...) correct
    # with no call-site change; identity for a combined scan or a .mat.
    data_path = image_source(data_path)
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

    data_path = image_source(data_path)     # split layout -> image_<stamp>.h5
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
