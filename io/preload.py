"""Pre-load background data from day folder or most recent scan config.

Day folder files (authoritative, updated every 200 shots):
  Data/YYYYMMDD/gridLocations.txt   — (M, 2) array [Y, X]
  Data/YYYYMMDD/threshold.mat       — thresholds (M,), infidelities (M,), gaussFitsStruct (M,)
  Data/YYYYMMDD/histData.mat        — histData (M,) struct array with .counts and .bin_centers

Fallback: most recent scan config .mat file's initThresholds, initGridLocations, histData.
"""

import os
import glob
import logging
from datetime import datetime, timedelta

import numpy as np

from yb_analysis.config import PATH_PREFIX

logger = logging.getLogger(__name__)


def _mat_scalar(val, default=0):
    if isinstance(val, np.ndarray):
        return val.flat[0]
    return val if val is not None else default


def _mat_vector(val):
    return np.asarray(val, dtype=np.float64).ravel()


def _today_day_dir():
    return os.path.join(PATH_PREFIX, 'Data', datetime.now().strftime('%Y%m%d'))


def load_background_data():
    """Auto-select and load background data from the most recent valid day folder.

    Walks back up to 30 days from today, then falls back to the latest scan
    config .mat file.

    Returns (data, source_dir) where source_dir is the Data/YYYYMMDD path used
    (or None if loaded from a scan config fallback), and data is the plot-ready
    dict (or None if nothing was found).
    """
    data_dir = os.path.join(PATH_PREFIX, 'Data')
    for delta in range(30):
        day = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
        day_dir = os.path.join(data_dir, day)
        data, _ = load_from_dir(day_dir)
        if data is not None:
            return data, day_dir

    data = _load_from_scan_config()
    return data, None


def load_from_dir(day_dir):
    """Load grid + thresholds from a specific directory.

    Returns (data_dict, status_msg).  data_dict is None on failure and
    status_msg explains what happened or how many sites were loaded.
    """
    if not os.path.isdir(day_dir):
        return None, 'Folder not found'

    grid_file = os.path.join(day_dir, 'gridLocations.txt')
    thresh_file = os.path.join(day_dir, 'threshold.mat')
    hist_file = os.path.join(day_dir, 'histData.mat')

    if not os.path.isfile(grid_file):
        return None, 'Missing gridLocations.txt'
    if not os.path.isfile(thresh_file):
        return None, 'Missing threshold.mat'

    logger.info('Loading background from day folder: %s', day_dir)

    try:
        grid = np.loadtxt(grid_file, skiprows=1)
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
    except Exception as e:
        return None, f'Failed to load grid: {e}'

    try:
        from scipy.io import loadmat
        td = loadmat(thresh_file, squeeze_me=True)
        thresholds = np.asarray(td['thresholds'], dtype=np.float64).ravel()
        infidelities = np.asarray(td['infidelities'], dtype=np.float64).ravel()
        gauss_fits = _parse_gauss_fits_struct(td.get('gaussFitsStruct'))
    except Exception as e:
        return None, f'Failed to load thresholds: {e}'

    num_sites = len(thresholds)

    bg_hist_data = None
    if os.path.isfile(hist_file):
        try:
            from scipy.io import loadmat as lm
            hd = lm(hist_file, squeeze_me=True)
            bg_hist_data = _parse_hist_data_struct(hd.get('histData'), num_sites)
        except Exception as e:
            logger.warning('Failed to load histData: %s', e)

    data = _build_plot_data(grid, thresholds, infidelities, gauss_fits,
                            bg_hist_data, num_sites)
    return data, f'{num_sites} sites loaded'


def _load_from_day_folder():
    """Walk back up to 30 days and return the first valid day folder's data."""
    data_dir = os.path.join(PATH_PREFIX, 'Data')
    for delta in range(30):
        day = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
        day_dir = os.path.join(data_dir, day)
        data, _ = load_from_dir(day_dir)
        if data is not None:
            return data
    return None


def _load_from_scan_config():
    """Fallback: load from the most recent scan .mat config."""
    data_dir = os.path.join(PATH_PREFIX, 'Data')
    if not os.path.isdir(data_dir):
        return None

    # Search recent date folders
    date_dirs = sorted(glob.glob(os.path.join(data_dir, '2*')), reverse=True)
    for dd in date_dirs[:5]:
        scans = sorted(glob.glob(os.path.join(dd, 'data_*', 'data_*.mat')))
        if not scans:
            continue

        mat_file = scans[-1]
        logger.info('Loading background from scan config: %s', mat_file)

        try:
            from yb_analysis.io.mat_reader import load_mat_file
            raw = load_mat_file(mat_file)
        except Exception as e:
            logger.warning('Failed to load %s: %s', mat_file, e)
            continue

        scan = raw.get('Scan', raw)
        if not isinstance(scan, dict):
            continue

        thresholds = _mat_vector(scan.get('initThresholds', []))
        if len(thresholds) == 0:
            continue

        num_sites = len(thresholds)
        grid_x = _mat_vector(scan.get('initGridLocationsX', []))
        grid_y = _mat_vector(scan.get('initGridLocationsY', []))
        grid = np.column_stack([grid_y, grid_x]) if len(grid_x) > 0 else None

        infidelities = _mat_vector(scan.get('initInfidelities', np.full(num_sites, np.nan)))

        # histData from scan config (MATLAB struct array in HDF5)
        bg_hist_data = None
        raw_hd = scan.get('histData')
        if raw_hd is not None:
            bg_hist_data = _parse_hist_data_struct(raw_hd, num_sites)

        gauss_fits = None
        raw_gf = scan.get('gaussFits')
        if raw_gf is not None:
            gauss_fits = _parse_gauss_fits_struct(raw_gf)

        return _build_plot_data(grid, thresholds, infidelities, gauss_fits,
                                bg_hist_data, num_sites)

    logger.warning('No previous scan data found in any date folder under %s', data_dir)
    return None


def _build_plot_data(grid, thresholds, infidelities, gauss_fits, bg_hist_data, num_sites):
    """Construct a plot-ready dict from loaded components."""
    box_size = 9  # default, will be overridden when DataManager starts

    return {
        'is_init': False,
        'cur_image': None,
        'cur_intensities': None,
        'logicals': None,
        'thresholds': thresholds,
        'infidelities': infidelities,
        'grid_locations': grid,
        'loading_rates': np.zeros(num_sites),
        'box_size': box_size,
        'num_sites': num_sites,
        'live_hist_data': None,
        'live_gauss_fits': None,
        'loaded_gauss_fits': gauss_fits,
        'hist_version': 0,
        'hist_rep_sites': list(range(min(4, num_sites))),
        'grid_shift_history': [],
        'grid_shift_heatmap': None,
        'n_accum_shots': 0,
    }


def _parse_hist_data_struct(raw, num_sites):
    """Parse MATLAB histData struct array → list of {'counts': ..., 'bin_centers': ...}."""
    if raw is None:
        return None

    # Already in our format
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        return raw

    # scipy loadmat with squeeze_me=True: structured ndarray (M,) with dtype [('counts','O'),('bin_centers','O')]
    if isinstance(raw, np.ndarray) and raw.dtype.names is not None:
        if 'counts' in raw.dtype.names and 'bin_centers' in raw.dtype.names:
            result = []
            for s in range(min(num_sites, len(raw))):
                counts = np.asarray(raw[s]['counts'], dtype=np.float64).ravel()
                centers = np.asarray(raw[s]['bin_centers'], dtype=np.float64).ravel()
                result.append({'counts': counts, 'bin_centers': centers})
            return result if result else None

    # HDF5 group (dict of arrays)
    if isinstance(raw, dict):
        counts_data = raw.get('counts')
        centers_data = raw.get('bin_centers')
        if counts_data is not None and centers_data is not None:
            try:
                ca = np.asarray(counts_data, dtype=np.float64)
                ba = np.asarray(centers_data, dtype=np.float64)
                if ca.ndim == 2:
                    return [{'counts': ca[s], 'bin_centers': ba[s]}
                            for s in range(min(num_sites, ca.shape[0]))]
            except Exception:
                pass

    return None


def _parse_gauss_fits_struct(raw):
    """Parse MATLAB gaussFitsStruct → list of {'params': ndarray[6]}."""
    if raw is None:
        return None

    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        return raw

    # scipy: structured ndarray (M,) with dtype [('params','O')]
    if isinstance(raw, np.ndarray) and raw.dtype.names is not None:
        if 'params' in raw.dtype.names:
            result = []
            for s in range(len(raw)):
                p = raw[s]['params']
                if p is not None and len(p) > 0:
                    result.append({'params': np.asarray(p, dtype=np.float64).ravel()})
                else:
                    result.append({'params': None})
            return result if result else None

    # HDF5: dict with 'params' key
    if isinstance(raw, dict):
        params = raw.get('params')
        if params is not None:
            try:
                arr = np.asarray(params, dtype=np.float64)
                if arr.ndim == 2:
                    return [{'params': arr[s]} for s in range(arr.shape[0])]
            except (TypeError, ValueError):
                # h5py object references can't be converted to float
                pass

    return None
