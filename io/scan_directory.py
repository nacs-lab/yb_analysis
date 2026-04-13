"""Date/time-stamped directory and filename utilities.

Port of scripts/DateTimeStampDirectory.m and DateTimeStampFilename.m
"""

import os
from datetime import datetime

from yb_analysis.config import PATH_PREFIX


def make_scan_dir(date_stamp=None, time_stamp=None, save_path=None, prefix='data'):
    """Create and return a timestamped scan directory path.

    E.g. N:\\NaCsLab\\Data\\20260403\\data_20260403_152030\\

    Parameters
    ----------
    date_stamp : str or None
        Date as 'YYYYMMDD'. None = use current date.
    time_stamp : str or None
        Time as 'HHMMSS'. None = use current time.
    save_path : str or None
        Override base directory. None = PATH_PREFIX/Data/YYYYMMDD/
    prefix : str
        Filename prefix (default 'data').

    Returns
    -------
    dname : str
        Full directory path (created if it doesn't exist).
    date_out : str
    time_out : str
    """
    now = datetime.now()
    date_out = date_stamp or now.strftime('%Y%m%d')
    time_out = time_stamp or now.strftime('%H%M%S')

    if save_path is None:
        save_path = os.path.join(PATH_PREFIX, 'Data', date_out)

    os.makedirs(save_path, exist_ok=True)
    dname = os.path.join(save_path, f'{prefix}_{date_out}_{time_out}')
    os.makedirs(dname, exist_ok=True)

    return dname, date_out, time_out


def make_scan_fname(date_stamp=None, time_stamp=None, save_path=None, prefix='data', ext='.mat'):
    """Create a timestamped filename.

    E.g. N:\\NaCsLab\\Data\\20260403\\data_20260403_152030.mat
    """
    now = datetime.now()
    date_out = date_stamp or now.strftime('%Y%m%d')
    time_out = time_stamp or now.strftime('%H%M%S')

    if save_path is None:
        save_path = os.path.join(PATH_PREFIX, 'Data', date_out)

    os.makedirs(save_path, exist_ok=True)
    fname = os.path.join(save_path, f'{prefix}_{date_out}_{time_out}{ext}')

    return fname, date_out, time_out


def scan_id_to_stamps(scan_id):
    """Convert a 14-digit scan ID to (date_stamp, time_stamp).

    The MATLAB convention: scan_id = int('YYYYMMDDHHMMSS').
    """
    s = str(int(scan_id))
    if len(s) != 14:
        raise ValueError(f"scan_id must be 14 digits, got {len(s)}: {s}")
    return s[:8], s[8:]
