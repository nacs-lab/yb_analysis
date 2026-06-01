"""Enumerate completed scans on the lab PC.

Walks ``PATH_PREFIX/Data/<YYYYMMDD>/data_<YYYYMMDD>_<HHMMSS>/`` and
returns one row per scan. Best-effort metadata extracted from the .mat
sidecar; missing fields surface as ``None`` rather than failing the
listing.

Used by the dashboard ``/api/runs/list`` endpoint.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from yb_analysis import config as _yb_cfg

logger = logging.getLogger(__name__)

_SCAN_DIR_RE = re.compile(r'^data_(\d{8})_(\d{6})$')


def list_runs(*, since_days: Optional[int] = None,
              max_count: int = 500,
              with_meta: bool = True,
              include_incomplete: bool = False) -> List[dict]:
    """Return a list of scans found under the lab's data directory.

    Args:
      since_days: only include scans whose YYYYMMDD date is within this
                  many days of today. ``None`` = no cutoff.
      max_count:  cap on returned rows (newest first).
      with_meta:  if True, try to load the .mat sidecar for each scan
                  to populate ``name``, ``n_shots``, etc. Adds a few ms
                  per scan; disable for fast directory listings.
      include_incomplete: if False (default), skip scan dirs that don't
                  have BOTH the .h5 data file AND the .mat sidecar --
                  these are typically aborted-before-write scans that
                  the user doesn't want cluttering the runs list.

    Each entry::

        {
          'scan_id':   '20260529025015',
          'scan_dir':  '/.../Data/20260529/data_20260529_025015',
          'date':      '20260529',
          'time':      '025015',
          'name':      None | str,   # human-readable scan-class name
          'n_shots':   None | int,
          'n_params':  None | int,   # number of scan points (sweep size)
          'swept':     None | str,   # 'axis0 (23 pts)' or 'axis0,axis1'
          'has_diag':  bool,
          'has_code':  bool,
          'has_grid':  bool,
          'complete':  bool,         # True iff .h5 + .mat both exist
        }
    """
    base = Path(_yb_cfg.PATH_PREFIX) / 'Data'
    if not base.is_dir():
        return []
    out: List[dict] = []
    today_int: Optional[int] = None
    if since_days is not None and since_days > 0:
        from datetime import date, timedelta
        cutoff = date.today() - timedelta(days=since_days)
        today_int = int(cutoff.strftime('%Y%m%d'))
    for day_dir in sorted(base.iterdir(), reverse=True):
        if not day_dir.is_dir():
            continue
        if not day_dir.name.isdigit():
            continue
        if today_int is not None and int(day_dir.name) < today_int:
            continue
        for scan_dir in sorted(day_dir.iterdir(), reverse=True):
            m = _SCAN_DIR_RE.match(scan_dir.name)
            if not m:
                continue
            base_name = scan_dir.name   # data_YYYYMMDD_HHMMSS
            mat_path = scan_dir / f'{base_name}.mat'
            h5_path  = scan_dir / f'{base_name}.h5'
            complete = mat_path.is_file() and h5_path.is_file()
            if not include_incomplete and not complete:
                # Skip aborted-before-write scans (empty dirs / partial
                # writes); they clutter the runs table and can't be
                # analyzed anyway.
                continue
            row = {
                'scan_id':  m.group(1) + m.group(2),
                'scan_dir': str(scan_dir),
                'date':     m.group(1),
                'time':     m.group(2),
                'name':     None,
                'n_shots':  None,
                'n_params': None,
                'swept':    None,
                'has_diag': (scan_dir / 'slm_diag.h5').is_file(),
                'has_code': (scan_dir / 'slm_code.json').is_file(),
                'has_grid': (scan_dir / 'slm_grid.json').is_file(),
                'complete': complete,
            }
            if with_meta and complete:
                _enrich_meta(scan_dir, row)
            out.append(row)
            if len(out) >= max_count:
                return out
    return out


def _enrich_meta(scan_dir: Path, row: dict) -> None:
    """Best-effort: pull scan name + swept-param info from the .mat sidecar.

    Uses ``extract_scan_dims_h5`` to resolve real swept-param paths
    (dotted, e.g. ``Pushout.Green.Freq``) by walking ScanGroup.base.vars
    inside the v7.3 HDF5 .mat. Falls back gracefully when the structure
    isn't in the expected shape.
    """
    base = scan_dir.name
    mat_path = scan_dir / f'{base}.mat'
    if not mat_path.is_file():
        return
    try:
        import numpy as np
        from yb_analysis.io.mat_reader import load_scan_config_from_mat
        from yb_analysis.analysis.run_analysis import _resolve_scan_name
        from yb_analysis.detection.scan_analysis import extract_scan_dims_h5
        scan = load_scan_config_from_mat(str(mat_path)) or {}
        row['name'] = _resolve_scan_name(scan)
        # NumPerGroup is the shot count per scan point.
        npg = scan.get('NumPerGroup')
        if npg is not None:
            try:
                row['n_shots'] = int(np.asarray(npg).ravel()[0])
            except Exception:
                pass
        # Swept-param names via the HDF5 walker that already exists.
        dims = extract_scan_dims_h5(str(mat_path))
        if dims:
            names = [d.get('name') or f'axis{i}'
                     for i, d in enumerate(dims)]
            sizes = [int(d.get('size') or 0) for d in dims]
            row['swept_paths'] = names
            row['swept_dims']  = sizes
            row['n_params']    = int(np.prod(sizes)) if sizes else None
            if len(names) == 1:
                row['swept'] = f'{names[0]} ({sizes[0]} pts)'
            else:
                pairs = [f'{n} ({s})' for n, s in zip(names, sizes)]
                row['swept'] = '  ·  '.join(pairs)
        else:
            # Fallback: Params shape gives at least point counts.
            params = scan.get('Params')
            if params is not None:
                try:
                    p = np.asarray(params)
                    row['n_params'] = int(p.shape[0])
                    row['swept'] = f'{p.shape[0]} pts (path unknown)'
                except Exception:
                    pass
    except Exception as ex:
        logger.debug('runs_list: enrich %s failed: %s', scan_dir, ex)
