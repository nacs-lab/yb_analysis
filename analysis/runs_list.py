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
          'n_shots':   None | int,   # reps per scan point (NumPerGroup)
          'n_params':  None | int,   # number of scan points (sweep size)
          'n_total_shots': None | int,  # n_params * n_shots (total shots)
          'swept':     None | str,   # 'axis0 (4600 shots)' or 'axis0  ·  axis1'
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
            mat_path  = scan_dir / f'{base_name}.mat'
            json_path = scan_dir / f'{base_name}.json'
            h5_path   = scan_dir / f'{base_name}.h5'
            # A scan's config sidecar is a MATLAB ``.mat`` (matlab backend) OR a
            # ``.json`` (pyctrl backend) — either counts as a written config.
            # Without this, pyctrl scans (which never emit a ``.mat``) are seen
            # as incomplete and silently dropped from the analysis pane.
            has_config = mat_path.is_file() or json_path.is_file()
            complete = has_config and h5_path.is_file()
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


def _actual_shots(h5_path: Path) -> Optional[int]:
    """Number of recorded sequences (shots) = seq_ids dataset length.

    Opens the scan's HDF5 read-only and reads only the dataset SHAPE
    (no data). Falls back to logicals/logicals_img1 row count when seq_ids
    is absent. Returns None on any failure (file missing, not yet written).
    """
    if not h5_path.is_file():
        return None
    try:
        import h5py
        with h5py.File(str(h5_path), 'r') as f:
            for k in ('seq_ids', 'logicals_img1', 'logicals'):
                if k in f:
                    return int(f[k].shape[0])
    except Exception:
        return None
    return None


def _shots_tag(actual: Optional[int], total: Optional[int]) -> Optional[str]:
    """Format the shot-count tag: 'N shots', or 'A/T shots' when the actual
    count differs from the scheduled total (e.g. an aborted scan)."""
    if actual is not None and total is not None:
        if actual == total:
            return f'{total} shots'
        return f'{actual}/{total} shots'
    if actual is not None:
        return f'{actual} shots'
    if total is not None:
        return f'{total} shots'
    return None


def _enrich_meta(scan_dir: Path, row: dict) -> None:
    """Best-effort: pull scan name + swept-param info from the config sidecar.

    For a MATLAB ``.mat`` sidecar, ``extract_scan_dims_h5`` resolves real
    swept-param paths (dotted, e.g. ``Pushout.Green.Freq``) by walking
    ScanGroup.base.vars inside the v7.3 HDF5. For a pyctrl ``.json`` sidecar
    the ScanGroup is already a plain nested dict, so the dict-based
    ``extract_scan_dims`` recovers the same paths. Falls back gracefully when
    the structure isn't in the expected shape.
    """
    base = scan_dir.name
    mat_path  = scan_dir / f'{base}.mat'
    json_path = scan_dir / f'{base}.json'
    has_mat = mat_path.is_file()
    if not has_mat and not json_path.is_file():
        return
    # Actual recorded shots = number of saved sequences (seq_ids dataset
    # length in the scan's HDF5). Cheap: reads dataset SHAPE only, never the
    # data. Distinct from the scheduled nParams*NumPerGroup (a scan can be
    # aborted early). Best-effort — leaves n_actual_shots None on any error.
    row['n_actual_shots'] = _actual_shots(scan_dir / f'{base}.h5')
    try:
        import numpy as np
        from yb_analysis.io.mat_reader import load_scan_config
        from yb_analysis.analysis.run_analysis import _resolve_scan_name
        from yb_analysis.detection.scan_analysis import (
            extract_scan_dims, extract_scan_dims_h5)
        # load_scan_config prefers a .json sibling (pyctrl) and falls back to
        # the .mat reader (matlab), so this handles both backends.
        scan = load_scan_config(str(mat_path)) or {}
        row['name'] = _resolve_scan_name(scan)
        # Whether this scan carries a pyctrl per-run code snapshot (the
        # ``code_snapshot`` block in the .json sidecar). Distinct from ``has_code``
        # (the SLM-server ``slm_code.json``). Drives the Sequence-tab picker's
        # Reconstructable-vs-Unrecoverable split: a snapshot means a missing .seq
        # can be regenerated offline from the captured code (+ runtime globals).
        row['has_snapshot'] = bool(scan.get('code_snapshot'))
        # Whether the sidecar carries the self-contained reconstruction ``descriptor``
        # (scangroup_to_descriptor output). Reconstruction ALSO needs this -- it rebuilds
        # the ScanGroup + resolves the seq function from it. Scans that predate the
        # descriptor-storage change have a snapshot but NO descriptor, so they are NOT
        # actually reconstructable (the picker must require both).
        row['has_descriptor'] = bool(scan.get('descriptor'))
        # NumPerGroup is the shot count per scan point.
        npg = scan.get('NumPerGroup')
        if npg is not None:
            try:
                row['n_shots'] = int(np.asarray(npg).ravel()[0])
            except Exception:
                pass
        # Swept-param names: walk the v7.3 HDF5 refs for a .mat sidecar, or
        # the plain nested dict for a pyctrl .json sidecar.
        dims = (extract_scan_dims_h5(str(mat_path)) if has_mat
                else extract_scan_dims(scan))
        npg = row.get('n_shots')   # reps per scan point (NumPerGroup)
        if dims:
            names = [d.get('name') or f'axis{i}'
                     for i, d in enumerate(dims)]
            sizes = [int(d.get('size') or 0) for d in dims]
            row['swept_paths'] = names
            row['swept_dims']  = sizes
            n_params = int(np.prod(sizes)) if sizes else None
            row['n_params'] = n_params
            # Description shows the ACTUAL shot count when known (a scan can be
            # aborted before all scheduled shots run), falling back to the
            # scheduled TOTAL (points x reps-per-point). n_dims labels the
            # sweep dimensionality (1D / 2D / ...).
            total = (n_params * npg) if (n_params and npg) else None
            row['n_total_shots'] = total
            n_dims = len([s for s in sizes if s and s > 1])
            row['n_dims'] = n_dims
            actual = row.get('n_actual_shots')
            axis = names[0] if len(names) == 1 else '  ·  '.join(names)
            dim_tag = f'{n_dims}D' if n_dims else '0D'
            shots_tag = _shots_tag(actual, total)
            if shots_tag:
                row['swept'] = f'{axis} · {dim_tag} · {shots_tag}'
            elif len(names) == 1:
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
                    n_params = int(p.shape[0])
                    row['n_params'] = n_params
                    total = (n_params * npg) if npg else None
                    row['n_total_shots'] = total
                    shots_tag = _shots_tag(row.get('n_actual_shots'), total)
                    row['swept'] = (f'{shots_tag} (path unknown)'
                                    if shots_tag is not None
                                    else f'{n_params} pts (path unknown)')
                except Exception:
                    pass
    except Exception as ex:
        logger.debug('runs_list: enrich %s failed: %s', scan_dir, ex)
