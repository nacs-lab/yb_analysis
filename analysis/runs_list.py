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
import threading
from pathlib import Path
from typing import List, Optional

from yb_analysis import config as _yb_cfg

logger = logging.getLogger(__name__)

_SCAN_DIR_RE = re.compile(r'^data_(\d{8})_(\d{6})$')

# ---- Enrichment cache (incremental run-list) ----
# ``_enrich_meta`` is the expensive part of a listing: per scan it opens the
# HDF5 for the seq_ids shape, reads the .mat/.json config, and walks v7.3 HDF5
# refs. For a browse-list that's re-fetched on a poll, re-enriching every scan
# every call costs >1 s. A completed scan's metadata never changes, so we cache
# the enriched fields keyed by scan_id and only re-enrich a scan that is NEW or
# whose .h5 mtime changed (the live-growing current scan). Steady-state cost
# then drops to the cheap directory/stat pass + dict lookups (~tens of ms).
_ENRICH_CACHE: dict = {}          # scan_id -> {'mtime': float|None, 'enriched': {field: val}}
_ENRICH_CACHE_LOCK = threading.Lock()
_ENRICH_CACHE_MAX = 4000          # hard cap; clear wholesale past this (cheap to rebuild)
# Fields produced by ``_enrich_meta`` that are cached/restored. Cheap stat-based
# flags (has_diag/has_code/has_grid/complete) are recomputed fresh each pass.
_ENRICH_FIELDS = (
    'name', 'description', 'n_shots', 'n_params', 'swept', 'n_actual_shots',
    'has_snapshot', 'has_descriptor', 'swept_paths', 'swept_dims',
    'n_total_shots', 'n_dims',
)


def clear_enrich_cache() -> None:
    """Drop the whole enrichment cache (a Full-rescan re-enriches every scan)."""
    with _ENRICH_CACHE_LOCK:
        _ENRICH_CACHE.clear()


def list_dates() -> List[str]:
    """Return every available data day (``YYYYMMDD``), newest-first.

    Cheap by design: a single directory listing with NO per-scan stat or
    metadata enrichment, so the picker can offer a full date jump across the
    whole (multi-year, thousands-of-scans) archive without paying the cost of
    enumerating every run. Pair with ``list_runs(date_str=...)`` to load just
    the chosen day on demand.
    """
    base = Path(_yb_cfg.PATH_PREFIX) / 'Data'
    if not base.is_dir():
        return []
    days = [d.name for d in base.iterdir()
            if d.is_dir() and len(d.name) == 8 and d.name.isdigit()]
    days.sort(reverse=True)
    return days


def list_runs(*, since_days: Optional[int] = None,
              max_count: int = 500,
              with_meta: bool = True,
              include_incomplete: bool = False,
              use_cache: bool = False,
              force: bool = False,
              date_str: Optional[str] = None) -> List[dict]:
    """Return a list of scans found under the lab's data directory.

    Args:
      since_days: only include scans whose YYYYMMDD date is within this
                  many days of today. ``None`` = no cutoff.
      max_count:  cap on returned rows (newest first).
      date_str:   if set (an 8-digit ``YYYYMMDD``), restrict the listing to that
                  ONE day's folder. Lets the picker jump to any historical day
                  without walking/enriching the whole multi-year archive to
                  reach an old run; overrides ``since_days``. ``None`` (default)
                  = the normal newest-first walk across all days, capped at
                  ``max_count``.
      with_meta:  if True, try to load the .mat sidecar for each scan
                  to populate ``name``, ``n_shots``, etc. Adds a few ms
                  per scan; disable for fast directory listings.
      include_incomplete: if False (default), skip scan dirs that don't
                  have BOTH the .h5 data file AND the .mat sidecar --
                  these are typically aborted-before-write scans that
                  the user doesn't want cluttering the runs list.
      use_cache:  if True (and ``with_meta``), serve enriched metadata from
                  the module cache, re-enriching only NEW scans or ones whose
                  .h5 mtime changed (the live-growing current scan). Makes a
                  repeated poll cheap. Default False = always re-enrich (old
                  behavior; keeps existing callers/tests unchanged).
      force:      if True, clear the enrichment cache first (Full-rescan).

    Each entry::

        {
          'scan_id':   '20260529025015',
          'scan_dir':  '/.../Data/20260529/data_20260529_025015',
          'date':      '20260529',
          'time':      '025015',
          'name':      None | str,   # human-readable scan-class name
          'n_shots':   None | int,   # reps per scan point = StackNum (total // points)
          'n_params':  None | int,   # number of scan points (sweep size)
          'n_total_shots': None | int,  # planned total shots ("supposed to do")
          'swept':     None | str,   # 'axis0 (4600 shots)' or 'axis0  ·  axis1'
          'has_diag':  bool,
          'has_code':  bool,
          'has_grid':  bool,
          'complete':  bool,         # True iff .h5 + .mat both exist
        }
    """
    if force:
        clear_enrich_cache()
    base = Path(_yb_cfg.PATH_PREFIX) / 'Data'
    if not base.is_dir():
        return []
    out: List[dict] = []
    # A single-day jump (the picker's date dropdown) restricts the walk to one
    # folder, so we never stat/enrich the whole archive just to reach an old
    # run -- ``since_days`` is moot in that case.
    if date_str:
        one = base / date_str
        day_dirs = [one] if one.is_dir() else []
    else:
        day_dirs = [d for d in sorted(base.iterdir(), reverse=True)
                    if d.is_dir() and d.name.isdigit()]
        if since_days is not None and since_days > 0:
            from datetime import date, timedelta
            cutoff = int((date.today() - timedelta(days=since_days))
                         .strftime('%Y%m%d'))
            day_dirs = [d for d in day_dirs if int(d.name) >= cutoff]
    for day_dir in day_dirs:
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
                'description': None,
                'n_shots':  None,
                'n_params': None,
                'swept':    None,
                'has_diag': (scan_dir / 'slm_diag.h5').is_file(),
                'has_code': (scan_dir / 'slm_code.json').is_file(),
                'has_grid': (scan_dir / 'slm_grid.json').is_file(),
                'complete': complete,
            }
            if with_meta and complete:
                if use_cache:
                    _enrich_meta_cached(scan_dir, row)
                else:
                    _enrich_meta(scan_dir, row)
            out.append(row)
            if len(out) >= max_count:
                return out
    return out


def _enrich_meta_cached(scan_dir: Path, row: dict) -> None:
    """Cache-backed ``_enrich_meta``: reuse the cached enriched fields when the
    scan's .h5 mtime is unchanged, otherwise enrich fresh and (re)cache.

    The cheap stat-based flags (has_diag/has_code/has_grid/complete) on ``row``
    are left as the caller computed them; only the expensive ``_ENRICH_FIELDS``
    are cached. The live-growing current scan re-enriches because its .h5 mtime
    advances as shots are appended.
    """
    base = scan_dir.name
    h5_path = scan_dir / f'{base}.h5'
    try:
        mtime = h5_path.stat().st_mtime
    except OSError:
        mtime = None
    sid = row['scan_id']
    with _ENRICH_CACHE_LOCK:
        ent = _ENRICH_CACHE.get(sid)
        hit = ent is not None and ent.get('mtime') == mtime
        if hit:
            row.update(ent['enriched'])
    if hit:
        return
    # Miss or .h5 changed -> enrich fresh (outside the lock; this is the slow part).
    _enrich_meta(scan_dir, row)
    enriched = {k: row[k] for k in _ENRICH_FIELDS if k in row}
    with _ENRICH_CACHE_LOCK:
        if len(_ENRICH_CACHE) >= _ENRICH_CACHE_MAX:
            _ENRICH_CACHE.clear()   # wholesale reset; cheap to rebuild incrementally
        _ENRICH_CACHE[sid] = {'mtime': mtime, 'enriched': enriched}


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
        from yb_analysis.analysis.run_analysis import (
            _resolve_scan_name, _scan_description, _planned_shots)
        from yb_analysis.detection.scan_analysis import (
            extract_scan_dims, extract_scan_dims_h5)
        # load_scan_config prefers a .json sibling (pyctrl) and falls back to
        # the .mat reader (matlab), so this handles both backends.
        scan = load_scan_config(str(mat_path)) or {}
        row['name'] = _resolve_scan_name(scan)
        # Free-text run purpose/context (pyctrl stamps it from the descriptor's
        # `description`; blank for pre-feature scans). Surfaced in the runs picker
        # so runs are searchable by purpose, not just name.
        row['description'] = _scan_description(scan)
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
            # "supposed to do" = the realized run-order length (n_shots_planned /
            # len(Params), honoring an explicit rep), NOT nParams * NumPerGroup
            # (NumPerGroup is a TOTAL-shots target, not per-point reps -- multiplying
            # over-counts by xnParams). Fall back to the legacy estimate for old
            # sidecars lacking both fields. The tag shows ACTUAL/total when they differ
            # (e.g. an aborted scan). n_dims labels the sweep dimensionality (1D/2D/...).
            total = _planned_shots(
                scan, fallback=((n_params * npg) if (n_params and npg) else None))
            row['n_total_shots'] = total
            # reps per scan point (StackNum) = total // points, kept consistent with
            # the total above so the picker's "points x reps/pt = total shots" holds.
            if total and n_params:
                row['n_shots'] = int(total // n_params)
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
                    # No swept-dim info here, so Params length IS the planned total
                    # (the realized run order); _planned_shots returns it directly.
                    total = _planned_shots(
                        scan, fallback=((n_params * npg) if npg else None))
                    row['n_total_shots'] = total
                    shots_tag = _shots_tag(row.get('n_actual_shots'), total)
                    row['swept'] = (f'{shots_tag} (path unknown)'
                                    if shots_tag is not None
                                    else f'{n_params} pts (path unknown)')
                except Exception:
                    pass
    except Exception as ex:
        logger.debug('runs_list: enrich %s failed: %s', scan_dir, ex)
