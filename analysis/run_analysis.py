"""Per-scan run analysis for the unified dashboard (Phase 4).

Joins:
- the lab-side HDF5 (``data_<scan_id>.h5`` + sibling ``data_<scan_id>.mat``)
  for per-shot logicals + scan config
- the SLM-side ledger sidecar (``slm_diag.h5``) for per-shot rearrange
  diagnostics (timing, n_loaded, abort flags, ...)
- the code-snapshot sidecar (``slm_code.json``) for the manifest pointer
  the protocol-source viewer uses
- the per-run grid sidecar (``slm_grid.json``, Phase 4 addition) — the
  exact derived+reordered lattice the SLM server commanded, when the
  upstream `rearrange_grid_sidecar` path produced one

…and returns a JSON-safe dict shape-compatible with the SLM server's
`/runs/<id>/analysis` response, so the dashboard's Analysis tab can
render either source with the same panels.

The lab side cannot reproduce every SLM-side server analysis field
(some require server-only context: e.g. `round1` block requires the
two-round phase to be active during the scan). Those fields are set
to ``None`` when unavailable; the dashboard hides their panel.

Public API::

    from yb_analysis.analysis.run_analysis import analyze_scan
    result = analyze_scan('20260529025015')
    # or
    result = analyze_scan(scan_dir='/path/to/data_20260529_025015')
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from yb_analysis import config as _yb_cfg
from yb_analysis.analysis.load_data import load_scan_from_path
from yb_analysis.analysis.probabilities import (
    prob11, prob11_site_resolved,
    prob10, prob10_site_resolved,
    loading_rate, loading_rate_site_resolved,
)
from yb_analysis.analysis.unpack import unpack_scan_logicals

logger = logging.getLogger(__name__)


# Filenames the slm_sync pipeline writes next to data_*.h5.
DIAG_H5 = 'slm_diag.h5'
CODE_JSON = 'slm_code.json'
GRID_JSON = 'slm_grid.json'   # Phase 4: per-run grid sidecar


class RunAnalysisError(ValueError):
    """Raised when a scan can't be analyzed (path missing, etc.)."""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def analyze_scan(scan_id: Optional[str] = None,
                 *,
                 scan_dir: Optional[str] = None,
                 include_per_site: bool = True,
                 include_diag_aggregate: bool = True,
                 include_per_iteration: bool = True,
                 filters: Optional[dict] = None) -> dict:
    """Run lab-side analysis on a completed scan.

    Either ``scan_id`` (14-digit YYYYMMDDHHMMSS) or ``scan_dir`` must
    be given. When ``scan_id`` is supplied, the directory is located
    under ``yb_analysis.config.DATA_DIR / YYYYMMDD / data_<scan_id>``.

    Returns a JSON-safe dict — see ``analyze_scan_dir`` below for shape.
    """
    if scan_dir is None:
        if scan_id is None:
            raise RunAnalysisError("must pass scan_id or scan_dir")
        scan_dir = _resolve_scan_dir_from_id(scan_id)
        if scan_dir is None:
            raise RunAnalysisError(
                f"could not find scan directory for scan_id={scan_id!r}")
    return analyze_scan_dir(
        scan_dir,
        include_per_site=include_per_site,
        include_diag_aggregate=include_diag_aggregate,
        include_per_iteration=include_per_iteration,
        filters=filters)


def analyze_scan_dir(scan_dir,
                     *,
                     include_per_site: bool = True,
                     include_diag_aggregate: bool = True,
                     include_per_iteration: bool = True,
                     filters: Optional[dict] = None) -> dict:
    """Analyze a scan from its directory path. Same return shape as
    :func:`analyze_scan` — kept separate so callers that already
    have a path don't have to round-trip through scan_id resolution.

    Result dict (JSON-safe — every value goes through :func:`to_jsonable`):

    ::

        {
          'scan_id':       str,         # parsed from path, may be ''
          'scan_dir':      str,
          'scan_name':     str | None,  # from Scan struct
          'n_shots':       int,         # = nParams * maxReps
          'n_params':      int,
          'sweep': {
            'cols':     list[str],
            'values':   list[list[float]],  # per-axis values
            'dims':     list[int],          # axis sizes
          },
          'summary': {
            'survival_mean':       list[float] (nParams),  # P11
            'survival_sem':        list[float] (nParams),
            'loading_rate':        list[float] (nParams),
            'loading_rate_sem':    list[float] (nParams),
            'loss_mean':           list[float] (nParams),  # P10
            'loss_sem':            list[float] (nParams),
          },
          'per_site': {                   # omitted if include_per_site=False
            'survival_mean':   list[list[float]] (nSites, nParams),
            'loading_rate':    list[list[float]] (nSites, nParams),
          },
          'diag_aggregate': {             # None if no slm_diag.h5
            'n_rows':            int,
            'mean_total_ms':     float,
            'p99_total_ms':      float,
            'mean_n_loaded':     float,
            'mean_n_dropped':    float,
            'aborted_count':     int,
            'two_round_phases':  dict[str, int],  # phase -> count
          },
          'code': {                       # None if no slm_code.json
            'present':       bool,
            'manifest_path': str | None,
            'safe_run_id':   str | None,
            'n_files':       int,
          },
          'grid': {                       # None if no slm_grid.json
            'present':       bool,
            'schema':        str | None,
            'n_sites':       int,
            'grid_rotation': float | None,
          },
          # Forward-compat fields surfaced when present in slm_diag.h5 but
          # otherwise computed lab-side via probabilities.py:
          'round1':                          dict | None,
          'survival_vs_distance':            dict | None,
          'survival_vs_distance_per_step':   dict | None,
          'per_shot_extra':                  dict | None,
        }
    """
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        raise RunAnalysisError(f"scan_dir not a directory: {scan_dir}")

    out: dict = {
        'scan_id':  _scan_id_from_dir(scan_dir),
        'scan_dir': str(scan_dir),
    }

    # ---- Lab HDF5 + Scan struct + logicals -----------------------------
    try:
        bundle = load_scan_from_path(str(scan_dir))
    except (FileNotFoundError, OSError) as ex:
        raise RunAnalysisError(f"failed to load scan data: {ex}") from ex

    scan = bundle.get('Scan') or {}
    out['scan_name']     = _resolve_scan_name(scan)
    out['scan_filename'] = _str_or_none(scan.get('scanfilename'))

    # ---- Logicals: unpack scrambled order into (nSites, nParams, reps)
    logicals_kw = {}
    if bundle.get('two_array'):
        logicals_kw['logicals_img1'] = bundle.get('logicals_img1')
        logicals_kw['logicals_img2'] = bundle.get('logicals_img2')
    else:
        logicals_kw['logicals'] = bundle.get('logicals')

    seq_ids = bundle.get('seq_ids')
    out['unpack_error'] = None
    try:
        scan_params, logic1, logic2, reps_per_param = unpack_scan_logicals(
            scan, seq_ids=seq_ids,
            mat_path=bundle.get('mat_path'),
            **logicals_kw)
    except (ValueError, KeyError, AttributeError) as ex:
        # Surface the error so the dashboard shows what went wrong
        # (rather than rendering empty charts with no explanation).
        msg = f"{type(ex).__name__}: {ex}"
        logger.warning("analyze_scan_dir: unpack_scan_logicals failed: %s", msg)
        out['unpack_error'] = msg
        scan_params = np.empty(0)
        logic1 = np.zeros((0, 0, 0), dtype=bool)
        logic2 = None
        reps_per_param = np.zeros(0, dtype=int)
    except Exception as ex:
        # Catch-all: surface even unexpected failures rather than
        # blanket-swallowing.
        msg = f"{type(ex).__name__}: {ex}"
        logger.exception("analyze_scan_dir: unpack_scan_logicals crashed")
        out['unpack_error'] = msg
        scan_params = np.empty(0)
        logic1 = np.zeros((0, 0, 0), dtype=bool)
        logic2 = None
        reps_per_param = np.zeros(0, dtype=int)

    n_sites, n_params, max_reps = (
        logic1.shape if logic1.ndim == 3 else (0, 0, 0))
    out['n_params'] = int(n_params)
    out['n_shots']  = int(n_params * max_reps)
    out['n_sites']  = int(n_sites)
    # Surface what shapes we got so the dashboard can pinpoint
    # data-loading issues (logicals shape, two-array layout, etc.).
    out['data_shapes'] = {
        'two_array':         bool(bundle.get('two_array', False)),
        'logicals':          _shape_or_none(bundle.get('logicals')),
        'logicals_img1':     _shape_or_none(bundle.get('logicals_img1')),
        'logicals_img2':     _shape_or_none(bundle.get('logicals_img2')),
        'intensities':       _shape_or_none(bundle.get('intensities')),
        'seq_ids':           _shape_or_none(seq_ids),
        'logic1_unpacked':   list(logic1.shape) if hasattr(logic1, 'shape') else None,
    }

    # ---- Sweep description (axes + values + dim sizes) ----------------
    # Pass the .mat path so _build_sweep can use extract_scan_dims_h5,
    # which actually dereferences ScanVar HDF5 refs and recovers the
    # dotted swept-param path (e.g. "Pushout.Green.Freq"). Without the
    # path, sweep.cols falls back to "axis0"/"axis1" placeholders.
    out['sweep'] = _build_sweep(scan, scan_params,
                                mat_path=bundle.get('mat_path'))
    # Always include the FULL (unfiltered) sweep so the dashboard's
    # filter chips have every value to render -- otherwise selecting
    # one chip would cause the others to vanish from the UI.
    out['sweep_all'] = dict(out['sweep'])

    # ---- Filter mask over scan-param axis (when provided) -------------
    # ``filters`` is {axis_name: [allowed values]} keyed by the swept
    # parameter's dotted path. Empty / missing axis = no filter on that
    # axis. Applied BEFORE summary + per_site + per_iteration so every
    # downstream card honors the same filter.
    param_mask = _build_param_filter_mask(scan_params, out['sweep'], filters)
    if param_mask is not None and logic1.size:
        # Slice logic1 / logic2 along the n_params axis (axis=1).
        try:
            logic1 = logic1[:, param_mask, :]
            if logic2 is not None:
                logic2 = logic2[:, param_mask, :]
            scan_params = scan_params[param_mask] if scan_params.ndim == 1 \
                          else scan_params[param_mask, :]
            reps_per_param = reps_per_param[param_mask]
            n_params = int(param_mask.sum())
            out['n_params']  = n_params
            out['n_shots']   = int(n_params * max_reps)
            out['sweep']     = _build_sweep(scan, scan_params,
                                            mat_path=bundle.get('mat_path'))
        except (IndexError, ValueError) as ex:
            logger.warning('filter slicing failed: %s', ex)
    out['filter_active'] = bool(param_mask is not None)

    # ---- Summary: survival / loading / loss per param ------------------
    out['summary'] = _summary_stats(logic1, logic2)

    if include_per_site and logic1.size:
        try:
            # Per-site MAP values: one scalar per site (averaged over
            # all swept param points + reps in the filtered subset).
            # `loading_rate_site_resolved` / `prob11_site_resolved` give
            # site-by-param breakdowns; the map panel wants site-only,
            # so we mean over the param axis after they return.
            lr_per_param, _ = loading_rate_site_resolved(
                logic1, reps_per_param=reps_per_param)
            with np.errstate(invalid='ignore'):
                lr_mean = np.nanmean(np.asarray(lr_per_param, dtype=float),
                                     axis=1)
            per_site = {
                'loading_rate':   lr_mean.tolist(),
                'survival_mean':  None,
                'fp_rate':        None,
            }
            if logic2 is not None and logic2.size:
                sr_per_param, _ = prob11_site_resolved(logic1, logic2)
                with np.errstate(invalid='ignore'):
                    sr_mean = np.nanmean(np.asarray(sr_per_param, dtype=float),
                                         axis=1)
                per_site['survival_mean'] = sr_mean.tolist()
                with np.errstate(invalid='ignore', divide='ignore'):
                    empty = (~logic1).astype(np.float64)
                    detected_in_2 = (logic2 & ~logic1).astype(np.float64)
                    n_empty = empty.sum(axis=(1, 2))
                    n_fp    = detected_in_2.sum(axis=(1, 2))
                    fp = np.where(n_empty > 0, n_fp / n_empty, np.nan)
                per_site['fp_rate'] = fp.tolist()
            per_site['x'], per_site['y'] = _site_grid_xy(scan)
            out['per_site'] = per_site
        except (ValueError, IndexError) as ex:
            logger.warning("analyze_scan_dir: per_site failed: %s", ex)
            out['per_site'] = None
    else:
        out['per_site'] = None

    # ---- Per-iteration arrays (TIME-ORDER, one per shot) -------------
    # IMPORTANT: walks raw interleaved bundle.logicals using seq_ids so
    # the X axis is true shot order, NOT the (param × rep) unpacked
    # order. Without this, the swept-param trace looks like a slow
    # staircase (consecutive iterations share a param) instead of the
    # scrambled order the scan actually executed in.
    if include_per_iteration:
        try:
            out['per_iteration'] = _per_iteration_time_order(
                scan, bundle, seq_ids, param_mask, filters)
        except Exception as ex:
            logger.warning('per_iteration_time_order failed: %s', ex)
            out['per_iteration'] = None
    else:
        out['per_iteration'] = None

    # ---- SLM diag aggregate (from synced sidecar) ----------------------
    diag_path = scan_dir / DIAG_H5
    if include_diag_aggregate and diag_path.is_file():
        out['diag_aggregate'] = _diag_aggregate(diag_path)
    else:
        out['diag_aggregate'] = None

    # ---- Code snapshot sidecar pointer ---------------------------------
    out['code'] = _code_info(scan_dir / CODE_JSON)

    # ---- Per-run grid sidecar (Phase 4 addition) -----------------------
    out['grid'] = _grid_info(scan_dir / GRID_JSON)

    # ---- Upstream-compat fields ---------------------------------------
    # These are computed by the SLM server and ride through `slm_diag.h5`
    # via the diag_json vlen column. Lab-side replay would need the
    # same algorithms; for now we leave them None and the dashboard
    # hides the matching panels. Phase 5 verification can decide
    # whether to mirror them here vs. lazy-fetch from SLM.
    out['round1'] = None
    out['survival_vs_distance'] = None
    out['survival_vs_distance_per_step'] = None
    out['per_shot_extra'] = None

    return to_jsonable(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCAN_DIR_RE = re.compile(r'data_(\d{8})_(\d{6})$')
_SCAN_ID_RE = re.compile(r'^\d{14}$')


def _resolve_scan_dir_from_id(scan_id: str) -> Optional[str]:
    """``DATA_DIR / YYYYMMDD / data_YYYYMMDD_HHMMSS``."""
    if not _SCAN_ID_RE.match(str(scan_id)):
        raise RunAnalysisError(
            f"scan_id must be 14 digits, got {scan_id!r}")
    s = str(scan_id)
    day, hms = s[:8], s[8:]
    candidate = Path(_yb_cfg.DATA_DIR) / day / f'data_{day}_{hms}'
    if candidate.is_dir():
        return str(candidate)
    # Some scans live under a different DATA_DIR override — also check
    # the env var explicitly.
    env_dir = os.environ.get('YB_DATA_DIR')
    if env_dir:
        candidate = Path(env_dir) / day / f'data_{day}_{hms}'
        if candidate.is_dir():
            return str(candidate)
    return None


def _scan_id_from_dir(scan_dir: Path) -> str:
    m = _SCAN_DIR_RE.search(str(scan_dir))
    if not m:
        return ''
    return m.group(1) + m.group(2)


def _shape_or_none(v):
    """Best-effort: return the shape of a numpy-like value, or None."""
    if v is None:
        return None
    try:
        return list(v.shape)
    except AttributeError:
        try:
            return [len(v)]
        except TypeError:
            return None


def _str_or_none(v) -> Optional[str]:
    """Decode a value that *might* be a MATLAB v7.3 uint16 char array
    into a Python string. Returns None for dicts (broken loader case
    where a nested struct came through where a string was expected),
    uint64 (HDF5 refs), and empty / non-printable arrays.

    The .mat sidecar stores Scan.scanname / Scan.scanfilename as
    column-vector uint16 arrays of MATLAB chars. Old `str(v)` returned
    `"array([[83],[112],...])"` which is useless. Same for `ScanName`
    when it's actually the whole nested struct (loader quirk).
    """
    if v is None:
        return None
    if isinstance(v, dict):
        # ScanName is sometimes the WHOLE Scan struct because the loader
        # didn't deref properly. Don't stringify the dict; caller will
        # try a fallback field.
        return None
    if hasattr(v, 'dtype') and hasattr(v, 'ravel'):
        try:
            arr = np.asarray(v).ravel()
            if arr.size == 0:
                return None
            # uint16 char array: decode each codepoint
            if arr.dtype.kind in ('u', 'i') and arr.dtype.itemsize <= 4:
                # uint64 with the value 0 is an HDF5 ref placeholder; skip.
                if arr.dtype.itemsize == 8:
                    return None
                try:
                    s = ''.join(chr(int(x)) for x in arr.tolist())
                    s = s.strip()
                    return s or None
                except (ValueError, OverflowError):
                    return None
            if arr.dtype.kind in ('U', 'S'):
                s = str(arr[0]) if arr.size else ''
                return s.strip() or None
        except Exception:
            return None
    try:
        s = str(v).strip()
    except Exception:
        return None
    return s or None


def _resolve_scan_name(scan: dict) -> Optional[str]:
    """Try multiple known fields to recover a clean scan name string."""
    # scanname / scanfilename are stored as uint16 char arrays at the
    # Scan struct top level. ScanName is sometimes a struct (bad
    # loader path) so try the lowercase fields first.
    for k in ('scanname', 'scanfilename', 'ScanName'):
        v = scan.get(k)
        # ScanName can itself be a dict containing scanname/scanfilename
        # uint16 arrays -- recurse one level if so.
        if isinstance(v, dict):
            for kk in ('scanname', 'scanfilename'):
                s = _str_or_none(v.get(kk))
                if s:
                    return s
            continue
        s = _str_or_none(v)
        if s:
            return s
    return None


def _build_sweep(scan: dict, scan_params: np.ndarray,
                 *, mat_path: Optional[str] = None) -> dict:
    """Best-effort description of the swept axes for the dashboard.

    Preferred path: when ``mat_path`` is supplied, walk the v7.3 HDF5
    via ``extract_scan_dims_h5`` to recover proper dotted swept-param
    paths (e.g. ``Pushout.Green.Freq``). Falls back to reading the
    cached scan dict's ``ScanVar`` (which often only contains uint64
    HDF5 refs we can't dereference here) and then to ``axisN``
    placeholders so the dashboard still has something to label.
    """
    cols: list = []
    values: list = []
    dims: list = []

    # Preferred: walk the .mat HDF5 directly.
    if mat_path:
        try:
            from yb_analysis.detection.scan_analysis import extract_scan_dims_h5
            mat_dims = extract_scan_dims_h5(mat_path) or []
            for d in mat_dims:
                name = d.get('name') or f'axis{len(cols)}'
                cols.append(name)
        except Exception as ex:
            logger.debug('extract_scan_dims_h5 failed: %s', ex)

    if not cols:
        sv = scan.get('ScanVar')
        if sv is not None:
            try:
                sv_arr = np.atleast_1d(np.asarray(sv)).ravel()
                if sv_arr.dtype.kind in ('U', 'S'):
                    cols = [str(x).strip() for x in sv_arr.tolist()
                            if str(x).strip()]
                elif sv_arr.dtype.kind in ('u', 'i') and sv_arr.dtype.itemsize <= 4:
                    try:
                        s = ''.join(chr(int(x)) for x in sv_arr.tolist()).strip()
                        if s:
                            cols = [s]
                    except Exception:
                        pass
            except Exception:
                cols = []

    if scan_params.size:
        if scan_params.ndim == 1:
            values = [np.unique(scan_params).tolist()]
            dims = [len(values[0])]
        else:
            for axis in range(scan_params.shape[1]):
                col_vals = np.unique(scan_params[:, axis]).tolist()
                values.append(col_vals)
                dims.append(len(col_vals))

    while len(cols) < len(dims):
        cols.append(f'axis{len(cols)}')
    return {'cols': cols, 'values': values, 'dims': dims}


def _summary_stats(logic1: np.ndarray,
                   logic2: Optional[np.ndarray]) -> dict:
    if logic1.size == 0 or logic2 is None or logic2.size == 0:
        # Loading-only or empty scan: still report what we can.
        try:
            lr_mean, lr_sem = (loading_rate(logic1)
                               if logic1.size else (np.array([]),
                                                    np.array([])))
        except Exception:
            lr_mean, lr_sem = np.array([]), np.array([])
        n = lr_mean.size
        empty = [float('nan')] * n
        return {
            'survival_mean':     empty,
            'survival_sem':      empty,
            'loading_rate':      lr_mean.tolist(),
            'loading_rate_sem':  lr_sem.tolist(),
            'loss_mean':         empty,
            'loss_sem':          empty,
        }
    sr_mean, sr_sem = prob11(logic1, logic2)
    ls_mean, ls_sem = prob10(logic1, logic2)
    lr_mean, lr_sem = loading_rate(logic1)
    return {
        'survival_mean':     sr_mean.tolist(),
        'survival_sem':      sr_sem.tolist(),
        'loading_rate':      lr_mean.tolist(),
        'loading_rate_sem':  lr_sem.tolist(),
        'loss_mean':         ls_mean.tolist(),
        'loss_sem':          ls_sem.tolist(),
    }


def _diag_aggregate(diag_path: Path) -> Optional[dict]:
    """Aggregate stats from the synced slm_diag.h5 (Phase 2 schema).
    Returns ``None`` on read failure rather than raising — the rest of
    the analysis is still useful without diag rollups."""
    try:
        import h5py  # local import: this whole branch is optional
    except ImportError:
        logger.info("_diag_aggregate: h5py not available")
        return None
    try:
        with h5py.File(diag_path, 'r') as f:
            g = f['/diag'] if '/diag' in f else f
            n_rows = int(g['seq_id'].shape[0]) if 'seq_id' in g else 0

            def _stat(name, fn):
                if name not in g:
                    return None
                arr = np.asarray(g[name][:], dtype=np.float64)
                if not arr.size:
                    return None
                return float(fn(arr))

            mean_total = _stat('total_ms', np.nanmean)
            p99_total  = _stat('total_ms', lambda a: np.nanpercentile(a, 99))
            mean_load  = _stat('n_loaded', np.nanmean)
            mean_drop  = _stat('n_dropped', np.nanmean)

            aborted_count = 0
            if 'aborted' in g:
                aborted_count = int(np.sum(np.asarray(g['aborted'][:],
                                                      dtype=np.uint8) != 0))

            two_round_phases: dict[str, int] = {}
            # `two_round_phase` is a vlen-string. Skip with a clear log
            # if absent (it isn't surfaced as its own column yet).
            try:
                arr = g['two_round_phase'][:] if 'two_round_phase' in g else []
                for v in arr:
                    key = v.decode() if isinstance(v, bytes) else str(v)
                    two_round_phases[key] = two_round_phases.get(key, 0) + 1
            except Exception:
                pass

            return {
                'n_rows':          n_rows,
                'mean_total_ms':   mean_total,
                'p99_total_ms':    p99_total,
                'mean_n_loaded':   mean_load,
                'mean_n_dropped':  mean_drop,
                'aborted_count':   aborted_count,
                'two_round_phases': two_round_phases,
            }
    except (OSError, KeyError) as ex:
        logger.warning("_diag_aggregate: %s: %s", diag_path, ex)
        return None


def _code_info(code_json: Path) -> dict:
    if not code_json.is_file():
        return {'present': False, 'manifest_path': None,
                'safe_run_id': None, 'n_files': 0}
    try:
        with open(code_json, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning("_code_info: failed to read %s: %s", code_json, ex)
        return {'present': False, 'manifest_path': None,
                'safe_run_id': None, 'n_files': 0}
    manifest = payload.get('manifest') or {}
    files = manifest.get('files') or []
    return {
        'present': True,
        'manifest_path': payload.get('manifest_path'),
        'safe_run_id':   payload.get('safe_run_id'),
        'n_files':       len(files),
    }


def _grid_info(grid_json: Path) -> dict:
    if not grid_json.is_file():
        return {'present': False, 'schema': None,
                'n_sites': 0, 'grid_rotation': None}
    try:
        with open(grid_json, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning("_grid_info: failed to read %s: %s", grid_json, ex)
        return {'present': False, 'schema': None,
                'n_sites': 0, 'grid_rotation': None}
    # The SLM server's rearrange_grid_sidecar layout has 'init_grid'
    # (loading lattice) and 'target_grid'; we surface the bigger of
    # the two as n_sites for the dashboard, since either lattice
    # alone undercounts.
    init = payload.get('init_grid') or []
    targ = payload.get('target_grid') or []
    n_sites = max(len(init), len(targ))
    rot = payload.get('grid_rotation')
    try:
        rot_val = float(rot) if rot is not None else None
    except (TypeError, ValueError):
        rot_val = None
    return {
        'present': True,
        'schema':         payload.get('schema'),
        'n_sites':        int(n_sites),
        'grid_rotation':  rot_val,
    }


def to_jsonable(obj):
    """Coerce numpy / NaN / nested structures to JSON-safe Python.

    Shared with the dashboard's existing `_to_jsonable` helper but kept
    here so this module is import-free of plotting code.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (bool,)):
        return bool(obj)
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, (float,)):
        # Preserve NaN/Inf as None for JSON (the standard json module
        # encodes them as `NaN`/`Infinity`, which most browsers reject).
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return float(obj)
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return f'<{len(obj)} bytes>'
    if isinstance(obj, np.ndarray):
        # Replace NaN/Inf with None in the list form.
        if obj.dtype.kind in ('f', 'c'):
            return [to_jsonable(float(x)) for x in obj.flat] \
                if obj.ndim == 1 \
                else [[to_jsonable(float(x)) for x in row]
                      for row in obj]
        return obj.tolist()
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    try:
        return str(obj)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase-5 helpers: per-site grid + filter + per-iteration
# ---------------------------------------------------------------------------

def _site_grid_xy(scan: dict):
    """Pull per-site grid coordinates from the .mat sidecar.

    Returns (x_list, y_list) flat lists ordered to match logic1's site
    axis. Empty lists when the .mat doesn't have the initGridLocations
    fields. Used by the per-site map panels (loading, survival, FP).
    """
    xs = scan.get('initGridLocationsX')
    ys = scan.get('initGridLocationsY')
    if xs is None or ys is None:
        return [], []
    try:
        x = np.asarray(xs).ravel().astype(float).tolist()
        y = np.asarray(ys).ravel().astype(float).tolist()
        return x, y
    except (ValueError, TypeError):
        return [], []


def _build_param_filter_mask(scan_params: np.ndarray,
                             sweep: dict,
                             filters: Optional[dict]):
    """Build a boolean mask over the scan-param axis from a filter dict.

    ``filters`` is ``{axis_name: [allowed values]}`` keyed by swept-param
    dotted path. Multiple axes AND together. Empty list on an axis means
    "no constraint" (identical to absent). Returns ``None`` when no
    constraint applies so the caller can skip slicing.
    """
    if not filters or scan_params.size == 0:
        return None
    cols = sweep.get('cols') or []
    if scan_params.ndim == 1:
        sp = scan_params.reshape(-1, 1)
    else:
        sp = scan_params
    mask = np.ones(sp.shape[0], dtype=bool)
    any_active = False
    for axis_idx, axis_name in enumerate(cols):
        if axis_idx >= sp.shape[1]:
            continue
        allowed = filters.get(axis_name)
        if allowed is None or not allowed:
            continue
        try:
            allowed_arr = np.asarray(allowed, dtype=float)
        except (ValueError, TypeError):
            continue
        col = sp[:, axis_idx].astype(float)
        col_match = np.zeros(sp.shape[0], dtype=bool)
        for v in allowed_arr:
            col_match |= np.isclose(col, v, rtol=1e-6, atol=1e-9)
        mask &= col_match
        any_active = True
    if not any_active:
        return None
    return mask


def _per_iteration_time_order(scan: dict,
                              bundle: dict,
                              seq_ids,
                              param_mask: Optional[np.ndarray],
                              filters: Optional[dict]) -> dict:
    """Per-shot metrics IN TIME ORDER (matches scan execution sequence).

    Reads the bundle's raw ``logicals`` (interleaved img1/img2) and uses
    ``seq_ids`` to map each shot index to its actual scan-point index in
    ``Scan.Params``. This is the only way to get the swept-param values
    in execution order -- the unpacked (param, rep) shape is rearranged
    and would render the swept-param trace as a slow staircase rather
    than the scrambled time series the experiment actually ran.

    Filtering: when ``filters`` is provided, only shots whose param
    matches the filter are included. The filter axis-name lookup uses
    the same logic as ``_build_param_filter_mask``.
    """
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0])
    if num_images < 1:
        num_images = 1
    raw = bundle.get('logicals')
    raw1 = bundle.get('logicals_img1')
    raw2 = bundle.get('logicals_img2')
    if raw1 is not None and raw2 is not None:
        # Two-array layout: img1/img2 already split.
        img1 = np.asarray(raw1)
        img2 = np.asarray(raw2)
    elif raw is not None:
        a = np.asarray(raw)
        if num_images >= 2:
            img1 = a[0::num_images]
            img2 = a[(num_images - 1)::num_images]
        else:
            img1 = a
            img2 = None
    else:
        return None

    n_shots_avail = img1.shape[0]
    seq_ids_arr = (np.asarray(seq_ids).ravel().astype(int)
                   if seq_ids is not None
                   else np.arange(1, n_shots_avail + 1))
    n_shots = min(len(seq_ids_arr), n_shots_avail)
    img1 = img1[:n_shots]
    if img2 is not None:
        img2 = img2[:n_shots]

    # Map shot k -> 0-indexed param. Scan.Params holds the scan-point
    # index per seq_id (1-indexed in MATLAB).
    params_arr = np.asarray(scan.get('Params', [])).ravel().astype(int)
    seq_idx = seq_ids_arr[:n_shots] - 1
    if params_arr.size:
        seq_idx = np.clip(seq_idx, 0, params_arr.size - 1)
        per_shot_param0 = params_arr[seq_idx] - 1   # 0-indexed param
    else:
        per_shot_param0 = seq_idx

    # Apply the filter mask if any (mask is indexed by 0-indexed param).
    keep = np.ones(n_shots, dtype=bool)
    if param_mask is not None and per_shot_param0.size:
        # param_mask is bool array sized to n_params; per-shot param
        # index could exceed that if the scan was extended, so guard.
        valid_pidx = per_shot_param0 < param_mask.size
        keep[~valid_pidx] = False
        keep[valid_pidx] = param_mask[per_shot_param0[valid_pidx]]

    img1f = img1[keep]
    img2f = img2[keep] if img2 is not None else None
    shot_index = (np.where(keep)[0] + 1).tolist()

    with np.errstate(invalid='ignore', divide='ignore'):
        loaded = img1f.sum(axis=1)
        n_sites_per_shot = img1f.shape[1] if img1f.size else 0
        loaded_frac = (loaded / max(1, n_sites_per_shot)).astype(float)
        survival = fp = None
        if img2f is not None:
            survived = (img1f & img2f).sum(axis=1)
            survival = np.where(loaded > 0, survived / loaded, np.nan)
            empty    = (~img1f).sum(axis=1)
            falsep   = (img2f & ~img1f).sum(axis=1)
            fp       = np.where(empty > 0, falsep / empty, np.nan)

    out = {
        'shot_index':    shot_index,
        'loaded_frac':   [float(v) for v in loaded_frac],
        'survival_frac': None if survival is None
                         else [None if not np.isfinite(v) else float(v)
                               for v in survival],
        'fp_frac':       None if fp is None
                         else [None if not np.isfinite(v) else float(v)
                               for v in fp],
        'param_values':  {},
    }

    # Recover swept-param values per shot using the FULL scan-params
    # table from the .mat (this needs the UNFILTERED scan_params dims).
    try:
        from yb_analysis.detection.scan_analysis import extract_scan_dims_h5
        mat_dims = extract_scan_dims_h5(bundle.get('mat_path')) or []
    except Exception:
        mat_dims = []
    # If we have multi-axis dims, the per-shot param value for each axis
    # decomposes from the flat scan-point index using MATLAB column-major
    # ordering (axis 0 fastest -> idx0 = p % s0).
    if mat_dims:
        for axis_idx, d in enumerate(mat_dims):
            name = d.get('name') or f'axis{axis_idx}'
            vals = np.asarray(d.get('values'), dtype=float).ravel()
            s = int(d.get('size') or 0)
            if not vals.size:
                continue
            p = per_shot_param0[keep]
            if axis_idx == 0:
                idx = (p % s) if s else p
            elif axis_idx == 1 and len(mat_dims) >= 2:
                s0 = int(mat_dims[0].get('size') or 1)
                idx = (p // s0) % s
            else:
                continue
            idx = np.clip(idx, 0, vals.size - 1)
            out['param_values'][name] = vals[idx].tolist()
    return out
