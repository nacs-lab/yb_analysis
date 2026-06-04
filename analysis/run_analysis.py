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
ANALYSIS_JSON = 'slm_analysis.json'   # Phase 5a closeout: cached SLM-side
                                       # analysis (legacy runs from before
                                       # the Phase 1 ledger existed).
                                       # Written by
                                       # yb_analysis.scripts.backfill_slm_analysis.


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
          # Phase 5a: per-shot rearrangement paths.
          'paths_per_shot': list[dict] | None,   # see _paths_per_shot
          'paths_frame': 'camera_bitorder' | 'knm_native' | None,
          'paths_n_shots_with_pairing': int,
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

    # ---- Averaged camera image (Phase 5a closeout) ---------------------
    # Single-image scans (NumImages=1, e.g. loading-test LACScans) get
    # an averaged-frame view in the Sequence-specific tab. The actual
    # PNG is computed lazily by the dashboard's /api/runs/<id>/avg_image
    # endpoint; here we just report whether it's available/computable.
    out['avg_image'] = _avg_image_info(scan_dir, scan)

    # ---- Per-shot rearrangement PATHS (Phase 5a) ----------------------
    # Joins slm_diag.h5/{loaded_paired,target_paired} (per-shot bit
    # indices stamped by SLMnet's pairing_extra) with slm_grid.json's
    # {init_grid, target_grid} coords to emit per-shot source→target
    # path lists. See plan §Bit-order invariant: both grids in the
    # sidecar are in MATLAB bit-order, possibly post-gridLocations
    # affine reorder if grid_rotation was set at setup.
    if diag_path.is_file():
        paths_info = _paths_per_shot(diag_path, scan_dir / GRID_JSON)
    else:
        paths_info = {
            'paths_per_shot': None,
            'paths_frame': None,
            'paths_n_shots_with_pairing': 0,
        }
    out.update(paths_info)

    # Paths overlay for the per-site map: list shots with non-empty
    # paths so the dashboard can build a "show paths for shot K"
    # picker, plus precompute the line-segment data for the first
    # such shot (default render). Format follows Plotly's NaN-
    # separated polyline convention: a single trace with x =
    # [x0, x1, NaN, x0', x1', NaN, ...].
    out['paths_overlay'] = _paths_overlay_summary(paths_info.get('paths_per_shot'))

    # ---- Legacy SLM analysis fallback (Phase 5a closeout) -------------
    # When the synced sidecars (slm_diag.h5, slm_grid.json) don't exist
    # for a run -- pre-Phase-1 rearrangement runs from before the per-
    # shot ledger format existed -- we cache the SLM-server's own
    # /runs/<run_id>/analysis response as `slm_analysis.json` via the
    # one-shot `backfill_slm_analysis.py` script. Surface its
    # contents as `slm_analysis_cached` so the dashboard can render
    # the same panels for legacy runs that new runs get from the
    # native sidecars.
    slm_an = _load_slm_analysis(scan_dir)
    out['slm_analysis_cached'] = slm_an is not None

    # ---- Target-aware survival (Phase 5a closeout) --------------------
    # The lab-side `summary.survival_mean` is computed PER-SITE from
    # logic1/logic2 and treats every loaded site equally. For
    # rearrangement scans that under-counts the survival the experiment
    # actually achieves: atoms that moved AWAY from their initial site
    # are counted as "lost" by the per-site metric, even though they
    # may have arrived at their target site. The target-aware metric
    # asks the right question instead: "of the sites the protocol
    # tried to fill, what fraction had an atom in img2?".
    #
    # Two paths to populate it:
    #   1. Lab-side from paths_per_shot (post-Phase-5a runs that have
    #      slm_diag.h5 with paired columns + slm_grid.json coords).
    #   2. From the cached `slm_analysis.json` for legacy runs (the
    #      SLM server already computed `summary[0].rate_pct` =
    #      TP_return, which is target-aware).
    # When neither is available, leave the field absent and the
    # dashboard falls back to plain `summary.survival_mean`.
    target_aware = _target_aware_survival(
        slm_an, out, scan, scan_params,
        mat_path=bundle.get('mat_path'),
        bundle=bundle,
        paths_per_shot=paths_info.get('paths_per_shot'),
        reps_per_param=reps_per_param,
        param_mask=param_mask)
    out['target_aware'] = target_aware
    if target_aware and target_aware.get('per_param_mean') is not None:
        # Override the canonical curve so the dashboard's default
        # survival panel shows the rearrangement-meaningful number.
        # Keep the original per-site number under a backup key so
        # nothing's lost.
        out['summary']['survival_mean_per_site'] = out['summary']['survival_mean']
        out['summary']['survival_sem_per_site']  = out['summary']['survival_sem']
        out['summary']['survival_mean'] = target_aware['per_param_mean']
        out['summary']['survival_sem']  = target_aware['per_param_sem']
        out['summary']['survival_source'] = target_aware['source']
    elif target_aware and target_aware.get('overall_mean') is not None:
        # No per-param mapping, but we know the overall TP rate. Annotate
        # the summary so the dashboard can show "overall TP=84.4%" even
        # if it can't draw a per-param curve.
        out['summary']['survival_overall_target_aware'] = target_aware['overall_mean']
        out['summary']['survival_source'] = target_aware['source'] + '_overall_only'

    # ---- per_site override (Phase 5a closeout) -----------------------
    # When the cached SLM analysis exists, replace the lab-computed
    # per_site map (lab metric = per-site loaded-AND-survived, which
    # for rearrangement is wrong everywhere atoms moved AWAY) with the
    # SLM's TP/FP-aware per_site map. This adds `tp_rate`, `fp_rate`,
    # `is_target_site`, `is_nontarget_site` fields so the dashboard
    # can mark target sites visibly. Lab-side fallback (when no SLM
    # cache) keeps the existing logic1/logic2 per-site computation.
    #
    # Filter-aware nuance: the SLM cache is a WHOLE-SCAN per_site
    # aggregate -- it has no per-bin breakdown to slice. So when the
    # caller passed a filter we honor that filter on the lab-computed
    # per_site (which IS filter-aware because logic1/logic2 above were
    # already sliced by param_mask) and skip the SLM override. Cost:
    # under a filter the dashboard's per-site map shows lab per-site
    # survival without target/non-target markers; the override comes
    # back when the filter is cleared. A short note on the dashboard
    # explains.
    filter_active = bool(out.get('filter_active'))
    # Priority order (matches target_aware):
    # 1. lab-paths — when paths_per_shot has real target data
    #    (filter-aware, target markers preserved under any filter)
    # 2. SLM cache — legacy runs without paths data, NO filter
    # 3. lab-computed lab logic1/logic2 — legacy runs WITH filter
    #    (filter-aware but target markers lost)
    paths_list = paths_info.get('paths_per_shot') or []
    have_real_paths = any(
        (isinstance(e, dict) and (e.get('target_xy') or [])) for e in paths_list)
    if have_real_paths:
        ps_override = _per_site_from_lab_paths(paths_list, bundle, scan)
        if ps_override is not None:
            if out.get('per_site') is not None:
                out['per_site_lab_computed'] = out['per_site']
            out['per_site'] = ps_override
    elif slm_an is not None and not filter_active:
        ps_override = _per_site_from_slm_analysis(slm_an)
        if ps_override is not None:
            if out.get('per_site') is not None:
                out['per_site_lab_computed'] = out['per_site']
            out['per_site'] = ps_override
    elif slm_an is not None and filter_active and out.get('per_site') is not None:
        # Annotate so the dashboard can show a hint.
        out['per_site']['source'] = 'lab_filtered'
        out['per_site']['note'] = (
            'filter active — per-site rates from lab logicals; '
            'target/non-target markers suppressed (legacy run, SLM '
            'cache is whole-scan only)')

    # ---- per_iteration TP override (Phase 5a closeout) ----------------
    # When target_aware is in effect, the per-iteration "survival"
    # curve in the dashboard should show per-shot TP (target survival),
    # not per-site survival. Both sources expose per_shot_tp:
    #   - lab_paths       : computed shot-by-shot from logic2 at the
    #                        targets named in paths_per_shot
    #   - slm_server_cached: from cached `tp_series.hits / .elig`
    # If per_shot_tp isn't available, leave per_iteration.survival_frac
    # as the lab-computed per-site value.
    if target_aware and isinstance(target_aware.get('per_shot_tp'), list) \
            and out.get('per_iteration'):
        pi = out['per_iteration']
        per_shot_tp = target_aware['per_shot_tp']
        # pi.shot_index is 1-indexed position in bundle.seq_ids, which
        # matches the position in per_shot_tp.
        new_survival = []
        for s in pi.get('shot_index') or []:
            i = int(s) - 1
            v = per_shot_tp[i] if 0 <= i < len(per_shot_tp) else None
            new_survival.append(
                None if v is None or (isinstance(v, float) and v != v)
                else float(v))
        # Keep the per-site backup so the legacy view is still
        # accessible programmatically.
        pi['survival_frac_per_site'] = pi.get('survival_frac')
        pi['survival_frac'] = new_survival
        pi['survival_source'] = target_aware.get('source')
        pi['survival_label'] = 'TP (target)'

    # ---- diag_aggregate fall-through for legacy runs -----------------
    # When slm_diag.h5 isn't present, build the same rollup from the
    # cached SLM analysis's `diag_series_all` block (which has all the
    # per-shot diag arrays the SLM server already collected).
    if out['diag_aggregate'] is None and slm_an is not None:
        out['diag_aggregate'] = _diag_aggregate_from_slm_analysis(slm_an)

    # ---- Upstream-compat fields ---------------------------------------
    # These are computed by the SLM server and ride through `slm_diag.h5`
    # via the diag_json vlen column. Lab-side replay would need the
    # same algorithms; for now we leave them None and the dashboard
    # hides the matching panels. Phase 5b can decide whether to mirror
    # them here vs. lazy-fetch from SLM. (Phase 5a populates only
    # `paths_per_shot`; survival_vs_distance requires the lab-side
    # target-imaging story not in scope here.)
    #
    # For LEGACY runs where slm_analysis.json is cached, surface the
    # server's pre-computed survival_vs_distance verbatim. The user is
    # explicit that lab-side computation is the eventual home (Phase
    # 5.5 Track B) — this is a transitional copy-through.
    if slm_an is not None:
        out['survival_vs_distance'] = slm_an.get('survival_vs_distance')
        out['survival_vs_distance_per_step'] = slm_an.get(
            'survival_vs_distance_per_step')
        out['per_shot_extra'] = slm_an.get('per_shot_extra')
        out['round1'] = slm_an.get('round1')
    else:
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
            if arr.dtype.kind == 'f':
                # pyctrl JSON sidecar: char codes (scanname) pass through
                # _config_arrays, which floats numeric lists. Decode iff every
                # entry is an integral, in-range codepoint (else it's a real
                # numeric field, not a string — leave it alone).
                try:
                    codes = arr.tolist()
                    if codes and all(float(x).is_integer() and 0 < x < 0x110000
                                     for x in codes):
                        s = ''.join(chr(int(x)) for x in codes).strip()
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

    # pyctrl scans have no .mat (mat_path is None), but their JSON config
    # carries ScanGroup.base.vars as a plain dict — recover dotted axis names
    # from it so the dashboard labels the sweep correctly (not "axis0").
    if not cols:
        try:
            from yb_analysis.detection.scan_analysis import extract_scan_dims
            for d in (extract_scan_dims(scan) or []):
                cols.append(d.get('name') or f'axis{len(cols)}')
        except Exception as ex:
            logger.debug('extract_scan_dims (dict) failed: %s', ex)

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


AVG_IMAGE_PNG = 'avg_image.png'


def _scan_data_h5(scan_dir: Path) -> Optional[Path]:
    """Return the scan's data_*.h5 path (the one with /imgs, /logicals)."""
    cands = sorted(scan_dir.glob('data_*.h5'))
    return cands[0] if cands else None


def _avg_image_info(scan_dir: Path, scan: dict) -> Optional[dict]:
    """Report whether an averaged camera image is available / computable.

    Only single-image scans (NumImages=1, e.g. LACScan loading-test
    families) get the avg-image card: when there's no second image
    the per-site survival panel is empty anyway, and an averaged
    "where did atoms load" view is the actually-informative summary.

    Returns ``None`` when the scan has no ``/imgs`` dataset or
    NumImages > 1. Otherwise::

        {
          'available': bool,         # PNG already cached on disk
          'computable': bool,        # we have imgs but no PNG yet
          'png_path': str | None,    # absolute path when available
          'n_shots': int,            # how many frames are averaged
          'image_shape': [H, W],     # image dims when computable/available
        }
    """
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    if num_images != 1:
        return None
    h5_path = _scan_data_h5(scan_dir)
    if h5_path is None:
        return None
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            if 'imgs' not in f:
                return None
            shape = tuple(int(s) for s in f['imgs'].shape)
    except (OSError, KeyError):
        return None
    if len(shape) != 3:
        return None
    n_shots, h, w = shape
    png_path = scan_dir / AVG_IMAGE_PNG
    return {
        'available':   png_path.is_file(),
        'computable':  not png_path.is_file(),
        'png_path':    str(png_path) if png_path.is_file() else None,
        'n_shots':     int(n_shots),
        'image_shape': [int(h), int(w)],
    }


def ensure_avg_image_png(scan_dir, *, batch_size: int = 16) -> Optional[Path]:
    """Compute the per-shot-mean of /imgs and write it as avg_image.png.

    Idempotent: returns the existing path if the PNG is already cached.
    Reads imgs in batches to bound memory; total cost is one float64
    H×W accumulator (~35 MB for a 2100×2100 array). For a 287-shot
    scan this takes ~9 s on the lab PC (most of which is HDF5 chunk
    decompression). Returns the cache path or None on failure.
    """
    scan_dir = Path(scan_dir)
    png_path = scan_dir / AVG_IMAGE_PNG
    if png_path.is_file():
        return png_path
    h5_path = _scan_data_h5(scan_dir)
    if h5_path is None:
        return None
    try:
        import h5py
        from PIL import Image
    except ImportError as ex:
        logger.warning('ensure_avg_image_png: missing dependency %s', ex)
        return None
    try:
        with h5py.File(h5_path, 'r') as f:
            d = f.get('imgs')
            if d is None or d.ndim != 3:
                return None
            n_shots = d.shape[0]
            if n_shots == 0:
                return None
            acc = np.zeros(d.shape[1:], dtype=np.float64)
            for i in range(0, n_shots, batch_size):
                acc += d[i:i + batch_size].astype(np.float64).sum(axis=0)
            mean_img = acc / n_shots
        # Normalize to uint8 for PNG. Preserve contrast by min-max.
        lo = float(np.nanmin(mean_img))
        hi = float(np.nanmax(mean_img))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            scaled = np.zeros(mean_img.shape, dtype=np.uint8)
        else:
            scaled = ((mean_img - lo) / (hi - lo) * 255.0).astype(np.uint8)
        tmp = png_path.with_suffix('.png.tmp')
        Image.fromarray(scaled).save(tmp, format='PNG', optimize=True)
        os.replace(tmp, png_path)
        return png_path
    except (OSError, ValueError) as ex:
        logger.warning('ensure_avg_image_png(%s) failed: %s', scan_dir, ex)
        return None


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


def _load_slm_analysis(scan_dir: Path) -> Optional[dict]:
    """Read `slm_analysis.json` (Phase 5a closeout — cached SLM-server
    /runs/<run_id>/analysis response) if it exists. Returns parsed
    dict or None."""
    p = scan_dir / ANALYSIS_JSON
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning('_load_slm_analysis: %s read failed: %s', p, ex)
        return None


def _target_aware_survival(slm_an: Optional[dict],
                            out: dict,
                            scan: dict,
                            scan_params: np.ndarray,
                            *,
                            mat_path: Optional[str] = None,
                            bundle: Optional[dict] = None,
                            paths_per_shot: Optional[list] = None,
                            reps_per_param: Optional[np.ndarray] = None,
                            param_mask: Optional[np.ndarray] = None) -> Optional[dict]:
    """Build target-aware survival per scan param.

    Priority:
      1. **Lab-side from `paths_per_shot`** (new runs — Phase 5a
         ledger format with per-shot target_paired masks).
         Computes TP from the lab's OWN logic2 — never touches the
         SLM cache. ``source = 'lab_paths'`` in the result. The
         operator's preferred path: "SLM bitstrings are less
         trustworthy than the pure lab ones".
      2. **From cached `slm_analysis.json`** (legacy runs that
         predate Phase 1's per-shot ledger). Server-computed,
         passed through and re-ordered to lab `scan_params`.
         ``source = 'slm_server_cached'``.
      3. None — dashboard falls back to plain
         ``summary.survival_mean`` (per-site survival).

    Returns::

        {
          'source':        'slm_server_cached' | 'lab_paths',
          'per_param_mean': list[float] (n_params),  # fraction, 0..1
          'per_param_sem':  list[float] (n_params),
          'overall_mean':  float,  # whole-scan target survival
          'overall_sem':   float,
          'loss_overall':  float | None,
          'fp_overall':    float | None,
          'axes_matched':  list[str],
          'axes_lab':      list[str],
          'axes_slm':      list[str],
        }

    or None when no signal available.
    """
    # Try lab-paths first — it bypasses SLM entirely for the TP
    # computation, using only data from the lab-side data_*.h5.
    if paths_per_shot:
        result = _target_aware_from_lab_paths(
            paths_per_shot, bundle, scan, scan_params,
            reps_per_param=reps_per_param, param_mask=param_mask)
        if result is not None:
            return result
    if slm_an is not None:
        return _target_aware_from_slm_analysis(slm_an, out, scan_params,
                                                mat_path=mat_path)
    return None


def _target_aware_from_lab_paths(paths_per_shot: list,
                                  bundle: Optional[dict],
                                  scan: dict,
                                  scan_params: np.ndarray,
                                  *,
                                  reps_per_param: Optional[np.ndarray] = None,
                                  param_mask: Optional[np.ndarray] = None
                                  ) -> Optional[dict]:
    """Lab-side target-aware TP using paths_per_shot + lab img2.

    For each shot k, ``paths_per_shot[k]['target_xy']`` lists the
    physical (y, x) coords the protocol tried to fill. We map each
    target coord to the nearest lab-side site using
    ``Scan.initGridLocationsX/Y``, look up ``img2[shot, lab_site_idx]``
    from the lab's own logic2 (from data_*.h5), and divide by target
    count to get per-shot TP. Aggregating per scan-param gives a
    per-param mean + Bernoulli SEM.

    The lab's bitstring detection (gridLocations + thresholds) is the
    source of truth — no SLM cache values touch this path. Returns
    None when prerequisites are missing (no bundle, no targets,
    missing lab gridLocations).
    """
    if not paths_per_shot or bundle is None:
        return None
    # Lab-side site coords from Scan.initGridLocationsX/Y (used by the
    # per-site map). Without these we can't map target_xy → lab site.
    gx, gy = _site_grid_xy(scan)
    if not gx or not gy:
        return None
    site_xy = np.column_stack([np.asarray(gy, dtype=float),
                                np.asarray(gx, dtype=float)])  # (n_sites, 2) (y, x)
    if site_xy.size == 0:
        return None

    # Lab img2 (per-shot, per-site detection from data_*.h5).
    raw1 = bundle.get('logicals_img1')
    raw2 = bundle.get('logicals_img2')
    raw  = bundle.get('logicals')
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    if raw1 is not None and raw2 is not None:
        img2 = np.asarray(raw2, dtype=np.uint8)
    elif raw is not None:
        a = np.asarray(raw, dtype=np.uint8)
        if num_images >= 2:
            img2 = a[(num_images - 1)::num_images]
        else:
            img2 = a
    else:
        return None
    if img2.size == 0 or img2.shape[1] != site_xy.shape[0]:
        return None

    seq_ids = bundle.get('seq_ids')
    if seq_ids is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()
    n_shots_avail = min(len(seq_ids), img2.shape[0])
    seq_ids = seq_ids[:n_shots_avail]
    img2 = img2[:n_shots_avail]

    # Map seq_id -> 0-indexed scan-param (via Scan.Params, MATLAB 1-indexed).
    params_arr = np.asarray(scan.get('Params', [])).ravel().astype(int)
    seq_idx = seq_ids - 1
    if params_arr.size:
        seq_idx_safe = np.clip(seq_idx, 0, params_arr.size - 1)
        per_shot_param0 = params_arr[seq_idx_safe] - 1
    else:
        per_shot_param0 = seq_idx

    # We aggregate per-shot results in the UNFILTERED param space
    # (size = max(per_shot_param0) + 1), then slice down to the
    # caller's filtered scan_params by `param_mask` at the very end.
    # The caller may pass `scan_params` ALREADY filtered (analyze_scan_dir
    # filters before calling), so we can't use `scan_params.shape[0]`
    # as the aggregation size — that would be wrong by exactly the
    # filter mask. Without this, most shots get skipped under any
    # filter and the lab-paths source effectively disappears.
    if param_mask is not None:
        n_params_unfilt = int(param_mask.size)
    elif per_shot_param0.size:
        n_params_unfilt = int(per_shot_param0.max() + 1)
    else:
        sp_tmp = scan_params if scan_params.ndim == 2 else scan_params.reshape(-1, 1)
        n_params_unfilt = int(sp_tmp.shape[0])
    sum_tp = np.zeros(n_params_unfilt, dtype=np.float64)
    n_eligible = np.zeros(n_params_unfilt, dtype=np.int64)
    # Per-shot TP for the per-iteration override (indexed 0..n_shots_avail-1
    # matching bundle's processed-shot order, i.e. lab seq_ids order).
    per_shot_tp = [None] * n_shots_avail

    # Build a seq_id -> path entry index map (paths_per_shot may not be
    # ordered by seq_id, and may include rows with empty targets).
    by_seq = {}
    for entry in paths_per_shot:
        if not isinstance(entry, dict):
            continue
        sid = entry.get('seq_id')
        if sid is None:
            continue
        by_seq[int(sid)] = entry

    n_with_targets = 0
    for k in range(n_shots_avail):
        sid = int(seq_ids[k])
        entry = by_seq.get(sid)
        if entry is None:
            continue
        target_xy = entry.get('target_xy') or []
        if not target_xy:
            continue
        # Skip rows whose target_xy has placeholders (out-of-range
        # indices, e.g. from corrupted diag rows).
        valid_xy = [tuple(t) for t in target_xy
                    if isinstance(t, (list, tuple)) and len(t) == 2
                    and all(isinstance(x, (int, float)) and x is not None
                            for x in t)]
        if not valid_xy:
            continue
        t_arr = np.asarray(valid_xy, dtype=float)   # (n_targets, 2)
        # Nearest-neighbor lab site for each target (y, x).
        # (n_targets, n_sites) distance squared.
        diffs = site_xy.reshape(1, -1, 2) - t_arr.reshape(-1, 1, 2)
        d2 = (diffs ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)  # (n_targets,)
        # Reject targets too far from any lab site (>= 2x median NN
        # distance) — safety net so a coord-frame mismatch surfaces as
        # an empty TP rather than a garbage one.
        nearest_d = np.sqrt(d2[np.arange(t_arr.shape[0]), nearest])
        # Heuristic threshold: 2x typical site spacing. Approximate
        # site spacing as the median of per-site nearest-neighbor
        # distance, computed lazily once.
        if 'site_spacing' not in by_seq:   # cache in the dict object
            # Per-site NN distance (subsample if huge).
            ssub = site_xy[: min(site_xy.shape[0], 1000)]
            d2s = ((ssub.reshape(-1, 1, 2) - ssub.reshape(1, -1, 2)) ** 2
                   ).sum(axis=2)
            d2s[np.arange(d2s.shape[0]), np.arange(d2s.shape[0])] = np.inf
            spacing = float(np.median(np.sqrt(d2s.min(axis=1))))
            by_seq['site_spacing'] = max(spacing, 1.0)
        spacing = by_seq['site_spacing']
        ok = nearest_d <= 2.0 * spacing
        if not ok.any():
            continue
        lab_sites = nearest[ok]
        # TP this shot: of the matched target sites, how many showed
        # an atom in img2?
        n_targets_matched = int(ok.sum())
        n_hit = int(img2[k, lab_sites].sum())
        tp_this_shot = n_hit / n_targets_matched
        per_shot_tp[k] = tp_this_shot
        # Scan-param this shot belongs to (UNFILTERED 0-indexed).
        p0 = int(per_shot_param0[k])
        if p0 < 0 or p0 >= n_params_unfilt:
            continue
        sum_tp[p0] += tp_this_shot
        n_eligible[p0] += 1
        n_with_targets += 1

    if n_with_targets == 0:
        return None

    with np.errstate(invalid='ignore', divide='ignore'):
        per_mean_unfilt = np.where(n_eligible > 0, sum_tp / n_eligible, np.nan)
        per_sem_unfilt  = np.where(n_eligible > 0,
                                     np.sqrt(per_mean_unfilt * (1 - per_mean_unfilt)
                                             / np.maximum(1, n_eligible)),
                                     np.nan)
    if not np.any(np.isfinite(per_mean_unfilt)):
        return None

    # Apply the filter at the END (we aggregated in unfiltered space).
    if param_mask is not None and per_mean_unfilt.size == param_mask.size:
        per_mean = per_mean_unfilt[param_mask]
        per_sem  = per_sem_unfilt[param_mask]
        elig_used = n_eligible[param_mask]
    else:
        per_mean = per_mean_unfilt
        per_sem  = per_sem_unfilt
        elig_used = n_eligible

    # Eligibility-weighted overall TP across the FILTERED subset (so
    # the dashboard's "overall TP" tile reflects whatever filter is on).
    valid = np.isfinite(per_mean)
    if valid.any():
        finite_idx = np.where(valid)[0]
        means_f = per_mean[finite_idx]
        elig_f  = elig_used[finite_idx]
        total_eligible = int(elig_f.sum())
        overall = (float(np.average(means_f, weights=elig_f))
                    if elig_f.sum() else float('nan'))
        overall_sem = (float(np.sqrt(overall * (1 - overall)
                                       / max(1, total_eligible)))
                        if 0 <= overall <= 1 else None)
    else:
        overall, overall_sem, total_eligible = None, None, 0

    return {
        'source':          'lab_paths',
        'per_param_mean':  per_mean.tolist(),
        'per_param_sem':   per_sem.tolist(),
        'overall_mean':    overall,
        'overall_sem':     overall_sem,
        'loss_overall':    (1.0 - overall) if overall is not None else None,
        'fp_overall':      None,        # lab-computed FP needs the non-target
                                         # site convention -- skipped here; the
                                         # per_site fp_rate (when present from
                                         # SLM cache) covers it for now.
        'axes_matched':    list(scan.get('ScanVar') or []),
        'axes_lab':        list((scan_params.shape[1] if scan_params.ndim==2
                                  else 1) * [None]),
        'axes_slm':        [],
        'n_shots_with_paths': int(n_with_targets),
        'n_eligible':      int(total_eligible),
        # Per-shot TP indexed by 0..n_shots_avail-1 (same order as
        # bundle.logicals/seq_ids), so the per_iteration override can
        # look up directly by `pi.shot_index[k] - 1`.
        'per_shot_tp':     per_shot_tp,
    }


def _target_aware_from_slm_analysis(slm_an: dict,
                                      out: dict,
                                      scan_params: np.ndarray,
                                      *,
                                      mat_path: Optional[str] = None) -> Optional[dict]:
    """Extract target-aware survival from the cached SLM analysis.

    The SLM's analysis carries:
      * `summary` = list of {category, hits, eligible, rate_pct}
        with categories TP_return / Loss / FP_total.
      * `per_bin` = list of per-scan-point dicts, each with the
        sweep axes named (e.g. ``nsteps``, ``step_period_ms``) plus
        ``TP_return_%``, ``Loss_%``, ``FP_total_%``, ``n_shots``.
      * `sweep` = {kind: '1d'|'2d', params: [name, ...], p1_vals, p2_vals}.

    We map per_bin entries to the LAB-side scan_params by matching
    axis names. If all lab axes match SLM axes, per_param_mean is a
    direct array. If axes don't align, we degrade to an overall scalar.
    """
    summary_list = slm_an.get('summary') or []
    overall_tp = next((s.get('rate_pct') for s in summary_list
                       if isinstance(s, dict)
                       and s.get('category') == 'TP_return'), None)
    overall_loss = next((s.get('rate_pct') for s in summary_list
                         if isinstance(s, dict)
                         and s.get('category') == 'Loss'), None)
    overall_fp = next((s.get('rate_pct') for s in summary_list
                       if isinstance(s, dict)
                       and s.get('category') == 'FP_total'), None)

    sweep = slm_an.get('sweep') or {}
    slm_axes = sweep.get('params') or []
    per_bin = slm_an.get('per_bin') or []

    # Lab-side sweep axes
    lab_sweep = out.get('sweep') or {}
    lab_axes  = lab_sweep.get('cols') or []
    lab_vals  = lab_sweep.get('values') or []

    per_param_mean = None
    per_param_sem  = None

    # Try to map per_bin -> lab scan-point ordering. Both sides have
    # the same set of (axis, value) tuples; we just need to look up
    # each bin's tuple in the lab's enumerated scan-points.
    if per_bin and lab_axes and scan_params.size and slm_axes:
        sp_raw = scan_params if scan_params.ndim == 2 else scan_params.reshape(-1, 1)
        # Defensive: when `lab_axes` reports more cols than
        # scan_params actually has (build_sweep can over-report from
        # extract_scan_dims_h5 vs the dereffed scan_params), trim to
        # the actual axis count. This avoids broadcast failures like
        # `(40,2)` vs `(1,3)` on multi-axis scans.
        eff_n_axes = sp_raw.shape[1]
        lab_axes_eff = list(lab_axes)[:eff_n_axes]
        # Match SLM axis name -> lab axis name (index into lab_axes_eff)
        # with fuzzy suffix match. Lab axis names are typically dotted
        # paths like `rearrange_kwargs.nsteps`; SLM axes are bare
        # names like `nsteps`. Direct match OR suffix match
        # (case-insensitive after splitting on '.').
        def _lab_axis_match(slm_name):
            sn = slm_name.lower()
            for i, lab in enumerate(lab_axes_eff):
                ln = lab.lower()
                if ln == sn:
                    return i
                if ln.split('.')[-1] == sn or sn.split('.')[-1] == ln:
                    return i
            return None

        slm_to_lab_idx = [_lab_axis_match(n) for n in slm_axes]
        # Require that EVERY SLM axis maps to a distinct lab axis. If
        # there's overlap (two SLM axes mapping to one lab axis) or
        # any unmapped, abandon the per-param join. Saves us from
        # silently mis-attributing rates.
        if (all(idx is not None for idx in slm_to_lab_idx)
                and len(set(slm_to_lab_idx)) == len(slm_to_lab_idx)):
            sp = sp_raw
            n_params = sp.shape[0]
            means = np.full(n_params, np.nan, dtype=float)
            sems  = np.full(n_params, np.nan, dtype=float)
            elig  = np.zeros(n_params, dtype=np.int64)
            # Build the indexer once: which lab columns to check, and
            # the SLM axis name that supplies the value for each.
            check_cols = list(slm_to_lab_idx)
            slm_names_for_cols = list(slm_axes)
            sp_sub = sp[:, check_cols]
            # Per-axis tolerance: relative to the swept range of the
            # lab side's matching column.
            with np.errstate(invalid='ignore'):
                tol = np.maximum(1e-6,
                                  1e-4 * (np.nanmax(sp_sub, axis=0) -
                                          np.nanmin(sp_sub, axis=0)))
            for entry in per_bin:
                if not isinstance(entry, dict):
                    continue
                target_vec = np.empty(len(check_cols), dtype=float)
                bad = False
                for k, slm_name in enumerate(slm_names_for_cols):
                    v = entry.get(slm_name)
                    if v is None:
                        bad = True
                        break
                    target_vec[k] = float(v)
                if bad:
                    continue
                with np.errstate(invalid='ignore'):
                    diffs = np.abs(sp_sub - target_vec.reshape(1, -1))
                hit = np.all(diffs <= tol.reshape(1, -1), axis=1)
                idxs = np.where(hit)[0]
                if idxs.size == 0:
                    continue
                rate_pct = entry.get('TP_return_%') or entry.get('TP_%')
                n_shots  = entry.get('n_shots') or 0
                if rate_pct is None:
                    continue
                p = float(rate_pct) / 100.0
                sem = (np.sqrt(p * (1 - p) / max(1, n_shots))
                       if 0 <= p <= 1 else np.nan)
                for j in idxs:
                    means[j] = p
                    sems[j]  = sem
                    elig[j]  = int(n_shots)
            if np.any(np.isfinite(means)):
                per_param_mean = means.tolist()
                per_param_sem  = sems.tolist()
                # When per-param mean is populated and the lab side
                # is showing a filtered subset, recompute the overall
                # rate from THIS subset (eligibility-weighted). Gives
                # the dashboard's "overall TP" tile a number that
                # actually reflects what the operator is looking at.
                finite = np.isfinite(means)
                if finite.any() and elig[finite].sum() > 0:
                    overall_p_filtered = float(np.average(
                        means[finite], weights=elig[finite]))
                    overall_eligible_filtered = int(elig[finite].sum())
                else:
                    overall_p_filtered = None
                    overall_eligible_filtered = None
            else:
                overall_p_filtered = None
                overall_eligible_filtered = None
        else:
            overall_p_filtered = None
            overall_eligible_filtered = None
    else:
        overall_p_filtered = None
        overall_eligible_filtered = None

    if (overall_tp is None and per_param_mean is None):
        return None

    # Prefer the filtered, eligibility-weighted overall when per-param
    # matching produced a result — that's the lab-side filter-aware
    # number. Fall back to the SLM's whole-scan summary otherwise.
    if overall_p_filtered is not None:
        overall_p = overall_p_filtered
        overall_eligible = overall_eligible_filtered
    else:
        overall_eligible = next((s.get('eligible') for s in summary_list
                                  if isinstance(s, dict)
                                  and s.get('category') == 'TP_return'), None)
        overall_p = (overall_tp / 100.0) if overall_tp is not None else None
    overall_sem = (np.sqrt(overall_p * (1 - overall_p) / max(1, overall_eligible))
                   if overall_p is not None and overall_eligible
                   else None)

    # Re-derive matched-axis list using the same fuzzy matcher as above
    # so reports reflect what actually mapped.
    def _matched_lab_for(s):
        sn = s.lower()
        for lab in (lab_axes or []):
            ln = lab.lower()
            if ln == sn or ln.split('.')[-1] == sn or sn.split('.')[-1] == ln:
                return lab
        return None
    axes_matched = [_matched_lab_for(a) for a in slm_axes]
    axes_matched = [a for a in axes_matched if a is not None]

    # Per-shot TP for the per-iteration override: SLM cache puts
    # per-shot hits + eligibility under `tp_series`. Each entry
    # corresponds to one /slm/rearrange call in SLM-receive order;
    # for non-aborted runs this aligns 1:1 with lab seq_ids order.
    # We index by 0-based shot position (= lab seq_id - 1).
    per_shot_tp = None
    ts = slm_an.get('tp_series') or {}
    ts_hits = ts.get('hits')
    ts_elig = ts.get('elig')
    if isinstance(ts_hits, list) and isinstance(ts_elig, list) \
            and len(ts_hits) == len(ts_elig) and len(ts_hits) > 0:
        per_shot_tp = []
        for h, e in zip(ts_hits, ts_elig):
            if h is None or e is None or e == 0:
                per_shot_tp.append(None)
            else:
                per_shot_tp.append(float(h) / float(e))

    return {
        'source':          'slm_server_cached',
        'per_param_mean':  per_param_mean,
        'per_param_sem':   per_param_sem,
        'overall_mean':    overall_p,
        'overall_sem':     overall_sem,
        'loss_overall':    (overall_loss / 100.0) if overall_loss is not None else None,
        'fp_overall':      (overall_fp / 100.0) if overall_fp is not None else None,
        'axes_matched':    axes_matched,
        'axes_lab':        list(lab_axes),
        'axes_slm':        list(slm_axes),
        'per_shot_tp':     per_shot_tp,
    }


def _paths_overlay_summary(paths_per_shot: Optional[list]) -> Optional[dict]:
    """Precompute the Plotly-friendly line-segment data for a paths
    overlay on the per-site map.

    Returns a dict with:
      shot_indices: list[int]   1-indexed shot positions that have
                                  non-empty paths (for the picker)
      seq_ids:      list[int]   matching MATLAB seq_ids
      default_idx:  int         which shot to render by default
                                  (the first one, 0-indexed into
                                  shot_indices)
      segments_x / segments_y / segments_seq_ids:
                                NaN-separated polyline for the
                                default shot (one [x0, x1, NaN] per
                                source→target pair, plus a
                                seq_ids-paired per_segment_shot list
                                so the JS can rebuild for other
                                shots without a round-trip)
      all_segments: list[dict]  per-shot precomputed segments:
                                {'shot_index': int, 'seq_id': int,
                                 'x': [...], 'y': [...]}.
                                Capped at first 50 shots; the
                                dashboard can lazy-fetch more if needed.

    Returns None when no shots have paths data.
    """
    if not paths_per_shot:
        return None
    eligible = []
    for i, entry in enumerate(paths_per_shot):
        if not isinstance(entry, dict):
            continue
        init_xy = entry.get('init_xy') or []
        tgt_xy  = entry.get('target_xy') or []
        if not init_xy or not tgt_xy:
            continue
        n_pairs = min(len(init_xy), len(tgt_xy))
        if n_pairs == 0:
            continue
        valid_pairs = []
        for k in range(n_pairs):
            src = init_xy[k]
            dst = tgt_xy[k]
            if not (isinstance(src, (list, tuple)) and len(src) == 2
                    and isinstance(dst, (list, tuple)) and len(dst) == 2):
                continue
            if any(v is None for v in src) or any(v is None for v in dst):
                continue
            valid_pairs.append((float(src[0]), float(src[1]),
                                 float(dst[0]), float(dst[1])))
        if not valid_pairs:
            continue
        eligible.append({
            'shot_index': i + 1,         # 1-indexed
            'seq_id':     int(entry.get('seq_id') or (i + 1)),
            'pairs':      valid_pairs,
        })
    if not eligible:
        return None

    def _segments(pairs):
        # Plotly polyline: x = [x0, x1, NaN, x0', x1', NaN, ...]
        # Sidecar coords are (y, x) — image-space x is the SECOND
        # component. Flip to (x_for_plot, y_for_plot) accordingly so
        # the rendered overlay aligns with the per-site map (which
        # plots site_xy[:, 1] on x and site_xy[:, 0] on y).
        xs, ys = [], []
        for (y0, x0, y1, x1) in pairs:
            xs.extend([x0, x1, float('nan')])
            ys.extend([y0, y1, float('nan')])
        return xs, ys

    # Precompute first 50 shots' segment arrays so the dashboard can
    # switch instantly without a refetch.
    all_segments = []
    for e in eligible[:50]:
        xs, ys = _segments(e['pairs'])
        all_segments.append({
            'shot_index': e['shot_index'],
            'seq_id':     e['seq_id'],
            'x':          xs,
            'y':          ys,
            'n_paths':    len(e['pairs']),
        })

    return {
        'shot_indices':    [e['shot_index'] for e in eligible],
        'seq_ids':         [e['seq_id'] for e in eligible],
        'default_idx':     0,
        'segments_x':      all_segments[0]['x'] if all_segments else [],
        'segments_y':      all_segments[0]['y'] if all_segments else [],
        'all_segments':    all_segments,
        'n_shots_with_paths': len(eligible),
        'n_precomputed':   len(all_segments),
    }


def _per_site_from_lab_paths(paths_per_shot: list,
                              bundle: dict,
                              scan: dict) -> Optional[dict]:
    """Lab-side target-aware per_site map from paths_per_shot + lab img2.

    For each shot with non-empty target_xy, map every target coord to
    its nearest lab site (via Scan.initGridLocationsX/Y), then count
    TP hits at target sites and FP hits at non-target sites per shot.
    Aggregating over shots gives the per-site map the dashboard
    renders — entirely lab-computed, never touching SLM cache. Fully
    filter-aware because the caller passes already-filtered img1/img2.

    Returns same shape as `_per_site_from_slm_analysis` (so the
    dashboard's renderer doesn't care which side built it):
      x, y, loading_rate, survival_mean (= tp_rate at target sites,
      NaN elsewhere), fp_rate, tp_rate, tp_elig, fp_elig,
      is_target_site, is_nontarget_site, source='lab_paths'.

    Returns None when prerequisites are missing.
    """
    if not paths_per_shot or bundle is None:
        return None
    gx, gy = _site_grid_xy(scan)
    if not gx or not gy:
        return None
    site_xy = np.column_stack([np.asarray(gy, dtype=float),
                                np.asarray(gx, dtype=float)])
    n_sites = site_xy.shape[0]
    if n_sites == 0:
        return None

    raw1 = bundle.get('logicals_img1')
    raw2 = bundle.get('logicals_img2')
    raw  = bundle.get('logicals')
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    if raw1 is not None and raw2 is not None:
        img1 = np.asarray(raw1, dtype=np.uint8)
        img2 = np.asarray(raw2, dtype=np.uint8)
    elif raw is not None:
        a = np.asarray(raw, dtype=np.uint8)
        if num_images >= 2:
            img1 = a[0::num_images]
            img2 = a[(num_images - 1)::num_images]
        else:
            img1 = a
            img2 = np.zeros_like(a)
    else:
        return None
    if img2.shape[1] != n_sites:
        return None

    seq_ids = bundle.get('seq_ids')
    if seq_ids is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()
    n_shots = min(len(seq_ids), img2.shape[0], img1.shape[0])
    if n_shots == 0:
        return None

    by_seq = {int(e['seq_id']): e for e in paths_per_shot
               if isinstance(e, dict) and 'seq_id' in e}

    # Build per-shot boolean target mask. Memory: n_shots × n_sites
    # bytes — 1000×3270 = 3 MB max.
    target_mask = np.zeros((n_shots, n_sites), dtype=bool)
    has_paths   = np.zeros(n_shots, dtype=bool)
    # Median site spacing for the matching tolerance (rejects targets
    # too far from any lab site, e.g. coord-frame mismatches).
    ssub = site_xy[: min(n_sites, 1000)]
    d2s = ((ssub.reshape(-1, 1, 2) - ssub.reshape(1, -1, 2)) ** 2).sum(axis=2)
    d2s[np.arange(d2s.shape[0]), np.arange(d2s.shape[0])] = np.inf
    spacing = max(float(np.median(np.sqrt(d2s.min(axis=1)))), 1.0)
    tol2 = (2.0 * spacing) ** 2

    for k in range(n_shots):
        entry = by_seq.get(int(seq_ids[k]))
        if entry is None:
            continue
        txy = entry.get('target_xy') or []
        valid_xy = [tuple(t) for t in txy
                    if isinstance(t, (list, tuple)) and len(t) == 2
                    and all(isinstance(x, (int, float)) and x is not None
                            for x in t)]
        if not valid_xy:
            continue
        t_arr = np.asarray(valid_xy, dtype=float)
        diffs = site_xy.reshape(1, -1, 2) - t_arr.reshape(-1, 1, 2)
        d2 = (diffs ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        nearest_d2 = d2[np.arange(t_arr.shape[0]), nearest]
        ok = nearest_d2 <= tol2
        if not ok.any():
            continue
        target_mask[k, nearest[ok]] = True
        has_paths[k] = True

    # Only count shots that actually had paths data. (Legacy-style empty
    # entries shouldn't influence FP counts.)
    if not has_paths.any():
        return None
    img1_used = img1[has_paths]
    img2_used = img2[has_paths]
    tmask_used = target_mask[has_paths]
    nontmask_used = ~tmask_used
    target_elig    = tmask_used.sum(axis=0).astype(np.int64)
    nontarget_elig = nontmask_used.sum(axis=0).astype(np.int64)
    tp_hits = (img2_used.astype(bool) & tmask_used).sum(axis=0).astype(np.int64)
    fp_hits = (img2_used.astype(bool) & nontmask_used).sum(axis=0).astype(np.int64)

    with np.errstate(invalid='ignore', divide='ignore'):
        tp_rate = np.where(target_elig > 0, tp_hits / target_elig, np.nan)
        fp_rate = np.where(nontarget_elig > 0, fp_hits / nontarget_elig, np.nan)
        loading = img1_used.mean(axis=0).astype(float)

    def _opt_float_list(arr):
        return [None if (isinstance(x, float) and x != x) else float(x)
                for x in arr.tolist()]

    return {
        'x': site_xy[:, 1].tolist(),   # image-x
        'y': site_xy[:, 0].tolist(),   # image-y
        'loading_rate':       _opt_float_list(loading),
        'survival_mean':      _opt_float_list(tp_rate),
        'tp_rate':            _opt_float_list(tp_rate),
        'fp_rate':            _opt_float_list(fp_rate),
        'tp_elig':            target_elig.tolist(),
        'fp_elig':            nontarget_elig.tolist(),
        'is_target_site':     (target_elig > 0).tolist(),
        'is_nontarget_site':  (nontarget_elig > 0).tolist(),
        'source':             'lab_paths',
        'n_shots_with_paths': int(has_paths.sum()),
    }


def _per_site_from_slm_analysis(slm_an: dict) -> Optional[dict]:
    """Build the lab-side per_site dict from cached slm_analysis.json.

    The SLM's per_site block carries:
      * image_x, image_y      — per-site image-plane coords (3270)
      * tp_rate               — % survival at TARGET sites (NaN for
                                non-target sites where it's undefined)
      * fp_rate               — % spurious detection at NON-TARGET sites
      * loading_rate          — per-site loading (defined for all sites)
      * tp_elig, fp_elig      — eligibility counts per shot

    Produces the same shape `analyze_scan_dir` builds lab-side, but
    with the TP/FP framework instead of plain survival_mean.
    `survival_mean` is set to the per-site TP rate where defined,
    NaN elsewhere — so the dashboard's per-site map naturally shows
    survival only at target sites (the rest grey out). New fields:

      * `is_target_site` — bool per site, True when `tp_elig > 0`
      * `is_nontarget_site` — bool per site, True when `fp_elig > 0`
      * `tp_rate`, `fp_rate` — fractions 0..1 (NaN at non-applicable sites)
      * `tp_elig`, `fp_elig` — per-site eligibility counts (ints)
    """
    ps = slm_an.get('per_site') or {}
    if not ps:
        return None
    def _arr(k, dtype=float):
        v = ps.get(k)
        if not isinstance(v, list):
            return None
        try:
            return np.asarray(v, dtype=dtype)
        except (TypeError, ValueError):
            return None
    image_x = _arr('image_x')
    image_y = _arr('image_y')
    tp_rate = _arr('tp_rate')
    fp_rate = _arr('fp_rate')
    loading = _arr('loading_rate')
    tp_elig = _arr('tp_elig')
    fp_elig = _arr('fp_elig')
    if image_x is None or image_y is None:
        return None
    n = image_x.size
    out = {
        'x': image_x.tolist(),
        'y': image_y.tolist(),
        'loading_rate':  (loading / 100.0).tolist() if loading is not None
                          else [None] * n,
        # `survival_mean` shown on the per-site map = target TP rate
        # (NaN at non-target sites so the dashboard greys those out).
        'survival_mean': (tp_rate / 100.0).tolist() if tp_rate is not None
                          else [None] * n,
        'fp_rate':       (fp_rate / 100.0).tolist() if fp_rate is not None
                          else [None] * n,
        'tp_rate':       (tp_rate / 100.0).tolist() if tp_rate is not None
                          else [None] * n,
        'tp_elig':       tp_elig.astype(int).tolist() if tp_elig is not None
                          else [None] * n,
        'fp_elig':       fp_elig.astype(int).tolist() if fp_elig is not None
                          else [None] * n,
        'is_target_site':    ((tp_elig > 0).tolist()
                               if tp_elig is not None else [None] * n),
        'is_nontarget_site': ((fp_elig > 0).tolist()
                               if fp_elig is not None else [None] * n),
        'source': 'slm_server_cached',
    }
    return out


def _diag_aggregate_from_slm_analysis(slm_an: dict) -> Optional[dict]:
    """Build the same shape `_diag_aggregate` produces but from the
    cached SLM analysis's `diag_series_all` block (used when no
    slm_diag.h5 is on disk for a legacy run)."""
    d = slm_an.get('diag_series_all') or {}
    if not d:
        return None
    def _stat(name, fn):
        arr = d.get(name)
        if not isinstance(arr, list) or not arr:
            return None
        try:
            a = np.asarray([x for x in arr if x is not None], dtype=float)
            if not a.size:
                return None
            return float(fn(a))
        except (TypeError, ValueError):
            return None
    aborted_arr = d.get('aborted') or []
    aborted_count = sum(1 for x in aborted_arr if x)
    two_round_phases: dict[str, int] = {}
    tr = d.get('two_round_phase')
    if isinstance(tr, list):
        for v in tr:
            if v in (None, ''):
                continue
            key = str(v)
            two_round_phases[key] = two_round_phases.get(key, 0) + 1
    return {
        'n_rows':           len(d.get('aborted') or []) or len(d.get('total_ms') or []),
        'mean_total_ms':    _stat('total_ms', np.nanmean),
        'p99_total_ms':     _stat('total_ms', lambda a: np.nanpercentile(a, 99)),
        'mean_n_loaded':    _stat('n_loaded', np.nanmean),
        'mean_n_dropped':   _stat('n_dropped', np.nanmean),
        'aborted_count':    int(aborted_count),
        'two_round_phases': two_round_phases,
        'source':           'slm_analysis_cache',
    }


def _paths_per_shot(diag_path: Path, grid_path: Path) -> dict:
    """Phase 5a: per-shot source→target rearrangement paths.

    Joins ``slm_diag.h5/{loaded_paired,target_paired}`` (per-shot bit
    indices, both vlen-int64 in schema v2 or JSON-encoded in
    ``diag_json`` for schema v1) with ``slm_grid.json``'s ``init_grid``
    and ``target_grid`` coordinate arrays.

    Bit-order invariant (load-bearing — see plan):
      * ``init_grid[k]`` and ``target_grid[k]`` are coords (y, x) of the
        site that MATLAB's bit ``k`` references.
      * If ``gridloc_diag`` is non-null in the sidecar, both grids are
        in camera-frame post-gridLocations-affine ordering. Otherwise
        they're in knm-native (FFT-extraction) ordering.
      * ``loaded_paired[i]`` indexes ``init_grid`` directly.
      * ``target_paired[i]`` indexes ``target_grid`` directly.

    Returns::

      {
        'paths_per_shot': [                  # one entry per ledger row
          {
            'seq_id':         int,
            'loaded_paired':  [int, ...],
            'target_paired':  [int, ...],
            'init_xy':        [[y, x], ...], # from init_grid
            'target_xy':      [[y, x], ...], # from target_grid
            'two_round_phase': 'initial'|'final'|None,
            'two_round_idx':   int,           # -1 if not two-round
          }, ...
        ],
        'paths_frame': 'camera_bitorder' | 'knm_native' | None,
        'paths_n_shots_with_pairing': int,
      }
    """
    out_empty = {
        'paths_per_shot': None,
        'paths_frame': None,
        'paths_n_shots_with_pairing': 0,
    }
    try:
        import h5py  # local import
    except ImportError:
        return out_empty

    # ---- Grid sidecar coordinates (source of truth) -------------------
    init_grid = None
    target_grid = None
    paths_frame = None
    if grid_path.is_file():
        try:
            with open(grid_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as ex:
            logger.warning("_paths_per_shot: grid sidecar read failed: %s", ex)
            payload = None
        if payload:
            # Two payload shapes coexist:
            #   * SLMnet's rearrange_grid_sidecar layout (dict):
            #         {'init_grid': {'coords': [[y,x],...],
            #                         'gridloc_diag': {...}|None, ...}}
            #   * Older / test layout (bare list):
            #         {'init_grid': [[y,x], ...]}
            #   gridloc_applied is only knowable in the dict form;
            #   bare-list payloads default to 'knm_native'.
            def _coords_and_gridloc(node):
                if node is None:
                    return None, None
                if isinstance(node, dict):
                    return node.get('coords'), node.get('gridloc_diag')
                return node, None
            ig, init_gridloc = _coords_and_gridloc(payload.get('init_grid'))
            tg, _ = _coords_and_gridloc(payload.get('target_grid'))
            if ig is not None:
                init_grid = np.asarray(ig, dtype=float)
            if tg is not None:
                target_grid = np.asarray(tg, dtype=float)
            paths_frame = ('camera_bitorder'
                           if init_gridloc is not None
                           else 'knm_native')

    # ---- Read paths from slm_diag.h5 ----------------------------------
    rows: list[dict] = []
    try:
        with h5py.File(diag_path, 'r') as f:
            g = f['/diag'] if '/diag' in f else f
            schema_v = 1
            if 'meta' in f:
                try:
                    schema_v = int(f['meta'].attrs.get('schema_version', 1))
                except (TypeError, ValueError):
                    schema_v = 1

            n_rows = int(g['seq_id'].shape[0]) if 'seq_id' in g else 0
            if n_rows == 0:
                return out_empty

            seq_ids = np.asarray(g['seq_id'][:], dtype=np.int64)

            # Per-shot pairings: prefer surfaced vlen cols (v2), fall
            # through to diag_json parse (v1 back-compat).
            v2_loaded = 'loaded_paired' in g
            v2_target = 'target_paired' in g
            loaded_lists: list[np.ndarray]
            target_lists: list[np.ndarray]
            two_round_phases: list[str]
            two_round_idxs: list[int]

            if v2_loaded and v2_target and schema_v >= 2:
                loaded_lists = [np.asarray(g['loaded_paired'][i],
                                            dtype=np.int64).ravel()
                                for i in range(n_rows)]
                target_lists = [np.asarray(g['target_paired'][i],
                                            dtype=np.int64).ravel()
                                for i in range(n_rows)]
                if 'two_round_phase' in g:
                    raw = g['two_round_phase'][:]
                    two_round_phases = [
                        (v.decode() if isinstance(v, bytes) else str(v))
                        for v in raw]
                else:
                    two_round_phases = [''] * n_rows
                if 'two_round_idx' in g:
                    arr = np.asarray(g['two_round_idx'][:], dtype=float)
                    two_round_idxs = [
                        int(v) if np.isfinite(v) else -1 for v in arr]
                else:
                    two_round_idxs = [-1] * n_rows
            else:
                # v1: parse diag_json. Per-row decode is acceptable for
                # the typical ~hundreds of shots; for thousand-shot scans
                # the lab-side syncer should be upgraded to v2 anyway.
                if 'diag_json' not in g:
                    return out_empty
                raw = g['diag_json'][:]
                loaded_lists = []
                target_lists = []
                two_round_phases = []
                two_round_idxs = []
                for v in raw:
                    s = v.decode() if isinstance(v, bytes) else str(v)
                    try:
                        row = json.loads(s)
                    except (ValueError, TypeError):
                        row = {}
                    d = row.get('diag') or {}
                    loaded_lists.append(np.asarray(
                        d.get('loaded_paired') or [], dtype=np.int64).ravel())
                    target_lists.append(np.asarray(
                        d.get('target_paired') or [], dtype=np.int64).ravel())
                    two_round_phases.append(str(d.get('two_round_phase') or ''))
                    tri = d.get('two_round_idx')
                    two_round_idxs.append(
                        int(tri) if isinstance(tri, (int, float)) else -1)

            # ---- Build per-shot entries ---------------------------------
            n_with_pairing = 0
            for i in range(n_rows):
                lp = loaded_lists[i]
                tp = target_lists[i]
                # Mismatched lengths shouldn't happen by construction but
                # be defensive — clip to the shorter so init_xy/target_xy
                # stay in lockstep.
                if lp.size and tp.size:
                    n_pairs = min(lp.size, tp.size)
                    lp = lp[:n_pairs]
                    tp = tp[:n_pairs]
                    n_with_pairing += 1
                else:
                    lp = np.array([], dtype=np.int64)
                    tp = np.array([], dtype=np.int64)

                init_xy: list = []
                target_xy: list = []
                if lp.size and init_grid is not None and init_grid.size:
                    valid_l = (lp >= 0) & (lp < init_grid.shape[0])
                    for j, k in enumerate(lp):
                        if valid_l[j]:
                            init_xy.append(init_grid[k].tolist())
                        else:
                            init_xy.append([None, None])
                if tp.size and target_grid is not None and target_grid.size:
                    valid_t = (tp >= 0) & (tp < target_grid.shape[0])
                    for j, k in enumerate(tp):
                        if valid_t[j]:
                            target_xy.append(target_grid[k].tolist())
                        else:
                            target_xy.append([None, None])

                phase = two_round_phases[i] if i < len(two_round_phases) else ''
                rows.append({
                    'seq_id':           int(seq_ids[i]),
                    'loaded_paired':    lp.tolist(),
                    'target_paired':    tp.tolist(),
                    'init_xy':          init_xy,
                    'target_xy':        target_xy,
                    'two_round_phase':  phase if phase else None,
                    'two_round_idx':    (two_round_idxs[i]
                                          if i < len(two_round_idxs)
                                          else -1),
                })
    except (OSError, KeyError) as ex:
        logger.warning("_paths_per_shot: %s: %s", diag_path, ex)
        return out_empty

    return {
        'paths_per_shot':              rows,
        'paths_frame':                 paths_frame,
        'paths_n_shots_with_pairing':  n_with_pairing,
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
