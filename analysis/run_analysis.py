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
    per_shot_rate_stats, false_positive_rate,
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

# Lab-side analysis cache (clearly marked, written next to the data). Holds
# the EXPENSIVE, filter-independent results — currently the recomputed
# per-site discrimination (the 1068 double-Gaussian fits). Keyed by the
# recorded shot count so a live/growing run's cache is discarded once more
# shots arrive. Invalidated via the dashboard's "re-analyze" button
# (force_recache) which also clears the focus-metrics cache.
ANALYSIS_CACHE_JSON = 'analysis_cache.json'
ANALYSIS_CACHE_VERSION = 1

# Full default-view (unfiltered, no-recompute) analysis payload, cached to disk
# keyed by actual shot count. Lets a page reload / tab-switch return the whole
# rendered analysis in ~ms instead of re-reading the HDF5 + recomputing. Bumped
# whenever the payload SHAPE changes (so stale caches are ignored, not mis-read).
ANALYSIS_PAYLOAD_JSON = 'analysis_payload.json'
# v2: default view now carries imaging_fidelity (from the throughout-run logged
# infidelities) + rearrange_survival_cap; stale v1 payloads lack those fields.
# v3: invalidates v2 payloads cached with `per_iteration: None`. Older backends
# computed per_iteration only from the (interleaved) `logicals` array, which is
# None for two-array scans -> the field cached as None and stuck (the cache key
# is shot-count, which never changes once the run is done). The cache-write guard
# below now also refuses to persist a payload whose per_iteration failed, so this
# can't recur; the bump clears the already-poisoned v2 caches in one shot.
# v4: seq_specific now also carries the 556 push-out trap-depth panel
# (type='trap_depth') for |mj|=1 survival scans; bump so already-cached mj=1 runs
# recompute and surface it instead of their stale `seq_specific: None`.
ANALYSIS_PAYLOAD_VERSION = 4


def _analysis_cache_path(scan_dir: Path) -> Path:
    return Path(scan_dir) / ANALYSIS_CACHE_JSON


def _payload_cache_path(scan_dir: Path) -> Path:
    return Path(scan_dir) / ANALYSIS_PAYLOAD_JSON


def _probe_actual_shots(scan_dir: Path) -> Optional[int]:
    """Cheap recorded-shot count = seq_ids dataset length (shape only, no data).

    Mirrors ``runs_list._actual_shots`` but kept local to avoid a circular
    import (runs_list imports this module). Returns None on any failure so the
    payload fast-path simply falls through to a full analysis.
    """
    base = Path(scan_dir).name
    h5_path = Path(scan_dir) / f'{base}.h5'
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


def _read_payload_cache(scan_dir: Path) -> Optional[dict]:
    """Return the cached default-view payload wrapper, or None.

    Shape: ``{'_version': int, 'n_shots': int, 'payload': <analysis dict>}``.
    """
    p = _payload_cache_path(scan_dir)
    if not p.is_file():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) \
                and data.get('_version') == ANALYSIS_PAYLOAD_VERSION \
                and isinstance(data.get('payload'), dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_payload_cache(scan_dir: Path, payload: dict, n_shots) -> None:
    """Persist the default-view payload (atomic). No-op on a bad/empty result."""
    if not isinstance(n_shots, int) or n_shots <= 0:
        return
    if payload.get('unpack_error'):
        return   # don't cache a broken analysis
    p = _payload_cache_path(scan_dir)
    try:
        tmp = p.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'_version': ANALYSIS_PAYLOAD_VERSION,
                       'n_shots': int(n_shots), 'payload': payload}, f)
        os.replace(tmp, p)
    except (OSError, TypeError, ValueError) as ex:
        logger.warning('analysis payload cache write failed (%s): %s', p, ex)


def _read_analysis_cache(scan_dir: Path) -> Optional[dict]:
    p = _analysis_cache_path(scan_dir)
    if not p.is_file():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) \
                and data.get('_version') == ANALYSIS_CACHE_VERSION:
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_analysis_cache(scan_dir: Path, data: dict) -> None:
    p = _analysis_cache_path(scan_dir)
    try:
        tmp = p.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except OSError as ex:
        logger.warning('analysis cache write failed (%s): %s', p, ex)


def invalidate_analysis_cache(scan_dir) -> list:
    """Delete the lab-side analysis caches for a scan (analysis_cache.json +
    focus_metrics.json + trap_depth.json). Returns the list of removed paths.
    Used by the dashboard's "re-analyze" button."""
    scan_dir = Path(scan_dir)
    removed = []
    for name in (ANALYSIS_CACHE_JSON, ANALYSIS_PAYLOAD_JSON,
                 FOCUS_METRICS_JSON, TRAP_DEPTH_JSON, AVG_IMAGE_PNG):
        p = scan_dir / name
        try:
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except OSError as ex:
            logger.warning('invalidate cache: could not remove %s: %s', p, ex)
    return removed


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
                 filters: Optional[dict] = None,
                 recompute_infidelity: bool = False,
                 force_recache: bool = False,
                 sync_slm_diag: bool = True) -> dict:
    """Run lab-side analysis on a completed scan.

    Either ``scan_id`` (14-digit YYYYMMDDHHMMSS) or ``scan_dir`` must
    be given. When ``scan_id`` is supplied, the directory is located
    under ``yb_analysis.config.DATA_DIR / YYYYMMDD / data_<scan_id>``.

    ``sync_slm_diag`` (default True) lets the analysis pull a missing
    ``slm_diag.h5`` on demand from the SLM server (see
    ``_maybe_sync_slm_diag``) so rearrangement survival-vs-distance works
    even when the at-scan-end sync never fired.

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
        filters=filters,
        recompute_infidelity=recompute_infidelity,
        force_recache=force_recache,
        sync_slm_diag=sync_slm_diag)


def analyze_scan_dir(scan_dir,
                     *,
                     include_per_site: bool = True,
                     include_diag_aggregate: bool = True,
                     include_per_iteration: bool = True,
                     filters: Optional[dict] = None,
                     recompute_infidelity: bool = False,
                     force_recache: bool = False,
                     sync_slm_diag: bool = True) -> dict:
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
          # Phase 5.5 Track B: computed lab-side from paths_per_shot + logic2
          # (legacy runs fall back to the cached SLM value).
          'survival_vs_distance':            dict | None,
          'survival_vs_distance_skipped_reason': str | None,  # 'lattice_mismatch'
          'survival_vs_distance_per_step':   dict | None,
          'per_shot_extra':                  dict | None,
        }
    """
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        raise RunAnalysisError(f"scan_dir not a directory: {scan_dir}")

    # Fast path: the DEFAULT view (no filter, no recompute) of a scan whose
    # data hasn't grown is served straight from the cached payload on disk, so
    # a page reload / tab-switch returns in ~ms instead of re-reading the HDF5.
    # Self-invalidates when the scan grows (shot-count key mismatch); busted by
    # force_recache (which deletes the file just below).
    _default_view = (filters is None and not recompute_infidelity)
    if _default_view and not force_recache:
        _cached = _read_payload_cache(scan_dir)
        if _cached is not None:
            _cur_shots = _probe_actual_shots(scan_dir)
            if _cur_shots is not None and _cached.get('n_shots') == _cur_shots:
                return _cached['payload']

    # "Re-analyze" button: drop the cached (expensive) results so they're
    # recomputed fresh this call.
    if force_recache:
        invalidate_analysis_cache(scan_dir)

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
    out['scan_description'] = _scan_description(scan)

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

    # Unfiltered sweep values kept for seq-specific focus metrics (which
    # always show the full defocus curve, regardless of any active filter).
    scan_params_full = np.array(scan_params, copy=True)

    n_sites, n_params, max_reps = (
        logic1.shape if logic1.ndim == 3 else (0, 0, 0))
    # Keep UNFILTERED references for the global (filter-independent)
    # headline block computed near the end of this function. The filter
    # block below rebinds logic1/logic2/scan_params/reps_per_param.
    logic1_full = logic1
    logic2_full = logic2
    reps_per_param_full = reps_per_param
    n_params_full = int(n_params)
    # Actual recorded shots = number of saved sequences (seq_ids), NOT the
    # scheduled nParams*maxReps (a scan can be aborted early or padded).
    try:
        n_shots_actual = int(len(np.asarray(seq_ids).ravel())) \
            if seq_ids is not None else int(n_params * max_reps)
    except (TypeError, ValueError):
        n_shots_actual = int(n_params * max_reps)
    out['n_params'] = int(n_params)
    out['n_params_global']   = n_params_full
    out['n_shots']           = n_shots_actual
    # "supposed to do" = the run PLAN from the config sidecar (n_shots_planned /
    # len(Params), honoring an explicit rep), NOT n_params * observed-max_reps --
    # the observed depth grows as data accumulates and shrinks on an abort, so it
    # never reports a stable plan. Fall back to the legacy estimate for old sidecars.
    out['n_shots_scheduled'] = _planned_shots(scan, fallback=int(n_params_full * max_reps))
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
    # Every parameter that defines this run — swept axes (as value lists) +
    # fixed SetParams/DefaultParams — for the "Details" panel.
    out['run_parameters'] = _run_parameters(scan, out['sweep_all'])

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
            # n_shots stays GLOBAL/actual (header is filter-independent);
            # expose the filtered scheduled count separately for any card
            # that wants it.
            out['n_shots_filtered'] = int(n_params * max_reps)
            out['sweep']     = _build_sweep(scan, scan_params,
                                            mat_path=bundle.get('mat_path'))
        except (IndexError, ValueError) as ex:
            logger.warning('filter slicing failed: %s', ex)
    out['filter_active'] = bool(param_mask is not None)

    # ---- Summary: survival / loading / loss per param ------------------
    out['summary'] = _summary_stats(logic1, logic2, reps_per_param)

    # ---- Seq-specific: CALIBRATION-FREE focus metrics vs swept param ----
    # For loading-optimisation sweeps (e.g. LoadingDefocusScan) measure spot
    # focus straight from the raw images: per defocus point we average THIS
    # run's frames, detect the array spots as bright local maxima (a per-
    # defocus site map built from the data itself -- no grid, no thresholds,
    # no calibration), and average per-spot shape over the detected spots
    # (so spot count doesn't confound it). Cached to focus_metrics.json;
    # returns None for non-sweep / non-single-image scans.
    try:
        out['seq_specific'] = _focus_metrics_from_images(
            scan_dir, scan, scan_params_full, seq_ids,
            mat_path=bundle.get('mat_path'))
    except Exception as ex:
        logger.warning('focus metrics failed: %s', ex)
        out['seq_specific'] = None

    # ---- Seq-specific (2): 556 push-out trap-depth (|mj|=1 light shift) ------
    # For ANY 556 push-out SURVIVAL scan -- NumImages>=2 sweeping the push-out
    # green frequency, detected by the swept axis + the trap-shifted line, NOT by
    # the scan being named Spectrum556Scan -- compute the per-site trap-depth
    # histogram + CV (the trap-depth-feedback uniformity metric). Single-image
    # focus metrics and this survival measurement are mutually exclusive, so it
    # fills the same seq_specific slot when focus metrics didn't apply.
    if out.get('seq_specific') is None:
        try:
            out['seq_specific'] = _trap_depth_from_pushout(
                scan_dir, scan, scan_params_full, logic1_full, logic2_full,
                seq_ids, sweep_all=out.get('sweep_all'))
        except Exception as ex:
            logger.warning('trap-depth seq-specific failed: %s', ex)

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
            # Per-site survival/FP pair logic1[s] & logic2[s] site-for-site, so
            # they're only defined when img1 and img2 share a grid. For a
            # cross-grid rearrangement run (init pattern != target pattern) the
            # target-aware per-site map (_per_site_from_lab_paths) supplies the
            # survival map instead; here we keep the loading map (img1-only).
            if (logic2 is not None and logic2.size
                    and not _cross_grid_logicals(logic1, logic2)):
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

    # ---- SLM diag aggregate (from synced sidecar) ----------------------
    # The at-scan-end sync only fires when the NEXT scan evicts this run's
    # DataManager, so the most-recent run — or any run before a run_monitor
    # restart — never gets its slm_diag.h5. Pull it on demand here (the SLM
    # server keeps the ledger) so rearrange survival-vs-distance just works.
    _maybe_sync_slm_diag(scan_dir, out.get('scan_name'), enabled=sync_slm_diag,
                         expected_shots=out.get('n_shots'))
    diag_path = scan_dir / DIAG_H5
    if include_diag_aggregate and diag_path.is_file():
        out['diag_aggregate'] = _diag_aggregate(diag_path)
    else:
        out['diag_aggregate'] = None

    # Augment the run-parameters list with clean rearrangement settings from
    # the diag (protocol, etc.) that SetParams stores as undecodable string
    # objects. Dedup against names already present.
    have_names = {p['name'] for p in out.get('run_parameters', [])}
    out['run_parameters'].extend(
        _run_params_from_diag(diag_path, have_names))

    # ---- Code snapshot sidecar pointer ---------------------------------
    out['code'] = _code_info(scan_dir / CODE_JSON)
    # The SLM model that ACTUALLY ran (the checkpoint the server snapshotted into
    # slm_code.json) — the authoritative companion to the *requested* model in
    # descriptor['runp'] (warmup_kwargs.model_filename). Surface it in Details so
    # a rearrangement run records which model produced its data. Dedup by name.
    _slm_model = (out.get('code') or {}).get('model_filename')
    if _slm_model and not any(p['name'] == 'slm_model'
                              for p in out['run_parameters']):
        out['run_parameters'].append(
            {'name': 'slm_model', 'value': str(_slm_model), 'group': 'slm'})

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

    # ---- Per-iteration arrays (TIME-ORDER, one per shot) -------------
    # IMPORTANT: walks raw interleaved bundle.logicals using seq_ids so
    # the X axis is true shot order, NOT the (param × rep) unpacked
    # order. Without this, the swept-param trace looks like a slow
    # staircase (consecutive iterations share a param) instead of the
    # scrambled order the scan actually executed in. Computed here (after
    # paths_info + diag_path) so it can compute the rearrangement-correct
    # FP (target sites excluded) and attach per-shot numeric diag series.
    # Tracks whether an OPTIONAL sub-computation failed recoverably. A failed
    # per_iteration must NOT be persisted to the payload cache: the cache key is
    # shot-count, so a one-time failure (transient, or older code) would stick
    # forever and the dashboard would serve `per_iteration: None` on every later
    # view (the bug this guards against). See _write_payload_cache.
    cache_safe = True
    if include_per_iteration:
        try:
            pit = _per_iteration_time_order(
                scan, bundle, seq_ids, param_mask, filters,
                paths_info=paths_info,
                diag_path=diag_path if diag_path.is_file() else None)
            out['per_iteration'] = pit
            # A None return when logicals DID unpack (n_shots > 0) means the
            # per-shot walk silently failed (e.g. an unhandled data layout) --
            # treat it like an exception so the bad payload isn't cached.
            if pit is None and out.get('unpack_error') is None \
                    and (out.get('n_shots') or 0) > 0:
                cache_safe = False
        except Exception as ex:
            logger.warning('per_iteration_time_order failed: %s', ex)
            out['per_iteration'] = None
            cache_safe = False
    else:
        # Intentionally skipped (caller asked for no per_iteration); still cache.
        out['per_iteration'] = None

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
        # Target-aware FP curve + overall (so the sweep plot + headline can
        # show FP at non-target sites alongside TP).
        if target_aware.get('per_param_fp') is not None:
            out['summary']['fp_mean'] = target_aware['per_param_fp']
            out['summary']['fp_sem']  = target_aware.get('per_param_fp_sem')
            out['summary']['fp_overall'] = target_aware.get('fp_overall')
            out['summary']['fp_source'] = 'rearrange'
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
    # Target data is present when an entry carries per-shot target bit indices
    # (`target_paired`, from the diag — the sidecar-free primary) OR legacy
    # sidecar coords (`target_xy`). Either triggers the lab-side target-aware
    # per-site map; the grid sidecar is no longer required.
    have_real_paths = any(
        (isinstance(e, dict) and (e.get('target_site_indices')
                                  or e.get('target_paired') or e.get('target_xy')))
        for e in paths_list)
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

    # ---- survival_vs_distance — computed LAB-SIDE (Phase 5.5 Track B) ---
    # Bin rearrangement pairs by transit distance and measure survival at
    # the target from the lab's OWN logic2. This replaces the transitional
    # copy-through of the SLM server's value: per the operator, everything
    # is computed lab-side now (only FFT extraction stays SLM-side). For
    # legacy runs without paths_per_shot, fall back to the cached SLM value
    # so those panels don't go blank.
    seq_nsteps = _seq_to_nsteps_map(scan, scan_params_full,
                                    out.get('sweep_all'), seq_ids)
    svd = _survival_vs_distance(paths_info, bundle, scan,
                                scan_id=out.get('scan_id'),
                                seq_nsteps=seq_nsteps)
    out['survival_vs_distance_skipped_reason'] = None
    if svd is not None and 'skipped_reason' in svd:
        out['survival_vs_distance'] = None
        out['survival_vs_distance_skipped_reason'] = svd['skipped_reason']
    elif svd is not None:
        out['survival_vs_distance'] = svd
    elif slm_an is not None:
        # Legacy run (no lab-side paths) — surface the cached SLM curve.
        out['survival_vs_distance'] = slm_an.get('survival_vs_distance')
    else:
        out['survival_vs_distance'] = None

    # ---- Other upstream-compat fields (still SLM-computed) -------------
    # These ride through slm_diag.h5 / the cached slm_analysis.json. They
    # are not part of Track B; surface the cached value when present.
    if slm_an is not None:
        out['survival_vs_distance_per_step'] = slm_an.get(
            'survival_vs_distance_per_step')
        out['per_shot_extra'] = slm_an.get('per_shot_extra')
        out['round1'] = slm_an.get('round1')
    else:
        out['round1'] = None
        out['survival_vs_distance_per_step'] = None
        out['per_shot_extra'] = None

    # ---- Discrimination infidelity (per-site, filter-independent) ------
    # Default: the scan's own scan-start initInfidelities (matches the
    # detection convention). When recompute_infidelity is set, refit the
    # double-Gaussian from THIS run's stored intensities so the metric
    # reflects the actual run, not the scan-start calibration (the dashboard's
    # "recompute from this run" button — non-destructive).
    # Thresholds the run ACTUALLY used, reconstructed per-shot from the
    # per-pattern update log (seeded with scan-start initThresholds). Drives
    # the used-threshold imaging fidelity in BOTH views: the cheap default
    # reads the logged infidelities, the recompute evaluates these per-segment
    # thresholds against this run's freshly-fit Gaussians.
    thr_timeline = _run_threshold_timeline(scan, scan_dir, out['n_shots'])
    paths_for_cap = paths_info.get('paths_per_shot')

    disc = None
    imaging_fid = None
    rearr_cap = None
    if recompute_infidelity:
        # The recompute (1068 double-Gaussian fits, ~5 s) is whole-scan and
        # filter-INDEPENDENT, so it's cached to disk keyed by shot count.
        # Filtering / reloading reuses the cache instead of refitting; a
        # live run's cache is discarded once n_shots grows (key mismatch). ONE
        # fit pass yields BOTH the recompute discrimination (optimal cut) and
        # the imaging fidelity at the throughout-run thresholds.
        cache = _read_analysis_cache(scan_dir)
        if cache and cache.get('n_shots') == out['n_shots']:
            disc = cache.get('discrimination_recomputed')
            imaging_fid = cache.get('imaging_fidelity')
            rearr_cap = cache.get('rearrange_survival_cap')
        if disc is None or imaging_fid is None:
            segs_for_fit = ([(s.get('weight', 0), s.get('thresholds'))
                             for s in thr_timeline] if thr_timeline else None)
            disc2, imaging2 = _recompute_run_metrics(bundle, scan, segs_for_fit)
            disc = disc or disc2
            imaging_fid = imaging_fid or imaging2
            _update_analysis_cache(scan_dir, out['n_shots'],
                                   discrimination_recomputed=disc,
                                   imaging_fidelity=imaging_fid)
        # Loss #2: max rearrangement survival cap from source-site detection
        # confidence (needs its own img1-only fit, so it rides the recompute).
        if rearr_cap is None and paths_for_cap:
            try:
                rearr_cap = _rearrange_survival_cap(paths_for_cap, bundle, scan)
            except Exception as ex:  # noqa: BLE001
                logger.warning('rearrange survival cap failed: %s', ex)
                rearr_cap = None
            if rearr_cap is not None:
                _update_analysis_cache(scan_dir, out['n_shots'],
                                       rearrange_survival_cap=rearr_cap)
    else:
        # Default view: cheap throughout-run imaging fidelity from the logged
        # infidelities (no refit). Surface a cap cached by a prior recompute.
        imaging_fid = _imaging_fidelity_from_logged_timeline(thr_timeline)
        cache = _read_analysis_cache(scan_dir)
        if cache and cache.get('n_shots') == out['n_shots']:
            rearr_cap = cache.get('rearrange_survival_cap')
    if disc is None:
        disc = _discrimination_info(scan)
    out['discrimination'] = disc
    out['discrimination_global'] = disc   # alias: infidelity never filters
    # Imaging fidelity at the thresholds used THROUGHOUT the run — default view
    # from the logged infidelities, recompute from this run's refit.
    out['imaging_fidelity'] = imaging_fid
    # Loss #2 — max rearrangement survival cap (recompute-loaded / cached).
    out['rearrange_survival_cap'] = rearr_cap
    # Attach per-site infidelity onto the per_site panel payload so the
    # dashboard can render an infidelity map alongside loading/survival/FP.
    # Done last so it survives the per_site overrides above; length-guarded
    # against the active per_site site count.
    if disc is not None and isinstance(out.get('per_site'), dict):
        ps_inf = disc['per_site']
        ps_ = out['per_site']
        dx, dy = disc.get('x'), disc.get('y')
        main_n = len(ps_['x']) if ps_.get('x') else None
        if main_n is not None and len(ps_inf) == main_n:
            # Infidelity is on the main per-site grid.
            ps_['infidelity'] = ps_inf
        elif dx and dy and len(ps_inf) == len(dx):
            # Different grid than the main map (cross-grid run: infidelity is on
            # the INIT detection grid, the map is the TARGET grid) — carry its
            # own coords so the dashboard renders it on the init grid.
            ps_['infidelity'] = ps_inf
            ps_['infid_x'] = dx
            ps_['infid_y'] = dy
        elif main_n is None and dx:
            # per_site had no coords (legacy) — adopt disc's grid.
            ps_['infidelity'] = ps_inf
            ps_['x'] = dx
            ps_['y'] = dy

    # ---- Threshold provenance + summary (header) -----------------------
    if disc is not None and disc.get('source') == 'recomputed_from_run' \
            and disc.get('thresholds'):
        t = np.asarray(disc['thresholds'], dtype=float).ravel()
        with np.errstate(invalid='ignore'):
            out['thresholds_info'] = {
                'source': 'recomputed_from_run',
                'source_note': ('per-site thresholds refit from THIS run\'s '
                                'intensities (non-destructive; for analysis '
                                'only — not written back to disk).'),
                'n':      int(t.size),
                'mean':   float(np.nanmean(t)),
                'median': float(np.nanmedian(t)),
                'min':    float(np.nanmin(t)),
                'max':    float(np.nanmax(t)),
                'mean_infidelity': disc.get('mean_infidelity'),
            }
    else:
        out['thresholds_info'] = _thresholds_info(scan)
    # Loading-pattern name(s) + calibration age (staleness marker) shown
    # alongside the threshold source.
    if out.get('thresholds_info') is not None:
        out['thresholds_info']['patterns'] = _loading_pattern_names(scan)
        out['thresholds_info'].update(_calibration_age_info(scan, scan_dir))

    # ---- Avg intensity histogram + threshold marker(s) -----------------
    # Pooled per-site intensity histogram (data-quality view, mirrors live).
    # Mark the median detection threshold ON it: the scan-start value always,
    # and — when discrimination was recomputed from this run — the this-run
    # value too (so the operator sees both cuts). NOTE: the infidelity metric
    # is computed at the OPTIMAL Gaussian cut, so it does NOT depend on which
    # threshold detected the logicals; these markers are reference lines only.
    ihist = _intensity_hist(bundle)
    if ihist is not None:
        markers = []
        ss = _thresholds_info(scan)   # scan-start, independent of recompute
        if ss and ss.get('median') is not None:
            markers.append({'label': 'scan-start threshold',
                            'value': ss['median'], 'source': 'scan_init'})
        if disc is not None and disc.get('source') == 'recomputed_from_run' \
                and disc.get('thresholds'):
            t2 = np.asarray(disc['thresholds'], dtype=float).ravel()
            if t2.size and np.any(np.isfinite(t2)):
                markers.append({'label': 'this-run threshold',
                                'value': float(np.nanmedian(t2)),
                                'source': 'recomputed_from_run'})
        ihist['threshold_markers'] = markers
    out['intensity_hist'] = ihist

    # ---- GLOBAL (filter-independent) headline block --------------------
    # The dashboard's top stat bar must NOT move when a filter is applied.
    # Recompute survival / loss / loading over the UNFILTERED logicals and,
    # for rearrangement runs, the whole-scan target-aware (TP) curve.
    out['summary_global'] = _summary_stats(
        logic1_full, logic2_full, reps_per_param_full)
    try:
        ta_global = _target_aware_survival(
            slm_an, out, scan, scan_params_full,
            mat_path=bundle.get('mat_path'), bundle=bundle,
            paths_per_shot=paths_info.get('paths_per_shot'),
            reps_per_param=reps_per_param_full, param_mask=None)
    except Exception as ex:
        logger.warning('global target_aware failed: %s', ex)
        ta_global = None
    out['target_aware_global'] = ta_global

    # Filtered TP: target survival with outlier bad-fidelity target spots
    # dropped (so transport survival isn't dragged down by a few mis-detected
    # targets). Cheap — uses the throughout-run logged per-site infidelities
    # (collapsing to scan-start initInfidelities when nothing was logged), so it
    # computes WITHOUT a recompute and shows for old runs on re-analyze.
    out['target_aware_filtered'] = None
    try:
        per_site_infid = _throughout_run_per_site_infidelity(thr_timeline)
        if per_site_infid is not None and paths_info.get('paths_per_shot'):
            infid_src = ('throughout_run'
                         if any(s.get('source') != 'scan_init'
                                and s.get('weight', 0) > 0
                                and s.get('infidelities') is not None
                                for s in thr_timeline)
                         else 'scan_start')
            out['target_aware_filtered'] = _filtered_target_aware(
                paths_info.get('paths_per_shot'), bundle, scan,
                scan_params_full, reps_per_param=reps_per_param_full,
                per_site_infid=per_site_infid, infid_source=infid_src)
    except Exception as ex:
        logger.warning('filtered target_aware failed: %s', ex)
        out['target_aware_filtered'] = None
    if ta_global and ta_global.get('per_param_mean') is not None:
        out['summary_global']['survival_mean_per_site'] = \
            out['summary_global']['survival_mean']
        out['summary_global']['survival_mean'] = ta_global['per_param_mean']
        out['summary_global']['survival_sem']  = ta_global['per_param_sem']
        out['summary_global']['survival_source'] = ta_global['source']
        if ta_global.get('per_param_fp') is not None:
            out['summary_global']['fp_mean'] = ta_global['per_param_fp']
            out['summary_global']['fp_sem']  = ta_global.get('per_param_fp_sem')
            out['summary_global']['fp_overall'] = ta_global.get('fp_overall')
            out['summary_global']['fp_source'] = 'rearrange'
    elif ta_global and ta_global.get('overall_mean') is not None:
        out['summary_global']['survival_overall_target_aware'] = \
            ta_global['overall_mean']

    # Align the per-param summary curves with the (sorted) sweep['values'] the
    # dashboard plots them against. The unpacked param axis is in descriptor
    # order, which inverts the curve for non-ascending sweeps (e.g. precompute
    # [True, False]) -- see _reorder_summary_to_sweep. summary uses the FILTERED
    # scan_params; summary_global uses the full set.
    _reorder_summary_to_sweep(out.get('summary'), scan_params)
    _reorder_summary_to_sweep(out.get('summary_global'), scan_params_full)

    result = to_jsonable(out)
    # Persist the default-view payload so the next reload/tab-switch is instant.
    # Skip the write when a recoverable sub-computation failed (cache_safe) so a
    # one-time failure can't get baked into the shot-count-keyed cache forever.
    if _default_view and cache_safe:
        _write_payload_cache(scan_dir, result, out.get('n_shots'))
    return result


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


def _maybe_sync_slm_diag(scan_dir: Path, scan_name, *, enabled: bool = True,
                         expected_shots: Optional[int] = None):
    """Best-effort on-demand pull of ``slm_diag.h5`` — creating it when missing
    AND **completing a partial** one.

    The at-scan-end sync (``DataManager._schedule_slm_sync``) only fires when
    the NEXT scan evicts this run's DataManager — so the most-recent run, and
    ANY run completed before a ``run_monitor`` restart, never gets its
    ``slm_diag.h5``. Worse, a sync that ran MID-RUN (e.g. an analysis view
    while the scan was live) leaves a PARTIAL file with only the early shots;
    the old "skip if the file exists" guard then froze it there, so the
    per-shot TP only covered the first few shots. We now also resume the
    incremental sync when the local row count is short of ``expected_shots``
    (the SLM keeps the full ledger; ``sync_scan`` resumes via since_seq_id).

    Restores everything that rides the diag: the per-shot rearrange paths,
    **target-aware survival/TP**, and survival-vs-distance.

    Gated to rearrangement-like runs (name contains 'rearrang', or an
    ``slm_code.json`` exists). Swallowed on any error / SLM offline.
    """
    if not enabled:
        return
    scan_dir = Path(scan_dir)
    diag_path = scan_dir / DIAG_H5
    # If a complete-enough file is already present, skip (avoid a round-trip).
    if diag_path.is_file():
        try:
            import h5py
            with h5py.File(diag_path, 'r') as f:
                g = f['/diag'] if '/diag' in f else f
                local_rows = int(g['seq_id'].shape[0]) if 'seq_id' in g else 0
        except (OSError, KeyError):
            local_rows = 0
        # No expected count → trust the existing file. Otherwise only re-sync
        # when it's clearly short (a partial mid-run sync).
        if expected_shots is None or local_rows >= int(expected_shots):
            return
    name = (scan_name or '').lower()
    if 'rearrang' not in name and not (scan_dir / CODE_JSON).is_file():
        return
    scan_id = _scan_id_from_dir(scan_dir)
    if not scan_id:
        return
    try:
        from yb_analysis.slm_sync import sync_scan
        status = sync_scan(scan_id, scan_dir)   # resumes via since_seq_id
        if status.get('rows_written'):
            logger.info('on-demand slm_sync %s: pulled %d diag rows',
                        scan_id, status.get('rows_written'))
        else:
            logger.info('on-demand slm_sync %s: %s', scan_id,
                        status.get('reason'))
    except Exception as ex:   # never let a sync hiccup break analysis
        logger.info('on-demand slm_sync %s skipped: %s', scan_id, ex)


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


def _planned_shots(scan: dict, fallback: Optional[int] = None) -> Optional[int]:
    """Total shots the scan was SET to run -- the "supposed to do" count.

    Read from the config sidecar so it reflects the actual run PLAN, not the
    recorded-data depth (``logic1.shape[2] = max_reps`` is what actually ran and
    grows as data accumulates / shrinks on an abort -- never the plan). Prefers,
    in order:

      1. ``n_shots_planned`` -- pyctrl writes it explicitly (= len of the realized
         run order = nseqs * StackNum, StackNum honoring an explicit ``rep``).
      2. ``len(Params)`` -- the realized run order itself; present for pyctrl AND
         MATLAB sidecars, so this covers runs saved before n_shots_planned existed.
      3. ``fallback`` -- the caller's legacy estimate (e.g. nParams * observed reps).

    Distinct from ACTUAL recovered shots (``len(seq_ids)``); pair the two as
    "did / supposed-to-do"."""
    try:
        v = scan.get('n_shots_planned')
        if v is not None:
            return int(np.asarray(v).ravel()[0])
    except Exception:
        pass
    try:
        params = scan.get('Params')
        if params is not None:
            return int(np.asarray(params).shape[0])
    except Exception:
        pass
    return fallback


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
    # n_dims = number of genuinely swept axes (a single-point "sweep"
    # reads as 0-D in the dashboard). cols may be padded beyond dims when
    # extract_scan_dims_h5 over-reports, so key off dims.
    n_dims = len([d for d in dims if d and d > 1])
    return {'cols': cols, 'values': values, 'dims': dims, 'n_dims': n_dims}


def _reorder_summary_to_sweep(summary, scan_params) -> None:
    """Reorder per-param summary arrays to ASCENDING param-value order, IN PLACE.

    ``_build_sweep`` sorts the x-axis (``np.unique(scan_params)``), but the
    unpacked param axis -- and therefore every per-param array in ``summary``
    (survival_mean, loading_rate, fp_mean, the SEM/STD/n families, …) -- is in
    scan-DESCRIPTOR (param-index) order. When the sweep wasn't defined ascending
    (e.g. a boolean ``precompute`` sweep that unpacks as ``[True, False]`` =
    ``[1.0, 0.0]``), the dashboard pairs each y with the WRONG x, inverting the
    curve vs the live view (which sorts via ``compute_scan_curve``). Sorting the
    summary arrays here aligns them with ``sweep['values']``. No-op when the
    sweep is already ascending (the common case) or not 1-D.
    """
    if not isinstance(summary, dict) or scan_params is None:
        return
    sp = np.asarray(scan_params)
    if sp.ndim != 1 or sp.size < 2:
        return
    perm = np.argsort(sp, kind='stable')
    if np.array_equal(perm, np.arange(sp.size)):
        return   # already ascending -> nothing to do
    n = int(sp.size)
    for key, val in list(summary.items()):
        if isinstance(val, list) and len(val) == n:
            summary[key] = [val[i] for i in perm]


def _cross_grid_logicals(logic1, logic2) -> bool:
    """True when img1 and img2 were detected on DIFFERENT grids.

    A rearrangement scan whose initial loading pattern and final target
    pattern differ (e.g. 47x47 loading -> 33x33 target) detects img1 on the
    loading grid and img2 on the target grid, so the unpacked logicals have
    different site counts. In that case the per-site survival / loss / FP
    metrics -- which pair ``logic1[s] & logic2[s]`` site-for-site -- are
    undefined (site s of img1 is a different physical trap than site s of
    img2). The meaningful survival is the target-aware TP (``target_paired``
    -> img2) computed in ``_target_aware_from_lab_paths``.
    """
    return (logic2 is not None
            and getattr(logic1, 'ndim', 0) == 3
            and getattr(logic2, 'ndim', 0) == 3
            and logic1.shape[0] != logic2.shape[0])


def _summary_stats(logic1: np.ndarray,
                   logic2: Optional[np.ndarray],
                   reps_per_param: Optional[np.ndarray] = None) -> dict:
    """Per-param survival / loss / loading curves.

    Each rate carries TWO error families so the dashboard's error-bar
    dropdown can switch between them without a refetch:
      * ``*_sem`` — the per-SITE binomial SEM (existing convention from
        ``prob11`` / ``loading_rate``: count every atom).
      * ``*_sem_pershot`` / ``*_std_pershot`` — computed across SHOTS
        (each shot is one sample of the array-averaged rate; this is the
        dashboard default). ``*_n_shots`` is the eligible-shot count.
    """
    # Cross-grid rearrangement (init pattern != target pattern): img1 and img2
    # live on DIFFERENT detection grids, so per-site survival/loss/FP are
    # undefined (and would crash on the shape mismatch). Drop to loading-only
    # here; the target-aware TP override (in analyze_scan_dir) supplies the
    # meaningful survival curve.
    if _cross_grid_logicals(logic1, logic2):
        logic2 = None
    if logic1.size == 0 or logic2 is None or logic2.size == 0:
        # Loading-only or empty scan: still report what we can.
        try:
            lr_mean, lr_sem = (loading_rate(logic1, reps_per_param)
                               if logic1.size else (np.array([]),
                                                    np.array([])))
        except Exception:
            lr_mean, lr_sem = np.array([]), np.array([])
        n = lr_mean.size
        empty = [float('nan')] * n
        out = {
            'survival_mean':     empty,
            'survival_sem':      empty,
            'loading_rate':      lr_mean.tolist(),
            'loading_rate_sem':  lr_sem.tolist(),
            'loss_mean':         empty,
            'loss_sem':          empty,
        }
    else:
        sr_mean, sr_sem = prob11(logic1, logic2)
        ls_mean, ls_sem = prob10(logic1, logic2)
        lr_mean, lr_sem = loading_rate(logic1, reps_per_param)
        # Baseline (all-empty) false-positive rate. For rearrangement runs the
        # caller OVERRIDES fp_mean/fp_sem with the target-aware version (which
        # excludes target sites); this is what non-rearrange 2-image scans use.
        try:
            fp_mean, fp_sem = false_positive_rate(logic1, logic2)
        except Exception as ex:
            logger.warning('_summary_stats: false_positive_rate failed: %s', ex)
            fp_mean = fp_sem = None
        out = {
            'survival_mean':     sr_mean.tolist(),
            'survival_sem':      sr_sem.tolist(),
            'loading_rate':      lr_mean.tolist(),
            'loading_rate_sem':  lr_sem.tolist(),
            'loss_mean':         ls_mean.tolist(),
            'loss_sem':          ls_sem.tolist(),
            'fp_mean':           fp_mean.tolist() if fp_mean is not None else None,
            'fp_sem':            fp_sem.tolist() if fp_sem is not None else None,
            'fp_source':         'all_empty',
        }
    # Per-shot error families (default for the dashboard).
    try:
        ps = per_shot_rate_stats(logic1, logic2, reps_per_param)
    except Exception as ex:
        logger.warning('_summary_stats: per_shot_rate_stats failed: %s', ex)
        ps = {}
    out['survival_std_pershot']   = ps.get('survival_std_pershot')
    out['survival_sem_pershot']   = ps.get('survival_sem_pershot')
    out['survival_n_shots']       = ps.get('survival_n_shots')
    out['loading_std_pershot']    = ps.get('loading_std_pershot')
    out['loading_sem_pershot']    = ps.get('loading_sem_pershot')
    out['loading_n_shots']        = ps.get('loading_n_shots')
    return out


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
                'safe_run_id': None, 'n_files': 0, 'model_filename': None}
    try:
        with open(code_json, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning("_code_info: failed to read %s: %s", code_json, ex)
        return {'present': False, 'manifest_path': None,
                'safe_run_id': None, 'n_files': 0, 'model_filename': None}
    manifest = payload.get('manifest') or {}
    files = manifest.get('files') or []
    return {
        'present': True,
        'manifest_path': payload.get('manifest_path'),
        'safe_run_id':   payload.get('safe_run_id'),
        'n_files':       len(files),
        # The SLM model checkpoint the server actually loaded for this run —
        # the authoritative record of which model produced the data.
        'model_filename': manifest.get('model_filename'),
    }


AVG_IMAGE_PNG = 'avg_image.png'
FOCUS_METRICS_JSON = 'focus_metrics.json'   # cached calibration-free focus curve
TRAP_DEPTH_JSON = 'trap_depth.json'         # cached |mj|=1 trap-depth (CV + histogram)
_TRAP_DEPTH_CACHE_VERSION = 1
# Cap on frames averaged for avg_image.png. /imgs is gzip-compressed one frame
# per chunk (~25 ms/frame to read+decompress on CPU — a GPU can't help), so the
# only way to stay fast is to read fewer frames. A mean image is statistically
# identical from a random sample of ~250 (one-time compute ~10 s, then cached).
# Full image resolution is preserved (no spatial downsample).
AVG_IMAGE_MAX_FRAMES = 250


def _scan_data_h5(scan_dir: Path) -> Optional[Path]:
    """Return the scan's data_*.h5 path (the one with /imgs, /logicals)."""
    cands = sorted(scan_dir.glob('data_*.h5'))
    return cands[0] if cands else None


def _avg_image_info(scan_dir: Path, scan: dict) -> Optional[dict]:
    """Report whether an averaged camera image is available / computable.

    Shown for ANY scan with an ``/imgs`` dataset — the averaged frame is an
    informative "where did atoms appear / array health" view regardless of
    NumImages. For multi-image scans this averages ALL frames (e.g. img1+img2
    interleaved); ``num_images`` is reported so the dashboard can label it.

    Returns ``None`` only when there's no usable ``/imgs``. Otherwise::

        {
          'available': bool,         # PNG already cached on disk
          'computable': bool,        # we have imgs but no PNG yet
          'png_path': str | None,    # absolute path when available
          'n_shots': int,            # how many frames are averaged (total)
          'num_images': int,         # frames per shot
          'image_shape': [H, W],     # image dims
        }
    """
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
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
    n_frames, h, w = shape
    # We average only the FIRST image of each shot (frames 0, num_images, ...).
    n_first = (int(n_frames) + num_images - 1) // num_images
    n_avg = min(n_first, AVG_IMAGE_MAX_FRAMES)
    png_path = scan_dir / AVG_IMAGE_PNG
    return {
        'available':   png_path.is_file(),
        'computable':  not png_path.is_file(),
        'png_path':    str(png_path) if png_path.is_file() else None,
        'n_shots':     n_first,            # number of first-images (= shots)
        'n_avg':       int(n_avg),         # how many actually averaged (sampled)
        'sampled':     bool(n_avg < n_first),
        'num_images':  int(num_images),
        'image_shape': [int(h), int(w)],
    }


def ensure_avg_image_png(scan_dir, *, batch_size: int = 16,
                         max_frames: int = AVG_IMAGE_MAX_FRAMES) -> Optional[Path]:
    """Compute the mean of the FIRST image of each shot, write avg_image.png.

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
    # NumImages → average only the FIRST image of each shot (frames
    # 0, num_images, 2*num_images, ...). Read from the sibling config
    # (.json for pyctrl, .mat for MATLAB).
    num_images = 1
    try:
        from yb_analysis.io.mat_reader import load_scan_config
        cfg = load_scan_config(str(scan_dir / (h5_path.stem + '.mat'))) or {}
        num_images = int(np.asarray(cfg.get('NumImages', 1)).flat[0]) or 1
    except Exception:
        num_images = 1
    try:
        with h5py.File(h5_path, 'r') as f:
            d = f.get('imgs')
            if d is None or d.ndim != 3:
                return None
            n_frames = d.shape[0]
            if n_frames == 0:
                return None
            # First-image indices, then a random sub-sample to the frame cap
            # so the one-time compute stays fast (a mean image is identical
            # from a sample; reading every gzip frame is the cost). Seeded →
            # the cache is reproducible. Full resolution preserved.
            first_idx = np.arange(0, n_frames, max(1, num_images))
            if max_frames and first_idx.size > max_frames:
                rng = np.random.default_rng(0)
                pick = np.sort(rng.choice(first_idx.size, size=max_frames,
                                          replace=False))
                first_idx = first_idx[pick]
            acc = np.zeros(d.shape[1:], dtype=np.float64)
            for i in first_idx:
                acc += d[int(i)]
            mean_img = acc / float(first_idx.size)
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


def _discrimination_info(scan: dict) -> Optional[dict]:
    """Per-site discrimination infidelity recorded at scan start.

    Reads the scan's own ``initInfidelities`` (the per-site tail-overlap of
    the double-Gaussian fit at the threshold that DETECTED this run's
    logicals — matching the analysis convention, see module docstring). The
    value is per-site and whole-scan, so it is filter-independent. Returns
    ``None`` when the .mat sidecar carries no infidelities.

    Shape mirrors the per_site map panels (x / y / per_site) so the
    dashboard can render it with the same machinery as loading/survival.
    """
    inf = scan.get('initInfidelities')
    if inf is None:
        return None
    try:
        arr = np.asarray(inf, dtype=float).ravel()
    except (ValueError, TypeError):
        return None
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    x, y = _site_grid_xy(scan)
    with np.errstate(invalid='ignore'):
        return {
            'per_site':         arr.tolist(),
            'mean_infidelity':   float(np.nanmean(arr)),
            'median_infidelity': float(np.nanmedian(arr)),
            'max_infidelity':    float(np.nanmax(arr)),
            'n_sites':           int(arr.size),
            'source':            'scan_init',
            'x':                 x,
            'y':                 y,
        }


def _thresholds_info(scan: dict) -> Optional[dict]:
    """Summary of the per-site detection thresholds used for this run.

    Reads ``initThresholds`` (+ ``initInfidelities``) from the scan's .mat.
    These are the values at SCAN START — the per-run immutable record that
    matches how the stored logicals were detected. Mid-scan refits are only
    persisted to the day-folder ``threshold.mat`` (shared / last-write-wins),
    so end-of-run thresholds are not recoverable per-run today; ``source_note``
    states this. Returns a compact summary (no per-site dump — there can be
    thousands of sites).
    """
    thr = scan.get('initThresholds')
    if thr is None:
        return None
    try:
        t = np.asarray(thr, dtype=float).ravel()
    except (ValueError, TypeError):
        return None
    if t.size == 0 or not np.any(np.isfinite(t)):
        return None
    inf = scan.get('initInfidelities')
    mean_inf = None
    if inf is not None:
        try:
            ia = np.asarray(inf, dtype=float).ravel()
            if ia.size and np.any(np.isfinite(ia)):
                mean_inf = float(np.nanmean(ia))
        except (ValueError, TypeError):
            pass
    with np.errstate(invalid='ignore'):
        return {
            'source':      'scan_init',
            'source_note': ('scan-start per-site thresholds from the scan .mat '
                            '(the values that detected this run; matches the '
                            'analysis convention). Mid-scan refits live only in '
                            'the day-folder threshold.mat and are not stored '
                            'per-run, so end-of-run values are not shown.'),
            'n':           int(t.size),
            'mean':        float(np.nanmean(t)),
            'median':      float(np.nanmedian(t)),
            'min':         float(np.nanmin(t)),
            'max':         float(np.nanmax(t)),
            'mean_infidelity': mean_inf,
        }


def _update_analysis_cache(scan_dir, n_shots, **kv) -> None:
    """Merge keys into analysis_cache.json (resets it if n_shots changed, so a
    growing live run's cache is dropped). Skips None values."""
    cache = _read_analysis_cache(scan_dir) or {}
    if cache.get('n_shots') != n_shots:
        cache = {}
    cache['_version'] = ANALYSIS_CACHE_VERSION
    cache['n_shots'] = n_shots
    for k, v in kv.items():
        if v is not None:
            cache[k] = v
    _write_analysis_cache(scan_dir, cache)


def _recompute_run_metrics(bundle: dict, scan: dict, threshold_segments=None):
    """ONE double-Gaussian fit pass over THIS run's per-site intensities →
    ``(discrimination, imaging_fidelity)``.

    * ``discrimination`` — optimal-cut infidelity per site (the "recompute from
      this run" map/headline; the run's best-achievable discrimination).
    * ``imaging_fidelity`` — average fidelity (``1 - infidelity``) evaluated at
      the thresholds the run ACTUALLY used. When ``threshold_segments`` is given
      (the per-shot threshold timeline reconstructed from the update log) the
      infidelity is the shot-weighted average over those segments — i.e. the
      thresholds in effect THROUGHOUT the run, not just at scan start. With no
      timeline it falls back to the single scan-start ``initThresholds``. Either
      way it reuses the same fit, so it's free here.

    Either element is ``None`` when its prerequisites are missing (no
    intensities → both None; no usable thresholds → imaging_fidelity None).
    """
    inten = bundle.get('intensities')
    if inten is None and bundle.get('two_array'):
        inten = bundle.get('intensities_img1')
    if inten is None:
        return None, None
    arr = np.asarray(inten, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None, None
    try:
        from yb_analysis.detection.dynamical_threshold import (
            fit_run_infidelities, fit_run_infidelities_timeline)
        if threshold_segments:
            opt_thr, opt_inf, used_inf = fit_run_infidelities_timeline(
                arr, threshold_segments)
            used_source = 'used_thresholds_throughout_run'
        else:
            used = np.asarray(scan.get('initThresholds', []),
                              dtype=float).ravel()
            opt_thr, opt_inf, used_inf = fit_run_infidelities(
                arr, used if used.size else None)
            used_source = 'used_thresholds_scan_start'
    except Exception as ex:
        logger.warning('_recompute_run_metrics failed: %s', ex)
        return None, None

    disc = None
    if opt_inf.size and np.any(np.isfinite(opt_inf)):
        x, y = _site_grid_xy(scan)
        with np.errstate(invalid='ignore'):
            disc = {
                'per_site':          opt_inf.tolist(),
                'mean_infidelity':   float(np.nanmean(opt_inf)),
                'median_infidelity': float(np.nanmedian(opt_inf)),
                'max_infidelity':    float(np.nanmax(opt_inf)),
                'n_sites':           int(opt_inf.size),
                'source':            'recomputed_from_run',
                'x':                 x,
                'y':                 y,
                'thresholds':        opt_thr.tolist(),
            }

    imaging = None
    valid = used_inf[np.isfinite(used_inf)] if used_inf.size else np.zeros(0)
    if valid.size:
        with np.errstate(invalid='ignore'):
            imaging = {
                'mean_fidelity':     float(1.0 - np.mean(valid)),
                'median_fidelity':   float(1.0 - np.median(valid)),
                'mean_infidelity':   float(np.mean(valid)),
                'median_infidelity': float(np.median(valid)),
                'n_sites':           int(valid.size),
                'source':            used_source,
            }
    return disc, imaging


def _run_threshold_timeline(scan: dict, scan_dir, n_shots) -> list:
    """Reconstruct the per-site thresholds (and any logged per-site
    infidelities) that were ACTUALLY in effect during this run.

    Seeds with the scan-start ``initThresholds`` / ``initInfidelities``, then
    layers each ``cheap`` / ``fit`` update logged for THIS run's ``scan_id`` (in
    the per-pattern ``update_logs/thresholds/<pattern>.jsonl``) at its ``seq_no``.
    Each segment is weighted by the number of shots it was in effect; segment
    infidelities are carried forward across ``cheap`` updates (which log a new
    threshold but no fresh infidelity).

    Returns a list of segment dicts ``{seq_start, weight, thresholds,
    infidelities, infidelities_eff, source}``, or ``[]`` when there's no usable
    threshold info at all. A pattern run with no in-run update (and a day-folder
    scan with no per-pattern log) yields just the scan-start seed — correct,
    since scan-start then held for the whole run.
    """
    def _vec(key):
        v = scan.get(key)
        if v is None:
            return None
        try:
            a = np.asarray(v, dtype=float).ravel()
        except (ValueError, TypeError):
            return None
        return a if a.size and np.any(np.isfinite(a)) else None

    segs = [{'seq_start': 0, 'thresholds': _vec('initThresholds'),
             'infidelities': _vec('initInfidelities'), 'source': 'scan_init'}]

    scan_id = _scan_id_from_dir(Path(scan_dir))
    patterns = _loading_pattern_names(scan)
    records = []
    if scan_id and patterns:
        try:
            from yb_analysis.analysis import update_log
            for name in patterns:
                recs = update_log.read_threshold_records(name, scan_id=scan_id)
                if recs:
                    records = recs
                    break   # frame-0 loading pattern is what the live refit tracks
        except Exception as ex:  # noqa: BLE001
            logger.debug('threshold timeline read failed: %s', ex)

    for r in records:
        if r.get('source') == 'fit_rejected':
            continue   # rejected fit was never applied — thresholds unchanged
        thr = r.get('thresholds')
        if thr is None:
            continue
        try:
            thr = np.asarray(thr, dtype=float).ravel()
        except (ValueError, TypeError):
            continue
        if thr.size == 0:
            continue
        inf = r.get('infidelities')
        if inf is not None:
            try:
                inf = np.asarray(inf, dtype=float).ravel()
            except (ValueError, TypeError):
                inf = None
        try:
            seq = max(0, int(r.get('seq_no') or 0))
        except (TypeError, ValueError):
            seq = 0
        segs.append({'seq_start': seq, 'thresholds': thr,
                     'infidelities': inf, 'source': r.get('source') or 'update'})

    segs.sort(key=lambda s: s['seq_start'])
    total = max(int(n_shots or 0), segs[-1]['seq_start'] + 1)
    last_inf = None
    for i, s in enumerate(segs):
        nxt = segs[i + 1]['seq_start'] if i + 1 < len(segs) else total
        s['weight'] = max(0, int(nxt) - int(s['seq_start']))
        if s['infidelities'] is not None:
            last_inf = s['infidelities']
        s['infidelities_eff'] = (s['infidelities']
                                 if s['infidelities'] is not None else last_inf)
    if sum(s['weight'] for s in segs) == 0:
        segs[0]['weight'] = 1
    usable = [s for s in segs if s['thresholds'] is not None
              or s.get('infidelities_eff') is not None]
    return usable


def _throughout_run_per_site_infidelity(segs: list):
    """Shot-weighted per-site infidelity vector over the run's threshold
    timeline (scan-start ``initInfidelities`` seed + each full fit's logged
    infidelities, carried forward across cheap updates). Returns an
    ``(M,)`` float array (NaN where no segment contributed) or ``None`` when no
    segment carries a per-site infidelity vector. Cheap — no Gaussian refit."""
    if not segs:
        return None
    n_sites = None
    for s in segs:
        iv = s.get('infidelities_eff')
        if iv is not None and iv.size:
            n_sites = int(iv.size)
            break
    if n_sites is None:
        return None
    num = np.zeros(n_sites)
    wsum = np.zeros(n_sites)
    for s in segs:
        iv = s.get('infidelities_eff')
        w = s.get('weight', 0)
        if iv is None or w <= 0 or iv.size != n_sites:
            continue
        m = np.isfinite(iv)
        num[m] += w * iv[m]
        wsum[m] += w
    good = wsum > 0
    if not good.any():
        return None
    per_site = np.full(n_sites, np.nan)
    per_site[good] = num[good] / wsum[good]
    return per_site


def _imaging_fidelity_from_logged_timeline(segs: list) -> Optional[dict]:
    """Default-view "imaging fidelity (used thresholds)" — cheap, NO Gaussian
    refit. Mean of the shot-weighted per-site infidelity over the run's
    threshold timeline (see :func:`_throughout_run_per_site_infidelity`), then
    ``1 - mean``. Returns ``None`` when no per-site infidelity is available."""
    per_site = _throughout_run_per_site_infidelity(segs)
    if per_site is None:
        return None
    valid = per_site[np.isfinite(per_site)]
    if valid.size == 0:
        return None
    in_run_updates = sum(1 for s in segs
                         if s.get('source') != 'scan_init'
                         and s.get('weight', 0) > 0)
    with np.errstate(invalid='ignore'):
        return {
            'mean_fidelity':     float(1.0 - np.mean(valid)),
            'median_fidelity':   float(1.0 - np.median(valid)),
            'mean_infidelity':   float(np.mean(valid)),
            'median_infidelity': float(np.median(valid)),
            'n_sites':           int(valid.size),
            'source':            'logged_throughout_run',
            'n_in_run_updates':  int(in_run_updates),
        }


def _rearrange_survival_cap(paths_per_shot: list, bundle: dict,
                            scan: dict) -> Optional[dict]:
    """Maximum achievable rearrangement survival implied by source-site
    detection confidence (loss #2).

    Every rearrangement path starts at a SOURCE site the protocol believed held
    an atom (it read loaded in img1). If that detection was a false positive (no
    atom really there), the path cannot deliver an atom → the target stays empty
    → survival is capped below 1 no matter how good the transport is. For each
    path's source we evaluate the posterior ``P(atom present | its img1
    intensity)`` from that site's double-Gaussian fit ("where the detection
    falls relative to the Gaussians"). The whole-run cap is the mean of ``P``
    over all paths; modelling each path's "had an atom" as an independent
    Bernoulli(P), the expected number of nulled paths is ``Σ(1-P)`` and the 1σ
    uncertainty on the cap is ``sqrt(Σ P(1-P)) / N``.

    This quantifies the DETECTION-confidence part of loss #2 only — it does not
    include physical loss of a correctly-detected source atom before transport
    (not observable from the loading image). Returns ``None`` when prerequisites
    are missing (no per-site intensities — e.g. MATLAB scans — or no paths).
    """
    if not paths_per_shot:
        return None
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    if bundle.get('two_array'):
        inten = bundle.get('intensities_img1')
        img1 = np.asarray(inten, dtype=float) if inten is not None else None
    else:
        inten = bundle.get('intensities')
        if inten is None:
            img1 = None
        else:
            a = np.asarray(inten, dtype=float)
            img1 = a[0::num_images] if num_images >= 2 else a
    if img1 is None or img1.ndim != 2 or img1.size == 0:
        return None
    n_shots_av, n_sites = img1.shape

    seq_ids = bundle.get('seq_ids')
    if seq_ids is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()
    n = min(len(seq_ids), n_shots_av)
    if n == 0:
        return None
    seq_ids = seq_ids[:n]
    img1 = img1[:n]
    row_of_seq = {int(s): k for k, s in enumerate(seq_ids)}

    try:
        from yb_analysis.detection.dynamical_threshold import (
            _fit_run_site_params, _gauss_pdf)
        _, _, params_list = _fit_run_site_params(img1)
    except Exception as ex:  # noqa: BLE001
        logger.warning('_rearrange_survival_cap fit failed: %s', ex)
        return None
    if not params_list:
        return None

    src_sites, src_inten = [], []
    n_shots_with_paths = 0
    for entry in paths_per_shot:
        if not isinstance(entry, dict):
            continue
        sid = entry.get('seq_id')
        if sid is None:
            continue
        row = row_of_seq.get(int(sid))
        if row is None:
            continue
        lp = np.asarray(entry.get('loaded_paired') or [],
                        dtype=np.int64).ravel()
        lp = lp[(lp >= 0) & (lp < n_sites)]
        if lp.size == 0:
            continue
        n_shots_with_paths += 1
        for i in lp:
            src_sites.append(int(i))
            src_inten.append(float(img1[row, i]))
    if not src_sites:
        return None
    src_sites = np.asarray(src_sites, dtype=np.int64)
    src_inten = np.asarray(src_inten, dtype=np.float64)

    # Posterior per source path (vectorised over the gathered (site, intensity)
    # pairs). Sites whose fit failed/degenerate get P=1.0 (they WERE detected
    # loaded — the optimistic, honest fallback) and are counted as `n_no_fit`.
    mu_e = np.array([p[0] if p is not None else np.nan for p in params_list])
    s_e  = np.array([p[1] if p is not None else np.nan for p in params_list])
    A_e  = np.array([p[2] if p is not None else np.nan for p in params_list])
    mu_a = np.array([p[3] if p is not None else np.nan for p in params_list])
    s_a  = np.array([p[4] if p is not None else np.nan for p in params_list])
    A_a  = np.array([p[5] if p is not None else np.nan for p in params_list])
    se = s_e[src_sites]
    sa = s_a[src_sites]
    valid = np.isfinite(se) & np.isfinite(sa) & (se > 0) & (sa > 0)
    P = np.ones(src_sites.size, dtype=np.float64)
    if valid.any():
        with np.errstate(invalid='ignore', divide='ignore'):
            pe = A_e[src_sites][valid] * _gauss_pdf(
                src_inten[valid], mu_e[src_sites][valid], se[valid])
            pa = A_a[src_sites][valid] * _gauss_pdf(
                src_inten[valid], mu_a[src_sites][valid], sa[valid])
            denom = pe + pa
            pp = np.where(denom > 0, pa / denom, np.nan)
        P[valid] = np.clip(pp, 0.0, 1.0)
    n_no_fit = int((~valid).sum() + int(np.isnan(P).sum()))
    P = np.where(np.isfinite(P), P, 1.0)

    N = int(P.size)
    sum_P = float(P.sum())
    sum_PQ = float(np.sum(P * (1.0 - P)))
    cap = sum_P / N
    cap_sem = float(np.sqrt(max(sum_PQ, 0.0)) / N)
    return {
        'cap_mean':           cap,
        'cap_sem':            cap_sem,
        'expected_nulled':    float(N - sum_P),
        'nulled_sem':         float(np.sqrt(max(sum_PQ, 0.0))),
        'n_paths':            N,
        'n_shots_with_paths': int(n_shots_with_paths),
        'n_no_fit':           n_no_fit,
        'mean_paths_per_shot': (float(N / n_shots_with_paths)
                                if n_shots_with_paths else None),
        'source':             'source_site_detection_posterior',
    }


# Filtered TP: a target site counts as an "outlier bad-fidelity" spot — and is
# dropped from the filtered target survival — when its detection infidelity is
# BOTH a robust outlier among the target sites (> median + K·1.4826·MAD) AND
# above an absolute floor (so a uniformly-clean target set keeps every site).
FILTERED_TP_MAD_K = 5.0
FILTERED_TP_ABS_FLOOR = 0.02


def _outlier_bad_fidelity_sites(per_site_infid, candidate_sites,
                                mad_k=FILTERED_TP_MAD_K,
                                abs_floor=FILTERED_TP_ABS_FLOOR):
    """Among ``candidate_sites`` (indices into ``per_site_infid``), return the
    indices whose infidelity marks them as outlier bad-fidelity spots: above
    ``max(median + mad_k·1.4826·MAD, abs_floor)`` over the candidates' finite
    infidelities. Returns ``(bad_idx (int64), threshold (float|None))``. Sites
    with NaN / missing infidelity are NOT flagged (unknown ≠ bad)."""
    cand = np.asarray(candidate_sites, dtype=np.int64).ravel()
    psi = np.asarray(per_site_infid, dtype=float).ravel()
    cand = cand[(cand >= 0) & (cand < psi.size)]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64), None
    vals = psi[cand]
    finite = np.isfinite(vals)
    if not finite.any():
        return np.empty(0, dtype=np.int64), None
    fv = vals[finite]
    med = float(np.median(fv))
    mad = float(np.median(np.abs(fv - med)))
    thr = max(med + mad_k * 1.4826 * mad, float(abs_floor))
    bad_mask = finite & (vals > thr)
    return cand[bad_mask], thr


def _filtered_target_aware(paths_per_shot, bundle, scan, scan_params, *,
                           reps_per_param=None, per_site_infid=None,
                           infid_source='throughout_run'):
    """Target-aware (TP) survival recomputed with outlier bad-fidelity target
    spots EXCLUDED.

    Cheap — NO Gaussian refit: it uses the per-site infidelities already
    available (the throughout-run logged values, which collapse to the
    scan-start ``initInfidelities`` when the run logged no fit), so it shows in
    the default view WITHOUT a recompute. Drops the outlier bad-fidelity target
    sites (see :func:`_outlier_bad_fidelity_sites`) and re-runs the same cheap
    lab-paths TP over the remaining good targets.

    Returns a compact whole-run dict (overall TP over the good targets + what
    was excluded), or ``None`` when it can't be computed (no lab paths, no
    per-site infidelity, or no target sites)."""
    if not paths_per_shot or per_site_infid is None:
        return None
    psi = np.asarray(per_site_infid, dtype=float).ravel()
    if psi.size == 0 or not np.any(np.isfinite(psi)):
        return None
    # Candidate target sites = union over shots of the per-shot target indices
    # (the direct-index primary; matches _entry_target_sites' first branch).
    cand = set()
    for e in paths_per_shot:
        if not isinstance(e, dict):
            continue
        for key in ('target_site_indices', 'target_paired'):
            ids = e.get(key)
            if ids:
                for i in ids:
                    ii = int(i)
                    if 0 <= ii < psi.size:
                        cand.add(ii)
                break
    if not cand:
        return None
    cand_arr = np.array(sorted(cand), dtype=np.int64)
    bad, thr = _outlier_bad_fidelity_sites(psi, cand_arr)
    ta = _target_aware_from_lab_paths(
        paths_per_shot, bundle, scan, scan_params,
        reps_per_param=reps_per_param, param_mask=None,
        exclude_sites=bad if bad.size else None)
    if ta is None:
        return None
    n_excl = int(bad.size)
    excl_infid = psi[bad][np.isfinite(psi[bad])] if n_excl else np.zeros(0)
    return {
        'source':               ta.get('source'),
        'overall_mean':         ta.get('overall_mean'),
        'overall_sem':          ta.get('overall_sem'),
        'per_param_mean':       ta.get('per_param_mean'),
        'per_param_sem':        ta.get('per_param_sem'),
        'n_target_sites':       int(cand_arr.size),
        'n_excluded':           n_excl,
        'n_kept':               int(cand_arr.size - n_excl),
        'infidelity_threshold': (float(thr) if thr is not None else None),
        'excluded_max_infid':   (float(np.max(excl_infid))
                                  if excl_infid.size else None),
        'infid_source':         infid_source,
    }


def _intensity_hist(bundle: dict, n_bins: int = 120,
                    max_samples: int = 2_000_000) -> Optional[dict]:
    """Pooled histogram of per-site camera intensities for this run.

    Mirrors the live view's intensity histogram but aggregated over ALL sites
    + shots (image-1), so the operator can eyeball the empty/atom bimodality
    and overall data quality. Returns ``None`` when intensities aren't stored
    (e.g. MATLAB-written scans). Range clipped to [p0.05, p99.95] so a few
    cosmic-ray outliers don't squash the structure.
    """
    inten = bundle.get('intensities')
    if inten is None and bundle.get('two_array'):
        inten = bundle.get('intensities_img1')
    if inten is None:
        return None
    flat = np.asarray(inten, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    n_total = int(flat.size)
    if flat.size > max_samples:
        idx = np.linspace(0, flat.size - 1, max_samples).astype(int)
        flat = flat[idx]
    lo, hi = np.nanpercentile(flat, [0.05, 99.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat)) + 1.0
    counts, edges = np.histogram(flat, bins=n_bins, range=(float(lo), float(hi)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        'counts':      counts.tolist(),
        'bin_centers': centers.tolist(),
        'n_samples':   n_total,
    }


def _fmt_param_value(v):
    """Coerce a SetParams/DefaultParams value to a JSON-safe display value.

    Handles MATLAB char arrays (uint16 -> string), scalars, short numeric
    arrays, and falls back to a shape tag for big arrays.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        a = np.asarray(v)
    except Exception:
        return str(v)
    if a.size == 0:
        return None
    if a.dtype.kind in ('U', 'S'):
        return ' '.join(str(x) for x in a.ravel().tolist())
    if a.dtype.kind in ('u', 'i') and a.dtype.itemsize <= 8 and 1 < a.size < 256:
        # uint16 char array (a string) vs a small int array: if every value
        # is printable ASCII, treat as text (MATLAB stores char arrays this
        # way). A MATLAB `string` object (double-quoted) instead serializes
        # to an undecodable MCOS blob (huge sentinel values) — flag those as
        # unreadable rather than printing the raw integers.
        vals = a.ravel().tolist()
        if all(32 <= int(x) < 127 for x in vals):
            return ''.join(chr(int(x)) for x in vals)
        if any(abs(int(x)) > 1_000_000 for x in vals):
            return None   # MATLAB string-object blob — not decodable here
        return vals
    if a.size == 1:
        x = a.ravel()[0]
        return float(x) if a.dtype.kind == 'f' else int(x)
    if a.size <= 8:
        return a.ravel().tolist()
    return f'<{a.dtype} array {list(a.shape)}>'


def _run_parameters(scan: dict, sweep_all) -> list:
    """Every parameter that defines this run, for the Details panel.

    Order: the SWEPT axes first (each shown as its value list, the way it
    reads in MATLAB), then the fixed parameters from the Scan struct's
    ``SetParams`` + ``DefaultParams`` (MOT/LAC settings, rearrange kwargs, ...).
    Returns ``[{name, value, group}, ...]`` — group is 'swept' | 'SetParams' |
    'DefaultParams'.
    """
    sweep_all = sweep_all or {}
    cols = sweep_all.get('cols') or []
    values = sweep_all.get('values') or []
    swept_norm = {str(c).replace('.', '_').lower() for c in cols}
    out = []
    # Swept axes first — show the full value list (MATLAB-style).
    for i, name in enumerate(cols):
        vals = values[i] if i < len(values) else []
        out.append({'name': str(name),
                    'value': [round(float(v), 6) if isinstance(v, (int, float))
                              else v for v in vals],
                    'group': 'swept'})
    # Fixed params (skip any that are also swept, defensively).
    for group in ('SetParams', 'DefaultParams'):
        d = scan.get(group)
        if not isinstance(d, dict):
            continue
        for k in sorted(d.keys()):
            if str(k).replace('.', '_').lower() in swept_norm:
                continue
            val = _fmt_param_value(d[k])
            if val is None:
                continue   # unreadable (e.g. MATLAB string-object blob)
            out.append({'name': str(k), 'value': val, 'group': group})
    # pyctrl has no SetParams/DefaultParams — its fixed params live in
    # ScanGroup.base.params (a nested dotted tree, e.g. Pushout.Green.Amp).
    # Flatten to dotted leaf paths when the MATLAB fields produced nothing.
    if not any(p['group'] != 'swept' for p in out):
        sg = scan.get('ScanGroup')
        base = sg.get('base') if isinstance(sg, dict) else None
        bp = base.get('params') if isinstance(base, dict) else None
        if isinstance(bp, dict):
            for name, val in _flatten_dotted(bp):
                if str(name).replace('.', '_').lower() in swept_norm:
                    continue
                v = _fmt_param_value(val)
                if v is None:
                    continue
                out.append({'name': name, 'value': v, 'group': 'base'})
    # --- Everything ELSE that defines the run, so it is 100% reproducible -----
    # The blocks above cover the swept axes + the fixed *sequence* params. But a
    # pyctrl run also carries a RUN-SETTINGS block (descriptor['runp']) and a
    # baseline device-config snapshot (expConfig) that were stored but never
    # shown. In particular a rearrangement run records its SLM model checkpoint,
    # warmup phases, compile flags, defocus, scramble, lock mode, ... ONLY in
    # descriptor['runp'] (e.g. warmup_kwargs.model_filename). Surface them all,
    # de-duped against what's already listed.
    seen = {str(p['name']).replace('.', '_').lower() for p in out} | swept_norm
    desc = scan.get('descriptor')
    if isinstance(desc, dict):
        # descriptor['params'] is a FLAT dotted dict (fixed leaves + sweep
        # {scan,values} dicts). Skip the sweep leaves (already shown as 'swept').
        dparams = desc.get('params')
        if isinstance(dparams, dict):
            for name, raw in dparams.items():
                if not _is_sweep_desc(raw):
                    _add_param(out, seen, name, raw, 'param')
        # descriptor['runp'] is a FLAT dotted dict of run settings — the SLM
        # model + warmup/compile/defocus/scramble/lock knobs live here.
        runp = desc.get('runp')
        if isinstance(runp, dict):
            for name, raw in runp.items():
                _add_param(out, seen, name, raw, 'runp')
    # expConfig: the baseline device-config snapshot (NESTED) — imaging ROI,
    # resonance freqs, MOT/LAC/Pushout/AWG/SLM settings. Embedded verbatim by
    # pyctrl's scan_summary for provenance; flatten to dotted leaves.
    ec = scan.get('expConfig')
    if isinstance(ec, dict):
        for name, raw in _flatten_dotted(ec):
            _add_param(out, seen, name, raw, 'config')
    return out


def _flatten_dotted(d, prefix=''):
    """Flatten a nested dict to ``[(dotted.path, leaf_value), ...]``."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            out.extend(_flatten_dotted(v, key))
    else:
        out.append((prefix, d))
    return out


def _is_sweep_desc(v):
    """A descriptor sweep leaf: ``{"scan": dim, "values": [...]}`` (rendered as a
    swept axis, so skipped when listing the descriptor's fixed params)."""
    return isinstance(v, dict) and 'scan' in v and 'values' in v


def _add_param(out, seen, name, raw, group):
    """Append ``{name, value, group}`` to ``out`` unless ``name`` (case/dot-
    insensitive) was already listed or the value isn't display-able. Mutates
    ``seen``. Keeps the Details panel free of confusing duplicate rows (an
    operative scan value wins over a baseline snapshot of the same key) while
    surfacing every otherwise-unshown parameter."""
    key = str(name).replace('.', '_').lower()
    if key in seen:
        return
    v = _fmt_param_value(raw)
    if v is None:
        return
    seen.add(key)
    out.append({'name': str(name), 'value': v, 'group': group})


def _loading_pattern_names(scan: dict) -> list:
    """Loading-pattern name(s) for this run from ``imagePatternsJson``.

    The field is a JSON array of ``{"name": ..., ...}`` (one per image).
    Returns the unique names in order, or ``[]`` when absent/unparseable.
    """
    raw = _str_or_none(scan.get('imagePatternsJson'))
    if not raw:
        return []
    try:
        arr = json.loads(raw)
    except (ValueError, TypeError):
        return []
    names = []
    if isinstance(arr, list):
        for e in arr:
            if isinstance(e, dict) and e.get('name'):
                n = str(e['name'])
                if n not in names:
                    names.append(n)
    return names


# Diag columns that are run SETTINGS (not per-shot outcomes) worth showing in
# the Details panel when constant across the run. SetParams stores some of
# these as undecodable MATLAB string objects, so the diag is the clean source.
_DIAG_PARAM_FIELDS = ('protocol', 'handoff_protocol', 'two_round_phase',
                      'n_sites_model', 'n_total_sites')


def _run_params_from_diag(diag_path: Path, exclude_names) -> list:
    """Clean rearrangement settings from slm_diag.h5 (constant columns only).

    Surfaces the whitelisted ``_DIAG_PARAM_FIELDS`` when they hold a single
    value across all shots (e.g. ``protocol='rearrange'``). Skips names
    already present from SetParams. Returns ``[{name, value, group}, ...]``.
    """
    if not diag_path.is_file():
        return []
    try:
        import h5py
    except ImportError:
        return []
    out = []
    excl = {str(n).lower() for n in (exclude_names or [])}
    try:
        with h5py.File(diag_path, 'r') as f:
            g = f['/diag'] if '/diag' in f else f
            for name in _DIAG_PARAM_FIELDS:
                if name in excl or name not in g:
                    continue
                try:
                    arr = g[name][:]
                except Exception:
                    continue
                if getattr(arr, 'ndim', 0) != 1 or arr.shape[0] == 0:
                    continue
                # decode bytes -> str for the uniqueness check + display
                vals = [v.decode('utf-8', 'replace') if isinstance(v, bytes)
                        else v for v in arr.tolist()]
                uniq = set(vals)
                if len(uniq) != 1:
                    continue   # not constant -> not a fixed setting
                v = next(iter(uniq))
                if v in ('', None):
                    continue
                out.append({'name': name,
                            'value': (float(v) if isinstance(v, float)
                                      else int(v) if isinstance(v, int)
                                      else str(v)),
                            'group': 'rearrange'})
    except (OSError, KeyError):
        return out
    return out


def _human_duration(seconds: float) -> str:
    """Compact human duration: '2.4 days' / '5.1 h' / '37 min' / '12 s'."""
    s = abs(float(seconds))
    if s >= 86400:
        return f'{s / 86400:.1f} days'
    if s >= 3600:
        return f'{s / 3600:.1f} h'
    if s >= 60:
        return f'{s / 60:.0f} min'
    return f'{s:.0f} s'


def _calibration_age_info(scan: dict, scan_dir) -> dict:
    """How old the detection calibration was at run start (the operator's
    'staleness' marker). Run start = the scan_id timestamp; calibration time =
    a stamped ``calibrationTimestamp`` if pyctrl recorded one, else the mtime
    of the calibration source file (the per-pattern ``threshold.mat`` when
    ``calibrationSource`` is ``pattern:<name>``, else the day-folder
    ``threshold.mat``). Returns ``{}`` when nothing is resolvable."""
    from datetime import datetime
    scan_id = _scan_id_from_dir(Path(scan_dir))
    try:
        run_start = datetime.strptime(scan_id, '%Y%m%d%H%M%S')
    except (ValueError, TypeError):
        return {}
    src = scan.get('calibrationSource')
    calib_time = None
    basis = None
    iso = scan.get('calibrationTimestamp')
    if iso:
        try:
            calib_time = datetime.fromisoformat(str(iso))
            basis = 'stamped'
        except (ValueError, TypeError):
            calib_time = None
    if calib_time is None:
        path = None
        if isinstance(src, str) and src.startswith('pattern:'):
            name = src.split(':', 1)[1]
            path = os.path.join(_yb_cfg.PATH_PREFIX, 'yb_dashboard_state',
                                'patterns', name, 'threshold.mat')
        else:
            path = os.path.join(os.path.dirname(str(scan_dir)), 'threshold.mat')
        if path and os.path.isfile(path):
            calib_time = datetime.fromtimestamp(os.path.getmtime(path))
            basis = 'file_mtime'
    if calib_time is None:
        return {'calibration_source': src} if src else {}
    age_s = (run_start - calib_time).total_seconds()
    return {
        'calibration_source':    src or 'day_folder',
        'calibration_iso':       calib_time.isoformat(timespec='seconds'),
        'calibration_age_s':     age_s,
        'calibration_age_human': _human_duration(age_s),
        'calibration_age_basis': basis,   # 'stamped' (exact) | 'file_mtime' (approx)
    }


def _scan_description(scan: dict) -> Optional[str]:
    """Best-effort free-text description / note attached to the scan.

    Reads the top-level sidecar keys first (pyctrl stamps ``description`` there from the
    descriptor; MATLAB/legacy scans may use ``note``/``comment``/...), then falls back to the
    self-contained reconstruction ``descriptor`` block if a run only carries it there. Returns
    None when no description was set (every pre-feature scan)."""
    for k in ('scannote', 'scanNote', 'ScanNote', 'comment', 'Comment',
              'description', 'Description', 'note', 'notes'):
        s = _str_or_none(scan.get(k))
        if s:
            return s
    desc = scan.get('descriptor')
    if isinstance(desc, dict):
        s = _str_or_none(desc.get('description'))
        if s:
            return s
    return None


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


def _entry_target_sites(entry, site_xy, n_sites, tol2):
    """Lab-grid site indices targeted on one shot (rearrangement).

    PRIMARY (sidecar-free): ``target_paired`` are per-shot bit indices in the
    SHARED grid order, which equals the lab grid / logicals order (the SLM
    server reconciles its grid to the lab /api/grid), so they index the lab
    grid DIRECTLY — no grid sidecar needed. This is the path that should run
    for current runs: all the target info comes from the diag/rearrange
    ledger, computed entirely lab-side.

    FALLBACK (legacy entries that carry sidecar coords but no indices):
    ``target_xy`` coordinates matched to the nearest lab site within ``tol2``.

    Returns a unique int64 index array into ``site_xy`` (possibly empty)."""
    # Prefer the FULL active target set (every site the protocol is filling,
    # e.g. the every_other checkerboard) for target-aware TP/FP; fall back to
    # the placed pairing (target_paired) if the active set wasn't recorded.
    for key in ('target_site_indices', 'target_paired'):
        ids = entry.get(key)
        if ids:
            idx = np.asarray(ids, dtype=np.int64).ravel()
            return np.unique(idx[(idx >= 0) & (idx < n_sites)])
    # Legacy coord fallback needs the lab grid; skip when unavailable.
    if site_xy is None:
        return np.empty(0, dtype=np.int64)
    txy = entry.get('target_xy') or []
    valid_xy = [tuple(t) for t in txy
                if isinstance(t, (list, tuple)) and len(t) == 2
                and all(isinstance(x, (int, float)) and x is not None
                        for x in t)]
    if not valid_xy:
        return np.empty(0, dtype=np.int64)
    t_arr = np.asarray(valid_xy, dtype=float)
    d2 = ((site_xy.reshape(1, -1, 2) - t_arr.reshape(-1, 1, 2)) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    ok = d2[np.arange(t_arr.shape[0]), nearest] <= tol2
    return np.unique(nearest[ok])


def _target_aware_from_lab_paths(paths_per_shot: list,
                                  bundle: Optional[dict],
                                  scan: dict,
                                  scan_params: np.ndarray,
                                  *,
                                  reps_per_param: Optional[np.ndarray] = None,
                                  param_mask: Optional[np.ndarray] = None,
                                  exclude_sites: Optional[np.ndarray] = None
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

    # Lab img2 (per-shot, per-site detection from data_*.h5) — computed FIRST
    # so n_sites comes from the logicals width, not the grid coords.
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
    if img2.size == 0:
        return None
    n_sites = int(img2.shape[1])

    # Lab img1 (loaded) — for the target-aware FP (spurious atoms at empty
    # NON-target sites). Same layout handling as img2; None if unavailable.
    if raw1 is not None and raw2 is not None:
        img1 = np.asarray(raw1, dtype=np.uint8)
    elif raw is not None:
        _a1 = np.asarray(raw, dtype=np.uint8)
        img1 = _a1[0::num_images] if num_images >= 2 else _a1
    else:
        img1 = None

    # Lab-side site coords from Scan.initGridLocationsX/Y are OPTIONAL: the
    # primary target path (target_site_indices / target_paired) indexes the
    # logicals DIRECTLY, so a run with no baked grid (e.g. a legacy pyctrl run
    # whose config predates calibration-baking) still gets target-aware
    # survival. Coords are only needed for the legacy target_xy→nearest-site
    # fallback, so we keep them when present + length-consistent, else None.
    gx, gy = _site_grid_xy(scan)
    site_xy = None
    if gx and gy:
        _sxy = np.column_stack([np.asarray(gy, dtype=float),
                                np.asarray(gx, dtype=float)])  # (n_sites, 2)
        if _sxy.shape[0] == n_sites:
            site_xy = _sxy

    seq_ids = bundle.get('seq_ids')
    if seq_ids is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()
    n_shots_avail = min(len(seq_ids), img2.shape[0])
    seq_ids = seq_ids[:n_shots_avail]
    img2 = img2[:n_shots_avail]
    if img1 is not None:
        img1 = img1[:n_shots_avail]

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
    # Target-aware FP: spurious atoms at empty NON-target sites (per param).
    sum_fp = np.zeros(n_params_unfilt, dtype=np.float64)
    n_fp_elig = np.zeros(n_params_unfilt, dtype=np.int64)
    # Per-shot TP / FP for the per-iteration override (indexed
    # 0..n_shots_avail-1 matching bundle's processed-shot order).
    per_shot_tp = [None] * n_shots_avail
    per_shot_fp = [None] * n_shots_avail

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

    # Matching tolerance for the legacy target_xy fallback only (the primary
    # target_paired bit-index path ignores it). Computed once. n_sites comes
    # from the logicals width so the index path works without a grid; the
    # tolerance is only meaningful when we actually have coords.
    n_sites_ta = n_sites
    tol2_ta = 0.0
    if site_xy is not None:
        _ssub = site_xy[: min(n_sites_ta, 1000)]
        _d2s = ((_ssub.reshape(-1, 1, 2) - _ssub.reshape(1, -1, 2)) ** 2).sum(axis=2)
        if _d2s.shape[0] > 1:
            _d2s[np.arange(_d2s.shape[0]), np.arange(_d2s.shape[0])] = np.inf
            _spacing_ta = max(float(np.median(np.sqrt(_d2s.min(axis=1)))), 1.0)
        else:
            _spacing_ta = 1.0
        tol2_ta = (2.0 * _spacing_ta) ** 2

    # Cross-grid run (init pattern != target pattern): img2 is detected ON the
    # target grid, so EVERY target-grid site is an atom destination. The
    # run-level ACTIVE-target set (every site ever targeted) then defines the
    # NON-target sites for FP -- a target site left unfilled on a low-loading
    # shot is an unfilled target, NOT a false-positive site. For a full-array
    # target there are no non-target sites and FP is undefined (as expected).
    cross_grid = (img1 is not None and img1.shape[1] != n_sites)
    active_union = None
    if cross_grid:
        active_union = np.zeros(n_sites, dtype=bool)
        for entry in by_seq.values():
            aidx = _entry_target_sites(entry, site_xy, n_sites_ta, tol2_ta)
            if aidx.size:
                active_union[aidx[(aidx >= 0) & (aidx < n_sites)]] = True

    # Optional site exclusion (filtered TP: drop outlier bad-fidelity target
    # spots so transport survival isn't biased by mis-detected targets).
    excl_arr = (np.asarray(exclude_sites, dtype=np.int64).ravel()
                if exclude_sites is not None and len(exclude_sites) else None)

    n_with_targets = 0
    for k in range(n_shots_avail):
        sid = int(seq_ids[k])
        entry = by_seq.get(sid)
        if entry is None:
            continue
        # Target lab sites this shot: target_paired bit indices (sidecar-free
        # primary) or target_xy->nearest (legacy fallback).
        idx = _entry_target_sites(entry, site_xy, n_sites_ta, tol2_ta)
        if excl_arr is not None and idx.size:
            idx = idx[~np.isin(idx, excl_arr)]
        if idx.size == 0:
            continue
        # TP this shot: of the target sites, how many showed an atom in img2?
        n_targets_matched = int(idx.size)
        n_hit = int(img2[k, idx].sum())
        tp_this_shot = n_hit / n_targets_matched
        per_shot_tp[k] = tp_this_shot
        # Scan-param this shot belongs to (UNFILTERED 0-indexed).
        p0 = int(per_shot_param0[k])
        if p0 < 0 or p0 >= n_params_unfilt:
            continue
        sum_tp[p0] += tp_this_shot
        n_eligible[p0] += 1
        n_with_targets += 1
        # Target-aware FP this shot: of the EMPTY non-target sites, what
        # fraction spuriously showed an atom in img2? (atoms moved into
        # targets are NOT false positives — matches per_iteration fp_source.)
        #   * same grid (img1 width == img2 width): eligible = sites empty in
        #     img1 AND not in the target set.
        #   * cross grid (init pattern != target pattern): img1 lives on a
        #     DIFFERENT grid, so its per-site emptiness can't be indexed by an
        #     img2 site; eligible = the img2 (target-grid) sites NOT in the
        #     active target set (those are meant to be empty post-rearrange).
        tmask = np.zeros(n_sites, dtype=bool)
        tmask[idx[(idx >= 0) & (idx < n_sites)]] = True
        if not cross_grid and img1 is not None and img1.shape[1] == n_sites:
            base = (~img1[k].astype(bool)) & (~tmask)
        else:
            # Cross-grid: non-target = sites never part of the target pattern
            # (run-level), so unfilled targets aren't miscounted as FP.
            base = (~active_union) if active_union is not None else ~tmask
        nb = int(base.sum())
        if nb > 0:
            fp_s = float((img2[k].astype(bool) & base).sum()) / nb
            per_shot_fp[k] = fp_s
            sum_fp[p0] += fp_s
            n_fp_elig[p0] += 1

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

    # Target-aware FP per param + eligibility-weighted overall (same filter
    # handling as TP). NaN where no empty-non-target sites were eligible.
    with np.errstate(invalid='ignore', divide='ignore'):
        fp_mean_unfilt = np.where(n_fp_elig > 0, sum_fp / n_fp_elig, np.nan)
        fp_sem_unfilt  = np.where(n_fp_elig > 0,
                                   np.sqrt(np.clip(fp_mean_unfilt, 0, 1)
                                           * (1 - np.clip(fp_mean_unfilt, 0, 1))
                                           / np.maximum(1, n_fp_elig)), np.nan)
    if param_mask is not None and fp_mean_unfilt.size == param_mask.size:
        fp_mean = fp_mean_unfilt[param_mask]
        fp_sem  = fp_sem_unfilt[param_mask]
        fp_elig = n_fp_elig[param_mask]
    else:
        fp_mean, fp_sem, fp_elig = fp_mean_unfilt, fp_sem_unfilt, n_fp_elig
    fvalid = np.isfinite(fp_mean)
    if fvalid.any() and fp_elig[fvalid].sum():
        fp_overall = float(np.average(fp_mean[fvalid], weights=fp_elig[fvalid]))
    else:
        fp_overall = None

    return {
        'source':          'lab_paths',
        'per_param_mean':  per_mean.tolist(),
        'per_param_sem':   per_sem.tolist(),
        'overall_mean':    overall,
        'overall_sem':     overall_sem,
        'loss_overall':    (1.0 - overall) if overall is not None else None,
        # Target-aware FP (spurious atoms at empty non-target sites).
        'fp_overall':      fp_overall,
        'per_param_fp':    fp_mean.tolist(),
        'per_param_fp_sem': fp_sem.tolist(),
        'per_shot_fp':     per_shot_fp,
        'axes_matched':    ([] if scan.get('ScanVar') is None
                            else list(np.atleast_1d(scan.get('ScanVar')))),
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


def _target_grid_camera_xy(scan: dict, n_target_sites: int):
    """Camera-pixel coords for the TARGET (img2) grid of a cross-grid run.

    A rearrangement scan whose initial loading pattern and final target
    pattern differ bakes only the INITIAL (loading) grid into the scan config
    (``initGridLocationsX/Y``); the target grid lives in the per-pattern
    registry as knm-1024 coords. We recover an affine knm -> camera from the
    INITIAL pattern (its registry knm vs the run's own ``initGridLocations``,
    same col-major order) and apply it to the TARGET pattern's knm grid so the
    per-site survival/FP map can be drawn on the camera image.

    Returns ``(n_target_sites, 2)`` (y, x) camera px, or ``None`` when any
    input is missing / the affine can't be fit / the counts don't line up.
    """
    try:
        from yb_analysis.analysis import pattern_registry as _pr
    except Exception:   # noqa: BLE001 -- registry optional
        return None
    names = _loading_pattern_names(scan)
    if len(names) < 2:
        return None
    init_rec = _pr.get_pattern(names[0])
    tgt_rec  = _pr.get_pattern(names[-1])
    if not init_rec or not tgt_rec:
        return None
    try:
        k_init = np.asarray(init_rec.get('knm') or [], dtype=float)
        k_tgt  = np.asarray(tgt_rec.get('knm') or [], dtype=float)
    except (ValueError, TypeError):
        return None
    if (k_init.ndim != 2 or k_init.shape[1] != 2
            or k_tgt.ndim != 2 or k_tgt.shape[1] != 2
            or k_tgt.shape[0] != int(n_target_sites)):
        return None
    gx, gy = _site_grid_xy(scan)
    if not gx or not gy:
        return None
    cam = np.column_stack([np.asarray(gy, float), np.asarray(gx, float)])
    if cam.shape[0] != k_init.shape[0]:
        return None
    # Fit the knm -> camera affine from the initial pattern (registry knm and
    # the run's own initGridLocations share col-major order). Verify with the
    # residual: a real lattice<->lattice affine fits to << 1 px; a bad
    # correspondence/order blows up to many px and is rejected.
    X = np.column_stack([k_init, np.ones(len(k_init))])
    try:
        A, *_ = np.linalg.lstsq(X, cam, rcond=None)
    except np.linalg.LinAlgError:
        return None
    rms = float(np.sqrt((((X @ A) - cam) ** 2).sum(1).mean()))
    if not np.isfinite(rms) or rms > 5.0:
        return None
    return np.column_stack([k_tgt, np.ones(len(k_tgt))]) @ A   # (n, 2) (y, x)


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

    # Detection grid for the INITIAL (loading) image, baked into the scan
    # config as initGridLocationsX/Y.
    gx, gy = _site_grid_xy(scan)
    init_site_xy = (np.column_stack([np.asarray(gy, dtype=float),
                                     np.asarray(gx, dtype=float)])
                    if (gx and gy) else None)

    # Cross-grid run (init pattern != target pattern): img2 is detected on the
    # TARGET grid (different site count than the loading grid), so the per-site
    # SURVIVAL/FP map lives on the target grid. Recover its camera coords by
    # affine-mapping the target pattern's registry knm grid. Per-site loading
    # stays an init-grid quantity and is omitted from this (target-grid) map.
    cross_grid = (init_site_xy is not None
                  and img2.shape[1] != init_site_xy.shape[0])
    if cross_grid:
        site_xy = _target_grid_camera_xy(scan, img2.shape[1])
        loading_on_grid = False
    else:
        site_xy = init_site_xy
        loading_on_grid = True
    if site_xy is None or site_xy.shape[0] == 0:
        return None
    n_sites = site_xy.shape[0]
    if img2.shape[1] != n_sites:
        return None

    seq_ids = bundle.get('seq_ids')
    if seq_ids is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()
    n_shots = min(len(seq_ids), img2.shape[0], img1.shape[0])
    if n_shots == 0:
        return None
    # Truncate to the common shot count: img1/img2 can have more rows than
    # seq_ids (e.g. an extra/partial trailing frame, or uneven 0::N / N-1::N
    # interleave splits), and has_paths below is sized to n_shots. Without
    # this, img1[has_paths] raises "boolean index did not match" when
    # img1.shape[0] > len(has_paths). (Matches the other per-shot helpers.)
    img1 = img1[:n_shots]
    img2 = img2[:n_shots]
    seq_ids = seq_ids[:n_shots]

    by_seq = {int(e['seq_id']): e for e in paths_per_shot
               if isinstance(e, dict) and 'seq_id' in e}

    # Per-shot PLACED-target mask (drives TP) + the run-level ACTIVE-target set
    # (every site EVER targeted -> drives which sites are non-targets for FP).
    # Memory: n_shots × n_sites bytes — 1000×3270 = 3 MB max.
    target_mask  = np.zeros((n_shots, n_sites), dtype=bool)
    active_union = np.zeros(n_sites, dtype=bool)
    has_paths    = np.zeros(n_shots, dtype=bool)
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
        idx = _entry_target_sites(entry, site_xy, n_sites, tol2)
        if idx.size == 0:
            continue
        target_mask[k, idx] = True
        active_union[idx] = True
        has_paths[k] = True

    # Only count shots that actually had paths data. (Legacy-style empty
    # entries shouldn't influence FP counts.)
    if not has_paths.any():
        return None
    img1_used = img1[has_paths]
    img2_used = img2[has_paths]
    img2b = img2_used.astype(bool)
    tmask_used = target_mask[has_paths]
    target_elig = tmask_used.sum(axis=0).astype(np.int64)
    tp_hits = (img2b & tmask_used).sum(axis=0).astype(np.int64)
    # False positives = spurious atoms at sites NOT part of the target pattern.
    # The non-target set is the complement of the run-level ACTIVE-target set
    # (every site ever targeted), NOT the per-shot placed pairs: a target site
    # left unfilled on a low-loading shot is an UNFILLED TARGET, not a
    # false-positive site. For a full-array target (every site targeted) there
    # are no non-target sites, so FP is undefined everywhere (as expected).
    if cross_grid:
        nontarget_site = ~active_union
        n_pshots = int(has_paths.sum())
        nontarget_elig = np.where(nontarget_site, n_pshots, 0).astype(np.int64)
        fp_hits = np.where(nontarget_site,
                           img2b.sum(axis=0), 0).astype(np.int64)
    else:
        nontmask_used = ~tmask_used
        nontarget_elig = nontmask_used.sum(axis=0).astype(np.int64)
        fp_hits = (img2b & nontmask_used).sum(axis=0).astype(np.int64)

    with np.errstate(invalid='ignore', divide='ignore'):
        tp_rate = np.where(target_elig > 0, tp_hits / target_elig, np.nan)
        fp_rate = np.where(nontarget_elig > 0, fp_hits / nontarget_elig, np.nan)
        loading = img1_used.mean(axis=0).astype(float)   # init-grid quantity

    def _opt_float_list(arr):
        return [None if (isinstance(x, float) and x != x) else float(x)
                for x in arr.tolist()]

    out = {
        'x': site_xy[:, 1].tolist(),   # image-x
        'y': site_xy[:, 0].tolist(),   # image-y
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
    if loading_on_grid:
        # Same grid: loading shares the main per-site grid.
        out['loading_rate'] = _opt_float_list(loading)
    else:
        # Cross-grid: the main map is the TARGET grid, but loading lives on the
        # INIT (loading) grid. Carry it with its own coords so the dashboard's
        # loading panel renders on the init grid (survival/FP stay on target).
        out['loading_rate'] = None
        if (init_site_xy is not None
                and init_site_xy.shape[0] == loading.shape[0]):
            out['loading_init'] = _opt_float_list(loading)
            out['loading_x'] = init_site_xy[:, 1].tolist()
            out['loading_y'] = init_site_xy[:, 0].tolist()
    return out


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

            # Per-shot rearrangement step count (for per-step transit
            # distance = total / nsteps). Regular int column when present.
            nsteps_col = None
            if 'nsteps' in g:
                try:
                    nsteps_col = np.asarray(g['nsteps'][:], dtype=float).ravel()
                except (TypeError, ValueError):
                    nsteps_col = None

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

            # Active target set per shot (the full set the protocol fills,
            # e.g. the every_other checkerboard). Surfaced only inside
            # diag_json, so read it there regardless of schema version — this
            # is what makes target-aware TP/FP work from the diag alone, with
            # no grid sidecar.
            target_site_lists = [np.empty(0, dtype=np.int64)] * n_rows
            if 'diag_json' in g:
                _tmp = []
                for v in g['diag_json'][:]:
                    s = v.decode() if isinstance(v, bytes) else str(v)
                    try:
                        _d = (json.loads(s) or {}).get('diag') or {}
                        _tmp.append(np.asarray(
                            _d.get('target_site_indices') or [],
                            dtype=np.int64).ravel())
                    except (ValueError, TypeError):
                        _tmp.append(np.empty(0, dtype=np.int64))
                if len(_tmp) == n_rows:
                    target_site_lists = _tmp

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
                nsteps_i = None
                if nsteps_col is not None and i < nsteps_col.size \
                        and np.isfinite(nsteps_col[i]):
                    nsteps_i = int(nsteps_col[i])
                rows.append({
                    'seq_id':           int(seq_ids[i]),
                    'loaded_paired':    lp.tolist(),
                    'target_paired':    tp.tolist(),
                    'target_site_indices': (target_site_lists[i].tolist()
                                            if i < len(target_site_lists) else []),
                    'init_xy':          init_xy,
                    'target_xy':        target_xy,
                    'nsteps':           nsteps_i,
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

def _detect_spots_focus(img, *, half=6, min_dist=7, k_sigma=5.0, max_spots=500):
    """Calibration-free spot detection + per-spot focus measures on one image.

    No grid / no thresholds: find bright local maxima directly, then for each
    measure an RMS radius (spot width), peak-above-local-background, and a
    peak/background contrast. Returns (radius_px, peak, contrast) arrays, one
    entry per detected spot (empty arrays when nothing is found)."""
    from scipy import ndimage
    img = np.asarray(img, dtype=np.float64)
    empty = (np.array([]), np.array([]), np.array([]))
    if img.ndim != 2 or img.size == 0:
        return empty
    bg = float(np.median(img))
    mad = float(np.median(np.abs(img - bg))) * 1.4826
    if mad <= 0:
        mad = float(img.std()) or 1.0
    thresh = bg + k_sigma * mad
    mx = ndimage.maximum_filter(img, size=min_dist)
    peaks = (img == mx) & (img > thresh)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return empty
    if ys.size > max_spots:                       # keep the brightest, bound cost
        order = np.argsort(img[ys, xs])[::-1][:max_spots]
        ys, xs = ys[order], xs[order]
    H, W = img.shape
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rho2 = (yy ** 2 + xx ** 2).astype(np.float64)
    radii, peaksv, contrasts = [], [], []
    for y, x in zip(ys, xs):
        if y - half < 0 or y + half >= H or x - half < 0 or x + half >= W:
            continue
        win = img[y - half:y + half + 1, x - half:x + half + 1]
        border = np.concatenate([win[0, :], win[-1, :], win[:, 0], win[:, -1]])
        lbg = float(np.median(border))
        sub = win - lbg
        sub[sub < 0] = 0.0
        tot = float(sub.sum())
        if tot <= 0:
            continue
        radii.append(float(np.sqrt((sub * rho2).sum() / tot)))
        pk = float(win.max() - lbg)
        peaksv.append(pk)
        contrasts.append(pk / lbg if lbg > 0 else pk)
    return np.array(radii), np.array(peaksv), np.array(contrasts)


def _focus_metrics_from_images(scan_dir, scan, scan_params_full, seq_ids,
                               *, mat_path=None, max_shots_per_point=24):
    """Calibration-free seq-specific focus metrics vs the swept parameter
    (LoadingDefocusScan). Computed straight from the raw camera images with
    NO reliance on the detection grid or per-site thresholds: for each scan
    point we average up to ``max_shots_per_point`` frames, detect the array
    spots as bright local maxima, and measure each spot's shape.

    Every per-spot quantity is averaged over the DETECTED spots, so the number
    of loaded spots (which changes with defocus) does not confound the focus
    measure. Used to pick the best SLM defocus (Zernike).

    Metrics (each with a real unit; the dashboard shows units on hover):
      * ``spot_width``    — median spot RMS radius (px). LOWER is better (a
        tighter focus = smaller spot). Shape-based and brightness-independent
        -> the cleanest focus measure.
      * ``spot_peak``     — median peak counts above local background. Higher
        near focus, but scales with how often a site is loaded; read it
        alongside spot_width.
      * ``spot_contrast`` — median peak/background ratio. A calibration-free
        discrimination proxy: how cleanly spots stand out from the background.
        Higher is better.

    Cached to ``<scan_dir>/focus_metrics.json``. Returns None for
    non-sweep / non-single-image / no-image scans."""
    scan_dir = Path(scan_dir)
    cache = scan_dir / FOCUS_METRICS_JSON
    if cache.is_file():
        try:
            with open(cache) as f:
                return json.load(f)
        except Exception:
            pass
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    if num_images != 1 or seq_ids is None:
        return None
    h5_path = _scan_data_h5(scan_dir)
    if h5_path is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()

    params_arr = np.asarray(scan.get('Params', [])).ravel().astype(int)
    idx = seq_ids - 1
    if params_arr.size:
        idx = np.clip(idx, 0, params_arr.size - 1)
        per_shot_param0 = params_arr[idx] - 1
    else:
        per_shot_param0 = idx
    if per_shot_param0.size == 0:
        return None
    n_params = int(per_shot_param0.max()) + 1
    if n_params < 2:
        return None

    spot_width = [None] * n_params
    spot_peak = [None] * n_params
    spot_contrast = [None] * n_params
    n_spots = [0] * n_params
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(h5_path, 'r') as f:
            d = f.get('imgs')
            if d is None or d.ndim != 3:
                return None
            n_avail = min(d.shape[0], per_shot_param0.size)
            p0 = per_shot_param0[:n_avail]
            for p in range(n_params):
                rows = np.nonzero(p0 == p)[0]
                if rows.size == 0:
                    continue
                sel = np.unique(rows[:max_shots_per_point])
                mean_img = d[sel].astype(np.float64).mean(axis=0)
                r, pk, ctr = _detect_spots_focus(mean_img)
                if r.size == 0:
                    continue
                n_spots[p] = int(r.size)
                spot_width[p] = float(np.median(r))
                spot_peak[p] = float(np.median(pk))
                spot_contrast[p] = float(np.median(ctr))
    except (OSError, ValueError) as ex:
        logger.warning('focus-from-images failed (%s): %s', scan_dir, ex)
        return None

    if all(v is None for v in spot_width):
        return None

    sp = np.asarray(scan_params_full)
    if sp.ndim == 1 and sp.size == n_params:
        x = sp.astype(float).tolist()
    elif sp.ndim == 2 and sp.shape[0] == n_params and sp.shape[1] == 1:
        x = sp[:, 0].astype(float).tolist()
    else:
        x = list(range(1, n_params + 1))
    try:
        sweep = _build_sweep(scan, sp, mat_path=mat_path)
        cols = sweep.get('cols') or []
        x_label = cols[0] if cols else 'scan point'
    except Exception:
        x_label = 'scan point'

    result = {
        'type': 'focus_metrics',
        'source': 'images',
        'calibration_free': True,
        'x': x,
        'x_label': x_label,
        'n_spots': n_spots,
        'metrics': {
            'spot_width':    {'values': spot_width,
                              'label': 'spot RMS radius',
                              'unit': 'px', 'higher_better': False},
            'spot_peak':     {'values': spot_peak,
                              'label': 'spot peak (above bg)',
                              'unit': 'counts', 'higher_better': True},
            'spot_contrast': {'values': spot_contrast,
                              'label': 'spot/bg contrast',
                              'unit': 'ratio', 'higher_better': True},
        },
    }
    try:
        with open(cache, 'w') as f:
            json.dump(result, f)
    except OSError:
        pass
    return result


# A 556 push-out SURVIVAL scan is a TRAP-DEPTH (|mj|=1) measurement only when the
# measured line sits RED of the mj=0 resonance f0 (the trap-shifted feature). This
# gate is what lets the panel fire on ANY such scan by its physics -- not its name
# -- while skipping the mj=0 calibration (line AT f0) and the field-shifted
# Rydberg / Autler-Townes scans (line BLUE of f0). Margin > the mj=0 linewidth.
_TRAP_DEPTH_F0_MARGIN_HZ = 0.15e6


def _trap_depth_from_lightshift(delta_nu):
    """556 |mj|=1-vs-mj=0 excited-state light-shift difference (Hz) -> ground-
    state trap depth (uK). ``delta_nu = 2*(f0 - f_site)``. Ported verbatim from
    the trap-depth-feedback campaign (``_feedback47x47/fit_trap_depths.py``): a
    532 nm tweezer, mj1=1, mj0=0, theta=0. Depth is LINEAR in ``delta_nu``, so the
    overall scale cancels in the CV; the absolute uK only sets the histogram axis.
    Vectorised over sites."""
    h = 6.62607015e-34
    kB = 1.380649e-23
    mj1, mj0, theta = 1, 0, np.deg2rad(0.0)
    alpha_s, alpha_t, alpha_g = 22.4, -7.6, 37.9
    T = (1 - 3 * np.cos(theta) ** 2) / 2
    alpha_e1 = alpha_s - alpha_t * T * (3 * mj1 ** 2 - 2)
    alpha_e0 = alpha_s - alpha_t * T * (3 * mj0 ** 2 - 2)
    intensity = -4 * np.asarray(delta_nu, dtype=float) / (alpha_e1 - alpha_e0)
    depth_Hz = 0.25 * abs(alpha_g) * intensity
    return h * depth_Hz / kB * 1e6   # K -> uK


def _is_pushout_green_freq(name) -> bool:
    """True if a swept-axis dotted name is the 556 push-out green frequency
    (``Pushout.Green.Freq``), tolerant of punctuation/case."""
    s = re.sub(r'[^a-z0-9]', '', str(name).lower())
    return ('pushout' in s) and ('green' in s) and ('freq' in s)


def _trap_depth_from_pushout(scan_dir, scan, scan_params_full, logic1_full,
                             logic2_full, seq_ids, *, sweep_all=None):
    """Seq-specific |mj|=1 trap-depth histogram + CV for ANY 556 push-out
    survival scan -- detected by the swept axis + the trap-shifted line, NOT by
    the scan being named ``Spectrum556Scan``.

    Mirrors the trap-depth-feedback campaign
    (``_feedback47x47/fit_trap_depths.py``): a per-site Lorentzian PEAK on
    ``1 - survival`` vs the push-out green frequency gives each site's |mj|=1
    center ``f_i``; the differential light shift converts it to a trap depth
    ``d_i`` (``delta_nu = 2*(f0 - f_i)``), with ``f0`` = the mj=0 resonance from
    the run's expConfig snapshot (``Resonance556mj0Freq``). The array-uniformity
    headline is ``CV = std/mean`` over the good-fit sites.

    Returns the ``seq_specific`` dict (``type='trap_depth'``) or ``None`` when the
    scan isn't a |mj|=1 556 push-out survival scan: a non-1-D / wrong sweep, no
    survival image, no f0 in the snapshot, the line not red of f0 (the mj=0
    calibration / field-shifted Rydberg / Autler-Townes scans), or too few good
    fits. Cached to ``trap_depth.json`` keyed by shot count -- the per-site fits
    are the expensive, filter-independent part."""
    # --- cheap gates: survival scan, same-grid, 1-D push-out-freq sweep, f0 ----
    try:
        num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    except (TypeError, ValueError, IndexError):
        num_images = 1
    if num_images < 2 or logic2_full is None:
        return None
    if not (getattr(logic1_full, 'ndim', 0) == 3
            and getattr(logic2_full, 'ndim', 0) == 3):
        return None
    if logic1_full.shape[0] != logic2_full.shape[0]:
        return None   # cross-grid (rearrangement) -- not a per-site survival map
    cols = (sweep_all or {}).get('cols') or []
    if len(cols) != 1 or not _is_pushout_green_freq(cols[0]):
        return None
    ec = scan.get('expConfig')
    f0 = None
    if isinstance(ec, dict):
        try:
            f0 = float(ec.get('Resonance556mj0Freq'))
        except (TypeError, ValueError):
            f0 = None
    if not f0 or not np.isfinite(f0) or f0 <= 0:
        return None

    sp = np.asarray(scan_params_full, dtype=float).ravel()
    n_sites, n_params = int(logic1_full.shape[0]), int(logic1_full.shape[1])
    if sp.size != n_params or n_params < 5 or n_sites < 4:
        return None

    # --- cache (keyed by recorded shot count; refit only when more shots land) -
    try:
        shots = int(len(np.asarray(seq_ids).ravel())) if seq_ids is not None \
            else int(n_params * logic1_full.shape[2])
    except (TypeError, ValueError):
        shots = int(n_params * logic1_full.shape[2])
    cache = Path(scan_dir) / TRAP_DEPTH_JSON
    if cache.is_file():
        try:
            with open(cache) as f:
                cached = json.load(f)
            if (cached.get('_version') == _TRAP_DEPTH_CACHE_VERSION
                    and cached.get('n_shots') == shots):
                return cached.get('result')
        except (OSError, ValueError):
            pass

    from yb_analysis.analysis.fitting import (
        fit_lorentzian, fit_lorentzian_site_resolved)

    # --- array-averaged fit: the |mj|=1 gate + context numbers ----------------
    p11_mean, p11_sem = prob11(logic1_full, logic2_full)
    afit = fit_lorentzian(sp, 1.0 - p11_mean, yerr=p11_sem, mode='peak')
    if afit is None:
        return None
    a_center = float(afit['center'])
    a_r2 = float(afit['r_squared'])
    # Gate: the measured line must sit RED of f0 (the trap-shifted |mj|=1 feature)
    # and the array fit must be real. Excludes mj=0 (line at f0) + the field-
    # shifted Rydberg / Autler-Townes scans (line blue of f0).
    if not (np.isfinite(a_center) and a_center < f0 - _TRAP_DEPTH_F0_MARGIN_HZ
            and a_r2 >= 0.3):
        return None

    # --- per-site Lorentzian peak -> per-site trap depth ----------------------
    p11_sr, p11_sem_sr = prob11_site_resolved(logic1_full, logic2_full)
    centers, widths, params, fits = fit_lorentzian_site_resolved(
        sp, 1.0 - p11_sr, p11_sem_sr, mode='peak')
    r2 = np.array([f['r_squared'] if f else np.nan for f in fits])
    amp = params[:, 1]   # peak amplitude A
    depth_uK = _trap_depth_from_lightshift(2.0 * (f0 - centers))

    # Quality mask (matches fit_trap_depths.py): finite, R^2>0.5, center inside
    # the swept window, peak amp > 0.05, finite positive width, positive depth.
    fmin, fmax = float(sp.min()), float(sp.max())
    good = (np.isfinite(centers) & np.isfinite(depth_uK) & (r2 > 0.5)
            & (centers > fmin) & (centers < fmax) & (amp > 0.05)
            & np.isfinite(widths) & (widths > 0) & (depth_uK > 0))
    n_good = int(good.sum())
    if n_good < 5:
        return None
    dg = depth_uK[good]
    mean_depth = float(np.mean(dg))
    std_depth = float(np.std(dg))
    cv = (std_depth / mean_depth) if mean_depth > 0 else float('nan')

    result = {
        'type': 'trap_depth',
        'source': 'mj1_lightshift',
        'x_label': str(cols[0]),
        'f0_MHz': f0 / 1e6,
        'cv': cv,
        'cv_pct': 100.0 * cv,
        'mean_depth_uK': mean_depth,
        'std_depth_uK': std_depth,
        'min_depth_uK': float(dg.min()),
        'max_depth_uK': float(dg.max()),
        'n_good': n_good,
        'n_sites': n_sites,
        'good_frac': n_good / max(n_sites, 1),
        'r2_median': float(np.nanmedian(r2)),
        'n_reps_max': int(logic1_full.shape[2]),
        'n_shots': shots,
        'depths_uK': [float(v) for v in dg],
        'gauss': {'mu': mean_depth, 'sigma': std_depth},
        'array_fit': {
            'center_MHz': a_center / 1e6,
            'fwhm_MHz': abs(float(afit['width'])) / 1e6,
            'r_squared': a_r2,
            'shift_MHz': (f0 - a_center) / 1e6,
        },
    }
    try:
        with open(cache, 'w') as f:
            json.dump({'_version': _TRAP_DEPTH_CACHE_VERSION,
                       'n_shots': shots, 'result': result}, f)
    except OSError:
        pass
    return result


def _affine_scale_for_scan(scan_id: Optional[str]):
    """camera-px-per-knm-px scale of the affine *in effect for this run*.

    There is one global SLM(knm)->science-camera affine; we want the value
    THAT run committed/used, not whatever the current global affine is now
    (the operator's note). We search current+history for the entry whose
    ``last_scan_id`` matches this scan; else the most recent entry committed
    at or before it; else the current affine. Returns ``(scale, provenance)``
    where provenance is 'run' | 'nearest' | 'current' | None.
    """
    try:
        from yb_analysis.analysis import affine_transform as _aff
        data = _aff._read()
    except Exception as ex:
        logger.debug('affine read for svd scale failed: %s', ex)
        return None, None

    def _scale_of(e):
        if not isinstance(e, dict):
            return None
        sx, sy = e.get('scale_x'), e.get('scale_y')
        try:
            if sx and sy and float(sx) > 0 and float(sy) > 0:
                return float(np.sqrt(float(sx) * float(sy)))
            if e.get('det'):
                return float(np.sqrt(abs(float(e['det']))))
        except (TypeError, ValueError):
            return None
        return None

    entries = []
    if data.get('current'):
        entries.append(data['current'])
    entries.extend(data.get('history') or [])
    sid = str(scan_id) if scan_id else None
    if sid:
        for e in entries:
            if str((e or {}).get('last_scan_id')) == sid:
                return _scale_of(e), 'run'
        cand = [e for e in entries
                if (e or {}).get('last_scan_id')
                and str(e['last_scan_id']) <= sid]
        if cand:
            cand.sort(key=lambda e: str(e.get('last_scan_id')))
            return _scale_of(cand[-1]), 'nearest'
    if data.get('current'):
        return _scale_of(data['current']), 'current'
    return None, None


def _seq_to_nsteps_map(scan: dict, scan_params_full, sweep_all,
                       seq_ids) -> dict:
    """Map seq_id -> nsteps from the swept 'nsteps' axis.

    The diag's per-shot ``nsteps`` column is frequently NaN, but for scans
    that sweep ``rearrange_kwargs.nsteps`` the value is recoverable from the
    swept parameter itself. Returns ``{}`` when there's no nsteps axis.
    """
    try:
        cols = (sweep_all or {}).get('cols') or []
        axis = next((i for i, c in enumerate(cols)
                     if 'nstep' in str(c).lower()), None)
        if axis is None or seq_ids is None:
            return {}
        sp = np.asarray(scan_params_full)
        if sp.ndim == 1:
            sp = sp.reshape(-1, 1)
        if sp.size == 0 or axis >= sp.shape[1]:
            return {}
        params_arr = np.asarray(scan.get('Params', [])).ravel().astype(int)
        sids = np.asarray(seq_ids).ravel().astype(int)
        out: dict = {}
        for sid in sids:
            i = int(sid) - 1
            if params_arr.size:
                if not (0 <= i < params_arr.size):
                    continue
                p0 = int(params_arr[i]) - 1
            else:
                p0 = i
            if 0 <= p0 < sp.shape[0]:
                v = float(sp[p0, axis])
                if np.isfinite(v) and v > 0:
                    out[int(sid)] = int(round(v))
        return out
    except Exception as ex:
        logger.debug('_seq_to_nsteps_map failed: %s', ex)
        return {}


def _survival_vs_distance(paths_info: Optional[dict],
                          bundle: Optional[dict],
                          scan: dict,
                          *,
                          scan_id: Optional[str] = None,
                          seq_nsteps: Optional[dict] = None,
                          n_bins: int = 12) -> Optional[dict]:
    """Lab-side survival-vs-transit-distance aggregate (Phase 5.5 Track B).

    For each rearrangement pair ``i`` in each shot, bin by transit distance
    ``||target_xy[i] - init_xy[i]||`` and compute per-bin survival = fraction
    of pairs whose target site held an atom in img2. Survival at the target
    is read from the lab's OWN logic2 (``Scan.initGridLocationsX/Y`` nearest-
    neighbour of ``target_xy``), so no SLM-computed value is consumed — this
    replaces the transitional copy-through of the SLM's survival_vs_distance.

    Returns the curve dict (see the ``survival_vs_distance`` shape in
    ``analyze_scan_dir``'s docstring), or:
      * ``None`` when prerequisites are missing (no paths / no img2 / no
        lab grid), so the dashboard hides the panel; or
      * ``{'skipped_reason': 'lattice_mismatch'}`` when target coords don't
        resolve to any lab site (distinct lattices — case 2 not implemented).
    """
    paths_per_shot = (paths_info or {}).get('paths_per_shot') or []
    paths_frame = (paths_info or {}).get('paths_frame')
    if not paths_per_shot or bundle is None:
        return None

    gx, gy = _site_grid_xy(scan)
    if not gx or not gy:
        return None
    site_xy = np.column_stack([np.asarray(gy, dtype=float),
                               np.asarray(gx, dtype=float)])  # (n_sites, 2) (y,x)
    if site_xy.size == 0:
        return None

    # Lab img2 (per-shot, per-site), same extraction as the TP path.
    raw1 = bundle.get('logicals_img1')
    raw2 = bundle.get('logicals_img2')
    raw = bundle.get('logicals')
    num_images = int(np.asarray(scan.get('NumImages', 1)).flat[0]) or 1
    if raw1 is not None and raw2 is not None:
        img2 = np.asarray(raw2, dtype=np.uint8)
    elif raw is not None:
        a = np.asarray(raw, dtype=np.uint8)
        img2 = a[(num_images - 1)::num_images] if num_images >= 2 else a
    else:
        return None
    # Target grid: the lab (init) grid for a same-pattern run; for a
    # cross-pattern run (initial != final pattern) img2 lives on the
    # differently-sized TARGET grid -- recover its camera coords by affine-
    # mapping the target pattern's registry knm (as the per-site map does), so
    # transit distance ||target - init|| works across the two grids. init
    # endpoints (loaded_paired) index site_xy; target endpoints (target_paired,
    # + img2) index tgt_xy.
    cross_grid = bool(img2.size) and img2.shape[1] != site_xy.shape[0]
    if cross_grid:
        tgt_xy = _target_grid_camera_xy(scan, img2.shape[1])
        if tgt_xy is None:
            return None
    else:
        tgt_xy = site_xy
    if img2.size == 0 or img2.shape[1] != tgt_xy.shape[0]:
        return None
    n_tgt = tgt_xy.shape[0]

    seq_ids = bundle.get('seq_ids')
    if seq_ids is None:
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64).ravel()
    n_shots = min(len(seq_ids), img2.shape[0])
    seq_ids = seq_ids[:n_shots]
    img2 = img2[:n_shots]

    by_seq = {}
    for entry in paths_per_shot:
        if isinstance(entry, dict) and entry.get('seq_id') is not None:
            by_seq[int(entry['seq_id'])] = entry

    # Typical site spacing (median per-site nearest-neighbour distance) —
    # the matched-site tolerance and lattice-mismatch test both key off it.
    ssub = site_xy[: min(site_xy.shape[0], 1000)]
    d2s = ((ssub.reshape(-1, 1, 2) - ssub.reshape(1, -1, 2)) ** 2).sum(axis=2)
    d2s[np.arange(d2s.shape[0]), np.arange(d2s.shape[0])] = np.inf
    spacing = max(float(np.median(np.sqrt(d2s.min(axis=1)))), 1.0) \
        if d2s.size else 1.0
    tol = 2.0 * spacing

    n_sites = site_xy.shape[0]
    distances = []        # per-pair TOTAL transit distance ||target-init||
    per_step_dists = []   # per-pair total / nsteps (NaN when nsteps unknown)
    survivals = []        # per-pair survival flag (0/1)
    n_total_pairs = 0
    n_unmatched = 0
    used_lab_grid = False   # True when distances came from lab-grid indices

    for k in range(n_shots):
        entry = by_seq.get(int(seq_ids[k]))
        if entry is None:
            continue
        ns = entry.get('nsteps')
        if ns is None and seq_nsteps:        # diag column NaN -> swept value
            ns = seq_nsteps.get(int(seq_ids[k]))
        ns = float(ns) if (ns is not None and ns) else None   # >0 divisor
        # PRIMARY (sidecar-free): loaded_paired / target_paired are bit indices
        # in the shared grid order (== lab grid order), so the source + target
        # sites, their lab coords (-> transit distance), and the img2 survival
        # all come from the lab grid + the diag pairing. No sidecar coords.
        lpaired = entry.get('loaded_paired') or []
        tpaired = entry.get('target_paired') or []
        n_idx = min(len(lpaired), len(tpaired))
        if n_idx:
            for i in range(n_idx):
                si = int(lpaired[i]); ti = int(tpaired[i])
                # si -> init/lab grid (site_xy); ti -> target grid (tgt_xy + img2).
                if not (0 <= si < n_sites and 0 <= ti < n_tgt):
                    continue
                n_total_pairs += 1
                used_lab_grid = True
                dist = float(np.hypot(tgt_xy[ti, 0] - site_xy[si, 0],
                                      tgt_xy[ti, 1] - site_xy[si, 1]))
                distances.append(dist)
                per_step_dists.append(dist / ns if (ns and ns > 0)
                                      else float('nan'))
                survivals.append(int(img2[k, ti]))
            continue
        # FALLBACK (legacy entries with sidecar coords but no indices):
        # init_xy/target_xy -> nearest lab site.
        init_xy = entry.get('init_xy') or []
        target_xy = entry.get('target_xy') or []
        n_pairs = min(len(init_xy), len(target_xy))
        for i in range(n_pairs):
            s = init_xy[i]
            t = target_xy[i]
            if not (isinstance(s, (list, tuple)) and len(s) == 2
                    and isinstance(t, (list, tuple)) and len(t) == 2):
                continue
            if any(v is None for v in (*s, *t)):
                continue
            n_total_pairs += 1
            t_arr = np.asarray(t, dtype=float)
            d2 = ((site_xy - t_arr.reshape(1, 2)) ** 2).sum(axis=1)
            j = int(d2.argmin())
            if np.sqrt(d2[j]) > tol:
                n_unmatched += 1
                continue
            dist = float(np.hypot(t_arr[0] - float(s[0]), t_arr[1] - float(s[1])))
            distances.append(dist)
            per_step_dists.append(dist / ns if (ns and ns > 0) else float('nan'))
            survivals.append(int(img2[k, j]))

    if n_total_pairs == 0:
        return None
    # Every pair's target fell outside the lab lattice -> distinct lattices.
    if not distances:
        return {'skipped_reason': 'lattice_mismatch'}

    distances = np.asarray(distances, dtype=float)
    per_step_dists = np.asarray(per_step_dists, dtype=float)
    survivals = np.asarray(survivals, dtype=float)

    # Lab-grid (diag-pairing) distances are in camera pixels; the legacy
    # sidecar-coord path keeps its frame-derived units. Convert to SLM
    # computational (knm) pixels via the per-run affine scale.
    units = ('camera_pixels' if (used_lab_grid or paths_frame == 'camera_bitorder')
             else 'knm_pixels' if paths_frame == 'knm_native'
             else 'unknown')
    cam_per_knm, scale_src = (None, None)
    if units == 'camera_pixels':
        cam_per_knm, scale_src = _affine_scale_for_scan(scan_id)
    out_units = 'knm_pixels' if (cam_per_knm and cam_per_knm > 0) else units

    def _bin_curve(dist_arr, surv_arr):
        """Bin (distance, survival 0/1) into n_bins; knm-convert if scaled."""
        mask = np.isfinite(dist_arr)
        d = dist_arr[mask]
        s = surv_arr[mask]
        if d.size == 0:
            return None
        if cam_per_knm and cam_per_knm > 0:
            d = d / cam_per_knm
        dmin, dmax = float(d.min()), float(d.max())
        if dmax <= dmin:
            dmax = dmin + 1.0
        edges = np.linspace(dmin, dmax, int(n_bins) + 1)
        idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
        centers, mean, sem, counts = [], [], [], []
        for b in range(n_bins):
            m = idx == b
            n = int(m.sum())
            centers.append(float((edges[b] + edges[b + 1]) / 2.0))
            counts.append(n)
            if n:
                p = float(s[m].mean())
                mean.append(p)
                sem.append(float(np.sqrt(max(p * (1 - p), 0.0) / n)))
            else:
                mean.append(None)
                sem.append(None)
        return {
            'bins': edges.tolist(), 'centers': centers,
            'survival_mean': mean, 'survival_sem': sem,
            'n_pairs_per_bin': counts, 'distance_units': out_units,
            'n_pairs': int(d.size),
        }

    total_curve = _bin_curve(distances, survivals)
    if total_curve is None:
        return None
    # Per-step curve (total / nsteps). None when no shot had a usable
    # nsteps (e.g. legacy diag without the column).
    per_step_curve = _bin_curve(per_step_dists, survivals)

    result = dict(total_curve)
    result['cam_px_per_knm_px'] = cam_per_knm
    result['cam_px_per_knm_px_source'] = scale_src   # 'run'|'nearest'|'current'
    result['n_total_pairs'] = int(n_total_pairs)
    result['n_unmatched'] = int(n_unmatched)
    result['per_step'] = per_step_curve
    result['has_per_step'] = per_step_curve is not None
    return result


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


# Per-shot numeric diag columns surfaced for the per-iteration overlay +
# the rearrangement scatter. (column, label, unit). Only columns actually
# present in slm_diag.h5 are emitted; each carries its own unit so the
# dashboard can put it on its own (right-side) axis.
_DIAG_SERIES_SPEC = [
    ('total_ms',         'Total time',       'ms'),
    ('compute_total_ms', 'Compute total',    'ms'),
    ('paced_total_ms',   'Paced total',      'ms'),
    ('phase_load_ms',    'Phase load',       'ms'),
    ('lock_check_ms',    'Lock check',       'ms'),
    ('bits_decode_ms',   'Bits decode',      'ms'),
    ('step_period_ms',   'Step period',      'ms'),
    ('aborted_at_ms',    'Aborted at',       'ms'),
    ('n_loaded',         'N loaded',         'atoms'),
    ('n_loaded_total',   'N loaded (total)', 'atoms'),
    ('n_dropped',        'N dropped',        'atoms'),
    ('n_total_sites',    'N total sites',    'sites'),
    ('n_sites_model',    'N sites (model)',  'sites'),
    ('nsteps',           'N steps',          'steps'),
    ('retry_count',      'Retry count',      'count'),
    ('aborted_at_frame', 'Aborted at frame', 'frame'),
]


def _per_shot_diag_series(diag_path, kept_seq_ids) -> dict:
    """Per-shot numeric diag columns aligned to the per-iteration kept shots.

    ``kept_seq_ids`` is the seq_id of each kept shot (in time order). Returns
    ``{col: {'label', 'unit', 'values': [float|None per kept shot]}}`` for
    every spec column present + numeric in slm_diag.h5. Rearrangement-only
    (no diag → empty dict)."""
    if diag_path is None or not Path(diag_path).is_file():
        return {}
    try:
        import h5py
        with h5py.File(diag_path, 'r') as f:
            g = f['/diag'] if '/diag' in f else f
            if 'seq_id' not in g:
                return {}
            sids = np.asarray(g['seq_id'][:], dtype=np.int64).ravel()
            # seq_id -> row index (last wins if duplicated).
            row_of = {int(s): i for i, s in enumerate(sids)}
            series = {}
            for col, label, unit in _DIAG_SERIES_SPEC:
                if col not in g:
                    continue
                try:
                    vals = np.asarray(g[col][:], dtype=float).ravel()
                except (TypeError, ValueError):
                    continue
                if vals.shape[0] != sids.shape[0]:
                    continue
                out_vals = []
                for sid in kept_seq_ids:
                    i = row_of.get(int(sid))
                    v = vals[i] if i is not None else np.nan
                    out_vals.append(None if not np.isfinite(v) else float(v))
                # Skip all-None columns (column absent for these shots).
                if any(v is not None for v in out_vals):
                    series[col] = {'label': label, 'unit': unit,
                                   'values': out_vals}
            return series
    except (OSError, KeyError) as ex:
        logger.warning('_per_shot_diag_series: %s: %s', diag_path, ex)
        return {}


def _per_iteration_time_order(scan: dict,
                              bundle: dict,
                              seq_ids,
                              param_mask: Optional[np.ndarray],
                              filters: Optional[dict],
                              *,
                              paths_info: Optional[dict] = None,
                              diag_path=None) -> dict:
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

    img1f = img1[keep].astype(bool)
    img2f = img2[keep].astype(bool) if img2 is not None else None
    shot_index = (np.where(keep)[0] + 1).tolist()
    kept_seq_ids = seq_ids_arr[:n_shots][keep]

    # Rearrangement-correct FP base: atoms moved INTO target sites are NOT
    # false positives, so exclude per-shot target sites from the empty-site
    # FP denominator/numerator. Only when target info is available (= a
    # rearrangement run, via paths_per_shot target_paired/target_site_indices,
    # which index the logicals directly — no grid needed); otherwise the
    # plain "empty in img1 → occupied in img2" FP is correct.
    # Cross-grid rearrangement: img1 (loading grid) and img2 (target grid)
    # have different site counts, so the per-site survival (img1 & img2) and
    # the img1-emptiness FP are undefined. survival is left None (the
    # target-aware per-shot TP override in analyze_scan_dir fills it); FP is
    # measured over the img2 (target-grid) NON-target sites.
    cross_grid = (img2f is not None and img1f.size and img2f.size
                  and img1f.shape[1] != img2f.shape[1])
    # The target mask lives in the grid the targets index: img2 for cross-grid
    # runs (target_paired -> img2), else the shared img1/img2 grid.
    tmask_width = (img2f.shape[1] if (cross_grid and img2f is not None)
                   else (img1f.shape[1] if img1f.size else 0))
    target_mask_f = None
    fp_is_rearrange = False
    paths_list = (paths_info or {}).get('paths_per_shot') or []
    if paths_list and img1f.size and tmask_width:
        by_seq = {}
        for e in paths_list:
            if isinstance(e, dict) and e.get('seq_id') is not None:
                by_seq[int(e['seq_id'])] = e
        tmask = np.zeros((img1f.shape[0], tmask_width), dtype=bool)
        for r, sid in enumerate(kept_seq_ids):
            e = by_seq.get(int(sid))
            if e is None:
                continue
            idx = _entry_target_sites(e, None, tmask_width, 0.0)  # index path
            if idx.size:
                tmask[r, idx[(idx >= 0) & (idx < tmask_width)]] = True
                fp_is_rearrange = True
        if fp_is_rearrange:
            target_mask_f = tmask

    with np.errstate(invalid='ignore', divide='ignore'):
        loaded = img1f.sum(axis=1)
        n_sites_per_shot = img1f.shape[1] if img1f.size else 0
        loaded_frac = (loaded / max(1, n_sites_per_shot)).astype(float)
        survival = fp = None
        if img2f is not None and not cross_grid:
            survived = (img1f & img2f).sum(axis=1)
            survival = np.where(loaded > 0, survived / loaded, np.nan)
            empty_base = (~img1f) & (~target_mask_f) if target_mask_f is not None \
                else ~img1f
            empty  = empty_base.sum(axis=1)
            falsep = (img2f & empty_base).sum(axis=1)
            fp     = np.where(empty > 0, falsep / empty, np.nan)
        elif img2f is not None and cross_grid and target_mask_f is not None:
            # survival left None -> filled by the target-aware per-shot TP
            # override. FP = spurious atoms at img2 sites NOT in the target set.
            base   = ~target_mask_f
            empty  = base.sum(axis=1)
            falsep = (img2f & base).sum(axis=1)
            fp     = np.where(empty > 0, falsep / empty, np.nan)

    out = {
        'shot_index':    shot_index,
        # Frames per shot in /imgs (interleaved). Lets the dashboard map a
        # clicked shot back to its camera frame rows for the shot-image popup:
        # row = (shot_index - 1) * num_images + frame.
        'num_images':    int(num_images),
        'loaded_frac':   [float(v) for v in loaded_frac],
        'survival_frac': None if survival is None
                         else [None if not np.isfinite(v) else float(v)
                               for v in survival],
        'fp_frac':       None if fp is None
                         else [None if not np.isfinite(v) else float(v)
                               for v in fp],
        # 'rearrange' = FP excludes target sites (atoms moved into targets
        # aren't false positives); 'all_empty' = plain empty→occupied FP.
        'fp_source':     'rearrange' if target_mask_f is not None else 'all_empty',
        # Per-shot numeric diag columns (ms timings, counts, …) aligned to
        # shot_index — drives the per-iteration right-axis overlay + the
        # rearrangement scatter. Empty for non-rearrangement runs.
        'diag_series':   _per_shot_diag_series(diag_path, kept_seq_ids),
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


# ---------------------------------------------------------------------------
# Phase 5.5 Track E — SLM vs lab bitstring divergence diagnostic
# ---------------------------------------------------------------------------


def _coerce_bitstring(v) -> Optional[np.ndarray]:
    """Coerce one shot's bits into a 1-D uint8 0/1 array, or None.

    Accepts:
      * '0'/'1' character strings (the SLM ``received_bits`` form),
      * bytes of '0'/'1' chars,
      * list / tuple / ndarray of truthy values (lab logicals),
    Returns ``None`` for empty / unparseable input (e.g. a legacy shot
    with no ``received_bits``) so callers can skip it.
    """
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode('ascii', errors='replace')
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Only '0'/'1' chars are meaningful; anything else -> skip.
        if any(c not in '01' for c in s):
            return None
        return np.frombuffer(s.encode('ascii'), dtype=np.uint8) - ord('0')
    # array-like of truthy values
    try:
        arr = np.asarray(v).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return (arr != 0).astype(np.uint8)


def read_slm_received_bits(diag_path) -> list:
    """Read the per-shot SLM ``received_bits`` column from a synced
    ``slm_diag.h5`` (Track E, schema v3).

    Returns a list of '0'/'1' strings, one per ledger row (empty string
    for rows that predate the column / lack the field). Returns ``[]``
    when the file or column is absent, h5py is unavailable, or the read
    fails — this is a best-effort diagnostic, never fatal.
    """
    diag_path = Path(diag_path)
    if not diag_path.is_file():
        return []
    try:
        import h5py  # local import: optional dependency for this branch
    except ImportError:
        logger.info("read_slm_received_bits: h5py not available")
        return []
    try:
        with h5py.File(diag_path, 'r') as f:
            g = f['/diag'] if '/diag' in f else f
            if 'received_bits' not in g:
                return []
            out = []
            for v in g['received_bits'][:]:
                out.append(v.decode() if isinstance(v, bytes) else str(v))
            return out
    except (OSError, KeyError) as ex:
        logger.warning("read_slm_received_bits: %s: %s", diag_path, ex)
        return []


def compare_lab_vs_slm_bitstrings(lab_bits, slm_bits) -> dict:
    """Diff per-shot lab-side vs SLM-side bitstrings (Phase 5.5 Track E).

    Standalone diagnostic helper — NOT part of ``analyze_scan_dir``'s
    flow. Call it directly when debugging why the SLM-side bitstring
    detection (its own gridLocations / threshold pipeline) diverges from
    the lab-side detection.

    Args:
        lab_bits:  iterable of per-shot lab bitstrings. Each element may
                   be a '0'/'1' string, bytes, or an array-like of
                   truthy values (lab logicals).
        slm_bits:  iterable of per-shot SLM ``received_bits`` (same forms;
                   typically from :func:`read_slm_received_bits`).

    Shots are paired positionally up to ``min(len(lab), len(slm))``. A
    shot is *comparable* only when both sides parse to equal-length 0/1
    arrays; mismatched-length or unparseable shots are recorded in
    ``skipped`` (with a reason) and excluded from the distance stats.

    Returns::

        {
          'n_shots':         int,   # pairs attempted
          'n_comparable':    int,   # pairs actually diffed
          'hamming':         [int|None, ...],   # per-shot, None if skipped
          'disagreement':    [[0/1,...]|None],  # per-shot per-site mask
          'per_site_disagree_rate': [float,...] | None,  # over comparable shots
          'total_disagreements':    int,
          'mean_hamming':           float | None,
          'n_sites':                int | None,   # when all comparable shots agree on length
          'skipped':         [{'shot': int, 'reason': str}, ...],
        }
    """
    lab_list = list(lab_bits) if lab_bits is not None else []
    slm_list = list(slm_bits) if slm_bits is not None else []
    n_shots = min(len(lab_list), len(slm_list))

    hamming: list = []
    disagreement: list = []
    skipped: list = []
    comparable_masks: list = []
    lengths: set = set()

    for i in range(n_shots):
        lab_arr = _coerce_bitstring(lab_list[i])
        slm_arr = _coerce_bitstring(slm_list[i])
        if lab_arr is None or slm_arr is None:
            hamming.append(None)
            disagreement.append(None)
            skipped.append({'shot': i, 'reason': 'unparseable_or_empty'})
            continue
        if lab_arr.size != slm_arr.size:
            hamming.append(None)
            disagreement.append(None)
            skipped.append({
                'shot': i,
                'reason': f'length_mismatch(lab={lab_arr.size},'
                          f'slm={slm_arr.size})'})
            continue
        mask = (lab_arr != slm_arr).astype(np.uint8)
        hamming.append(int(mask.sum()))
        disagreement.append(mask.tolist())
        comparable_masks.append(mask)
        lengths.add(int(mask.size))

    n_comparable = len(comparable_masks)
    if n_comparable and len(lengths) == 1:
        stacked = np.vstack(comparable_masks)
        per_site_rate = stacked.mean(axis=0).tolist()
        n_sites = int(lengths.pop())
    else:
        per_site_rate = None
        n_sites = None
    valid_h = [h for h in hamming if h is not None]
    total_disagreements = int(sum(valid_h))
    mean_hamming = (float(np.mean(valid_h)) if valid_h else None)

    return {
        'n_shots': n_shots,
        'n_comparable': n_comparable,
        'hamming': hamming,
        'disagreement': disagreement,
        'per_site_disagree_rate': per_site_rate,
        'total_disagreements': total_disagreements,
        'mean_hamming': mean_hamming,
        'n_sites': n_sites,
        'skipped': skipped,
    }
