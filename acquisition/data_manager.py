"""DataManager: image acquisition, atom detection, histogram lifecycle, and saving.

State model (3 layers):
  LOADED  — from disk at startup, read-only during scan
  LIVE    — accumulates during scan, resets every 2000 shots
  EFFECTIVE — what detect_atom and the dashboard actually use
"""

import os
import time
import logging
import threading
import collections
import warnings
import numpy as np

from yb_analysis.config import (
    UPDATE_GRID_INTERVAL, UPDATE_GRID_BATCH_SIZE,
    UPDATE_THRES_INTERVAL, UPDATE_THRES_BATCH_SIZE,
    UPDATE_LOADING_INTERVAL, UPDATE_HIST_BATCH_SIZE,
    AFFINE_LIVE_INTERVAL, AFFINE_LIVE_BATCH, AFFINE_LIVE_SEARCH_RANGE,
    AFFINE_LIVE_EMA, THRES_LIVE_INTERVAL, THRES_LIVE_WINDOW,
    THRES_LIVE_EMA, THRES_LIVE_MIN_PER_SIDE,
    THRES_FIT_MIN_SEP_ABS, THRES_FIT_MIN_SEP_SIGMA, THRES_FIT_MIN_SEP_FRAC,
    THRES_FIT_MIN_LOADING, THRES_LIVE_VALLEY_MARGIN, THRES_LOADED_MAX_SPREAD,
    THRES_ACCUM_CROSS_RUN_WINDOW_S, THRES_ACCUM_CROSS_RUN_MAX,
    BLANK_FRAME_FLOOR, THRES_FIT_MAX_BLANK_FRAC, THRES_FIT_MAX_SHOTS,
    HIST_DISPLAY_CLIP_PCT,
    IMG2_SHAPE_MODEL_VARIANT,
)
from yb_analysis.detection.detect_atom import detect_atom
from yb_analysis.detection import spot_shape_model as ssm
from yb_analysis.detection.scan_analysis import (
    extract_scan_params, extract_scan_params_h5, extract_scan_name,
    extract_scan_dims, extract_scan_dims_h5, compute_scan_curve,
)
from yb_analysis.detection.locate_atom import locate_atom_update
from yb_analysis.detection.buffers import RingBuffer
from yb_analysis.io.scan_directory import make_scan_dir, make_scan_fname, scan_id_to_stamps
from yb_analysis.io.hdf5_store import create_scan_file, append_block
from yb_analysis.io.mat_reader import load_scan_config

logger = logging.getLogger(__name__)

_cache = {}
_cache_lock = threading.Lock()
# Serialises post-scan affine updates (one global affine file).
_AFFINE_UPDATE_LOCK = threading.Lock()

# Live target-aware survival: how often (shots) to refresh the per-shot diag
# targets from the SLM, and how many consecutive empty pulls before giving up
# (a non-rearrangement run has no SLM diag → stop polling, ~zero cost). Kept
# small so short / single-point (0d) rearrangement scans pull targets early.
DIAG_PULL_INTERVAL = 5
DIAG_PULL_MAX_EMPTY = 4

# Per-shot mean loading rate (img1), held at module scope so it survives
# scan transitions — the DataManager (and its log_buffer) is recreated on
# each new scan, but this deque persists for the camera-connection lifetime.
# Also fed from dummy-mode frames in control_panel so the live trace keeps
# updating while no real scan is running.
_loading_history = collections.deque(maxlen=100)
_loading_history_lock = threading.Lock()


def record_loading(logicals):
    """Append the per-shot mean loading rate to the persistent history."""
    if logicals is None:
        return
    arr = np.asarray(logicals)
    if arr.size == 0:
        return
    with _loading_history_lock:
        _loading_history.append(float(arr.mean()))


def get_loading_history():
    """Snapshot of the persistent loading-rate history as a 1-D array."""
    with _loading_history_lock:
        return np.array(_loading_history, dtype=float) if _loading_history else None


# --- Cross-run threshold accumulation (in-memory, per loading pattern) ---------
# A DataManager (and its frame-0 intensity buffer) is recreated per scan_id, so a
# short run never reaches the 200-shot full Gaussian refit. To let the full fit
# fire over SEVERAL short runs of the SAME pattern, carry recent frame-0 intensity
# vectors at module scope, keyed by pattern name, age-windowed to
# THRES_ACCUM_CROSS_RUN_WINDOW_S. Lost on process restart (acceptable). Each holder:
#   {'entries': deque[(ts, vec)], 'n_since_attempt': int, 'num_sites': int}
_pattern_accum = {}
_pattern_accum_lock = threading.Lock()


def _get_pattern_accum(name, num_sites):
    """Return the cross-run accumulator for ``name`` (reset if its site count
    changed, e.g. a different grid). None name -> None."""
    if not name:
        return None
    with _pattern_accum_lock:
        pa = _pattern_accum.get(name)
        if pa is None or pa['num_sites'] != int(num_sites):
            pa = {'entries': collections.deque(), 'n_since_attempt': 0,
                  'num_sites': int(num_sites)}
            _pattern_accum[name] = pa
        return pa


def _is_blank_intensities(intensities):
    """True if a shot is a blank / dropped frame — every masked-site intensity is
    essentially zero (the camera/ZMQ returned no real frame). Real frames always sit
    on the camera pedestal, far above BLANK_FRAME_FLOOR, so this never rejects a
    legitimately-empty atom shot. Such frames must be kept out of the threshold-fit
    accumulators and the per-site histograms (see BLANK_FRAME_FLOOR)."""
    arr = np.asarray(intensities)
    if arr.size == 0:
        return True
    mx = np.nanmax(arr)
    return not np.isfinite(mx) or float(mx) < BLANK_FRAME_FLOOR


def _prune_pattern_accum(pa, now):
    """Drop entries older than the cross-run window and cap the deque length."""
    if pa is None:
        return
    cutoff = now - THRES_ACCUM_CROSS_RUN_WINDOW_S
    entries = pa['entries']
    while entries and entries[0][0] < cutoff:
        entries.popleft()
    while len(entries) > THRES_ACCUM_CROSS_RUN_MAX:
        entries.popleft()


def get_data_manager(scan_id):
    with _cache_lock:
        if scan_id in _cache:
            return _cache[scan_id]
        # Evict all previous managers — only the active scan needs memory.
        # Each DataManager can hold hundreds of MB of image data. Save any
        # pending data before eviction so late-arriving frames from a
        # previous scan aren't lost.
        for old_id in list(_cache):
            if old_id != scan_id:
                old_dm = _cache[old_id]
                try:
                    old_dm.save_data()
                except Exception as e:
                    logger.warning('Pre-eviction save failed for scan %d: %s',
                                   old_id, e)
                # Phase 2 of the lab-PC migration: schedule the SLM
                # diag/code sync as a background thread. Runs in
                # parallel with the threaded HDF5 final-save above.
                # Independent files (slm_diag.h5 / slm_code.json) so
                # no ordering constraint. Off-by-default if the
                # DataManager opts out.
                try:
                    old_dm._schedule_slm_sync()
                except Exception as e:
                    logger.warning(
                        'SLM sync schedule failed for scan %d: %s',
                        old_id, e)
                # Loading-pattern affine migration: update the global
                # SLM->camera affine from this scan's observed drift
                # (background thread; no-op unless a pattern was declared).
                try:
                    old_dm._schedule_affine_update()
                except Exception as e:
                    logger.warning(
                        'Affine update schedule failed for scan %d: %s',
                        old_id, e)
                logger.debug('Evicting DataManager for scan %d', old_id)
                del _cache[old_id]
        dm = DataManager(scan_id)
        _cache[scan_id] = dm
        return dm


def drop_all():
    with _cache_lock:
        _cache.clear()


def _scalar(val, default=0):
    """Extract Python scalar from MATLAB value (may be (1,1) ndarray)."""
    if isinstance(val, np.ndarray):
        return float(val.flat[0])
    return float(val) if val is not None else default


def _scalar_or_none(val):
    """Like _scalar but returns None (not 0) when the value is absent/empty."""
    try:
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return None
            return float(val.flat[0])
        return float(val)
    except (TypeError, ValueError):
        return None


def _now_iso():
    """Local ISO timestamp (ms precision) for threshold-health stamping."""
    from datetime import datetime
    return datetime.now().isoformat(timespec='milliseconds')


def _mat_str(val):
    """Coerce a MATLAB string field to a Python str. v7.3 HDF5 stores char
    arrays as uint16; scipy gives str/bytes. Returns None for empty/None."""
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode('utf-8', 'replace').strip() or None
    if isinstance(val, str):
        return val.strip() or None
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ('u', 'i'):       # uint16/uint8 char codes
            try:
                return ''.join(chr(int(c)) for c in val.ravel()).strip() or None
            except Exception:
                return None
        if val.dtype.kind in ('S', 'U'):
            return ''.join(val.ravel().astype(str)).strip() or None
        if val.size == 1:
            return _mat_str(val.flat[0])
    return str(val).strip() or None


def _vector(val):
    """Flatten MATLAB vector to 1-D float64 array."""
    return np.asarray(val, dtype=np.float64).ravel()


class DataManager:

    # Phase 2 of the lab-PC migration plan: pull SLM-side per-shot diag
    # rows into <scan_dir>/slm_diag.h5 + code-snapshot manifest into
    # <scan_dir>/slm_code.json when the scan ends. Off-by-default at
    # the class level so tests / mock runs don't hit the network;
    # run_monitor flips it on at startup unless --no-slm.
    sync_after_finish = True

    # Loading-pattern affine migration: after each scan, update the global
    # SLM->camera affine from the observed drift (mirrors the live grid-shift
    # tracker). Off-by-default at the class level for tests/mock runs;
    # run_monitor leaves it on. Only acts when the scan declared a loading
    # pattern (self._pattern_grids is not None).
    affine_autoupdate = True

    def __init__(self, scan_id):
        self.scan_id = scan_id
        # Loading-pattern state (set by _build_pattern_grids; None = legacy
        # day-folder-grid behaviour, unchanged).
        self._pattern_grids = None      # {frame_idx: cropped grid [Y,X]}
        self._pattern_names = {}        # {frame_idx: pattern name}
        self._pattern_knm = {}          # {frame_idx: knm [y,x]} (for the live affine update)
        self._roi = None                # [Xoff, Yoff, W, H] for this scan
        self._affine_grid0 = None       # frame-0 affine-predicted grid (pre-drift)
        # Loading-phase health (Live status strip "phase" tile). _build_pattern_grids
        # records each declared pattern's phase-file status as it contacts the SLM
        # server ('ok'|'missing'|'unreachable'); _compute_pattern_health turns that
        # + the per-pattern expConfig (ByPattern) check into the chip payload.
        self._pattern_phase_status = {}  # pattern name -> 'ok'|'missing'|'unreachable'
        self._pattern_health = None      # get_plot_data payload; None = no loading pattern declared

        # Paths
        date_stamp, time_stamp = scan_id_to_stamps(scan_id)
        self.dname, self.date, self.time = make_scan_dir(date_stamp, time_stamp)
        mat_fname, _, _ = make_scan_fname(date_stamp, time_stamp, self.dname)
        self.fname = os.path.splitext(mat_fname)[0] + '.h5'
        self._day_dir = os.path.dirname(self.dname)  # Data/YYYYMMDD/

        # Load scan config
        self.config = load_scan_config(mat_fname)       # JSON sidecar (pyctrl) or .mat (MATLAB)
        fs = _vector(self.config.get('frameSize', [0, 0]))
        # MATLAB frameSize = [W, H]; after transpose in zmq_client, images are (H, W)
        self.frame_size = (int(fs[1]), int(fs[0]))  # (H, W)
        self.num_images_per_seq = int(_scalar(self.config.get('NumImages', 1)))
        self.is_init = bool(_scalar(self.config.get('isInit', 0)))
        self.is_hc = bool(_scalar(self.config.get('isHC', 0)))
        # Two-array mode (isGrid2=1): image-2 uses its own grid/thresholds
        # loaded from gridLocations_img2.txt / threshold_img2.mat in the day
        # folder. Drift correction is inherited from image-1; image-2
        # thresholds are never refit. See plan: two-array mode.
        self.is_two_array = bool(_scalar(self.config.get('isGrid2', 0)))
        self.grid_locations_img2 = None
        self.num_sites_img2 = 0
        self.loaded_thresholds_img2 = None
        self.loaded_infidelities_img2 = None
        self.loaded_gauss_fits_img2 = None
        self.loaded_hist_data_img2 = None
        self.loading_rates_img2 = None
        self.log_buffer_img2 = None
        # img2 INDEPENDENT threshold refit (loading-pattern affine migration):
        # when img2 is a DISTINCT loading pattern from img1, it gets its OWN
        # live Gaussian refit on the final-frame intensities. (When both frames
        # share the loading pattern, img2 keeps sharing img1's single refit via
        # _effective_thresholds — unchanged.) Same degenerate-fit guard as img1;
        # between full fits img2 HOLDS the last accepted fit (no cheap inter-fit
        # drift tracker — a post-protocol frame has no tracking need and holding
        # is strictly safe). Refined thresholds save back to img2's per-pattern
        # store, so it self-calibrates per pattern just like img1.
        self.live_thresholds_img2 = None
        self.live_infidelities_img2 = None
        self.live_gauss_fits_img2 = None
        self.live_hist_data_img2 = None
        self._intensity_accum_img2 = []
        self._img_cnt_refit_img2 = 0
        self._thr_has_accepted_fit_img2 = False
        self._threshold_health_img2 = {'state': 'init', 'reason': '',
                                       'source': None, 'spread': None,
                                       'updated_iso': None}
        # img2 spot-shape GMM detector: when active, img2 logicals come from the
        # model (not the intensity threshold) and each carries a per-site
        # posterior "% certainty". Resolved in _resolve_img2_model() once the
        # grid is known; None -> img2 falls back to threshold detection.
        self._img2_model = None
        self._img2_logicals_source = None   # provenance tag stored in the HDF5
        self._display_proba2 = None         # last img2 per-site posterior (display)
        self._proba_img2_to_save = []       # per-(is_last)-frame posteriors, for HDF5
        box_size = int(_scalar(self.config.get('boxSize', 11)))
        mask_sigma = float(_scalar(self.config.get('maskSigma', 2.0)))
        self.mask_mat = _gaussian_mask(box_size, mask_sigma)

        if self.is_init or self.is_hc:
            if self.is_hc:
                logger.info('High-coherence mode (isHC=1): no images via ZMQ')
            self._init_empty()
            # For isInit scans, still create the HDF5 so images get saved
            if self.is_init and self.frame_size[0] > 0:
                try:
                    create_scan_file(self.fname, self.config, self.frame_size, 1)
                    self._file_created = True
                except Exception as e:
                    logger.warning('Failed to create HDF5 for init scan: %s', e)
            return

        # --- LOADED state (from disk, read-only during scan) ---
        self.num_sites = 0
        self._load_from_disk()

        # If the scan declared loading pattern(s), replace the day-folder grid
        # with the pattern's simulated positions mapped through the global
        # affine (per-image). Falls back to the day-folder grid on any failure.
        try:
            self._build_pattern_grids()
        except Exception as e:
            logger.warning('pattern-grid build failed (%s); using day grid', e)
            self._pattern_grids = None
        # Loading-phase health chip (status strip): does each declared loading
        # phase actually exist on the SLM server, and does its pattern have an
        # expConfig (ByPattern) entry. Best-effort -> never break DataManager.
        try:
            self._compute_pattern_health()
        except Exception as e:  # noqa: BLE001
            logger.warning('pattern-health compute failed: %s', e)
            self._pattern_health = None

        # --- LIVE state (accumulates, resets at 2000) ---
        self._intensity_accum = []          # list of (num_sites,) arrays
        self.live_hist_data = None           # rebinned every shot after accumulation starts
        self.live_gauss_fits = None          # fitted every 200 shots
        self.live_thresholds = None          # from live_gauss_fits
        self.live_infidelities = None

        # --- Counters ---
        self._img_cnt_grid = 0
        self._img_cnt_refit = 0             # counts toward 200-shot Gaussian refit
        self._img_cnt_loading = 0
        self._img_cnt_affine = 0            # counts toward the live affine (position) update
        self._img_cnt_thres_live = 0        # counts toward the cheap threshold EWMA update
        self._blank_shot_count = 0          # whole-frame ~0 (dropped) shots skipped this scan
        self._seq_total = 0                 # cumulative completed seqs this run (= shot stamp)
        # Frame-drop safety: sequences the camera/compute stall delivered with
        # != pSeq frames. We drop them whole (see store_new_data) so they can't
        # phase-flip the img1/img2 parity demux for the rest of the run.
        self._dropped_seqs = 0              # count of incomplete sequences dropped
        self._dropped_seq_ids = []          # their seq_ids (for diagnostics)
        # ZMQ delivery gap tracking: detects seq_ids that were never received
        # (lost when a grab_imgs() timeout races a large server buffer drain).
        self._seq_ids_max_seen = 0          # highest seq_id received this scan
        self._seq_gap_count = 0             # total shots inferred missing
        self._seq_gap_ids = []              # first <=200 missing seq_ids
        # HDF5 save health surfaced to the dashboard (see get_plot_data). The
        # save runs in a daemon thread; if append_block ultimately fails (e.g. a
        # OneDrive lock that outlasts the retries) we record it here instead of
        # letting the thread die silently, so the Live view turns red.
        self._save_health = {'state': 'ok', 'reason': '', 'lost_seqs': 0,
                             'lost_seq_ids': [], 'last_error': '',
                             'updated_iso': None}
        self._hist_version = 0              # incremented on each Gaussian refit
        self._hist_rep_sites = self._pick_rep_sites()
        # Live self-calibration helpers (loading-pattern scans). Placement ratio
        # r = (thr - mu_empty)/(mu_atom - mu_empty) per site from the last full
        # Gaussian fit; the cheap inter-fit threshold tracker keeps the threshold
        # at this ratio between the (drifting) empty/atom population means.
        self._thr_place_ratio = None        # (num_sites,) or None -> default 0.5
        # Structural guard state. Valley clamp band [lo, hi] per site comes from
        # the last ACCEPTED full fit; the cheap tracker keeps each threshold inside
        # it so a drift can never climb onto/past a peak. None until an accepted fit
        # exists (this session or carried in via the cross-run accumulator).
        self._thr_clamp_lo = None
        self._thr_clamp_hi = None
        self._thr_fit_anchor = None          # last full-fit thresholds: the FIXED
                                             # cut the cheap tracker shifts (no ratchet)
        self._thr_fit_med_below = None       # fit-window medians below/above the cut
        self._thr_fit_med_above = None       # (same estimator the cheap tracker uses;
                                             # cheap shifts the cut by their drift)
        self._thr_has_accepted_fit = False   # gate for the cheap refit (need an anchor)
        # Detection-threshold health surfaced to the dashboard (see get_plot_data).
        self._threshold_health = {'state': 'init', 'reason': '', 'source': None,
                                  'spread': None, 'updated_iso': None}
        # Cross-run accumulation: the frame-0 (loading) pattern name this run
        # contributes its intensities to, plus the active loading defocus (display).
        self._accum_pattern_name = self._pattern_names.get(0)
        self._active_defocus = _scalar_or_none(self.config.get('loadingDefocus'))
        self._affine_update_running = False  # guard: one background affine thread at a time

        # Seed the frame-0 intensity buffer from the SAME pattern's recent shots
        # (other short runs within the cross-run window) so the 200-shot full fit
        # can fire across runs, not just within one. Carry the shots-since-attempt
        # counter so the trigger counts total accumulated shots.
        if self._accum_pattern_name:
            pa = _get_pattern_accum(self._accum_pattern_name, self.num_sites)
            _prune_pattern_accum(pa, time.time())
            with _pattern_accum_lock:
                # Drop blank/dropped frames carried over from a prior run — they
                # would otherwise re-poison the first full fit (a blank-vs-pedestal
                # toggle read as empty-vs-atom).
                self._intensity_accum = [v.copy() for _, v in pa['entries']
                                         if not _is_blank_intensities(v)]
                self._img_cnt_refit = min(int(pa['n_since_attempt']),
                                          UPDATE_THRES_INTERVAL)
            if self._intensity_accum:
                logger.info('Seeded %d cross-run frame-0 shots for pattern %s '
                            '(shots-since-attempt=%d)', len(self._intensity_accum),
                            self._accum_pattern_name, self._img_cnt_refit)

        # Same cross-run seeding for the img2 frame when it is a distinct
        # loading pattern (so its 200-shot full fit can also fire over several
        # short runs). Namespaced key — see _img2_accum_key.
        if self._img2_refit_active():
            akey = self._img2_accum_key()
            pa2 = _get_pattern_accum(akey, self.num_sites_img2)
            _prune_pattern_accum(pa2, time.time())
            with _pattern_accum_lock:
                self._intensity_accum_img2 = [v.copy() for _, v in pa2['entries']
                                              if not _is_blank_intensities(v)]
                self._img_cnt_refit_img2 = min(int(pa2['n_since_attempt']),
                                               UPDATE_THRES_INTERVAL)
            if self._intensity_accum_img2:
                logger.info('Seeded %d cross-run img2 shots for pattern %s '
                            '(shots-since-attempt=%d)',
                            len(self._intensity_accum_img2),
                            self._img2_pattern_name(), self._img_cnt_refit_img2)

        # On-load sanity of the starting thresholds (degraded store -> loud warn +
        # hold; healthy store -> prime the cheap-tracker anchor).
        try:
            self._validate_loaded_thresholds()
        except Exception as e:  # noqa: BLE001
            logger.debug('loaded-threshold validation failed: %s', e)

        # Resolve the img2 spot-shape detector now that the grid is known (so
        # the provenance tag is ready for create_scan_file below).
        self._resolve_img2_model()

        # --- Buffers ---
        buf_size = max(UPDATE_THRES_BATCH_SIZE, UPDATE_GRID_BATCH_SIZE)
        self.img_buffer = RingBuffer(buf_size, self.frame_size, dtype='int16')
        self.log_buffer = RingBuffer(buf_size, (self.num_sites,), dtype='float64')
        if self.is_two_array and self.num_sites_img2 > 0:
            self.log_buffer_img2 = RingBuffer(buf_size, (self.num_sites_img2,),
                                              dtype='float64')
            self.loading_rates_img2 = np.zeros(self.num_sites_img2)

        # --- Scan curve ---
        raw_params = self.config.get('Params')
        self._param_indices = _vector(raw_params).astype(int) if raw_params is not None else None
        # Multi-dim scan support
        self._scan_dims = extract_scan_dims(self.config)
        if self._scan_dims is None:
            h5_dims = extract_scan_dims_h5(mat_fname)
            if h5_dims:
                self._scan_dims = h5_dims
        sp = self._scan_dims[0]['values'] if self._scan_dims else None
        self._scan_params = sp if sp is not None else extract_scan_params_h5(mat_fname)
        self._scan_title = _extract_scan_title(self.config)  # e.g. "FreqPushOut308Scan"
        self._scan_param_path = self._scan_dims[0]['name'] if self._scan_dims else extract_scan_name(self.config)
        self._scan_name = self._scan_title or self._scan_param_path
        self._plot_scale = float(_scalar(self.config.get('PlotScale', 1)))
        self._scan_logicals = []  # list of (seq_id, logic1, logic2_or_None)
        self._last_batch_seq_ids = []  # seq_ids added in the most recent batch

        # Live target-aware survival: per-shot target site sets pulled from the
        # SLM diag during the scan (seq_id -> np.ndarray of lab-site indices).
        # Lets the live scan curve show TP (target survival), matching the
        # Analysis tab. Empty -> per-site survival (unchanged). Populated by a
        # guarded background pull (_pull_live_targets); reset per scan.
        self._seq_targets = {}
        self._diag_since = 0          # last seq_id pulled (incremental)
        self._diag_pull_cnt = 0       # shots since the last pull
        self._diag_pull_running = False
        self._diag_empty_streak = 0   # consecutive empty pulls (back-off)

        # --- Other state ---
        self.loading_rates = np.zeros(self.num_sites)
        self.grid_shift_history = []
        self.grid_shift_heatmap = None

        # --- Display state: image-1, image-2 (final), image-mid of latest seq ---
        # For NumImages == 3 (two-round SLM rearrangement), image-mid holds the
        # middle frame; the dashboard can toggle between displaying the final
        # frame and the middle frame in the second image panel.
        self._display_image = None
        self._display_intensities = None
        self._display_logicals = None
        self._display_image2 = None
        self._display_intensities2 = None
        self._display_logicals2 = None
        self._display_image_mid = None
        self._display_intensities_mid = None
        self._display_logicals_mid = None

        # --- Processing buffers ---
        self._frame_size_fixed = False
        self._imgs_to_process = []
        self._seq_ids_to_process = []
        self._imgs_to_save = []
        self._logicals_to_save = []
        self._intensities_to_save = []
        self._seq_ids_to_save = []

        # --- File I/O ---
        self._save_lock = threading.Lock()
        self._file_created = False
        # Two-array HDF5 layout only when we'll actually capture img2 frames.
        # With isGrid2=1 but NumImages=1, fall back to single-array layout.
        self._save_two_array = (self.is_two_array
                                and self.num_images_per_seq >= 2
                                and self.num_sites_img2 > 0)
        # img2 logicals from the shape model -> persist a per-site certainty
        # dataset + a provenance tag, but only when we're actually saving img2.
        img2_src = self._img2_logicals_source if self._save_two_array else None
        if self.num_sites > 0:
            try:
                create_scan_file(
                    self.fname, self.config, self.frame_size, self.num_sites,
                    two_array=self._save_two_array,
                    num_sites_img2=self.num_sites_img2,
                    img2_logicals_source=img2_src,
                )
                self._file_created = True
            except Exception as e:
                logger.warning('Failed to create HDF5: %s', e)

    def _init_empty(self):
        """Initialize empty state for isInit / isHC scans."""
        self.num_sites = 0
        self.loaded_thresholds = np.array([])
        self.loaded_infidelities = np.array([])
        self.loaded_gauss_fits = None
        self.grid_locations = np.zeros((0, 2))
        # Two-array state stays empty in isInit/isHC paths
        self.is_two_array = False
        self.grid_locations_img2 = None
        self.num_sites_img2 = 0
        self.loaded_thresholds_img2 = None
        self.loaded_infidelities_img2 = None
        self.loaded_gauss_fits_img2 = None
        self.loaded_hist_data_img2 = None
        self.loading_rates_img2 = None
        self.log_buffer_img2 = None
        self.live_thresholds_img2 = None
        self.live_infidelities_img2 = None
        self.live_gauss_fits_img2 = None
        self.live_hist_data_img2 = None
        self._intensity_accum_img2 = []
        self._img_cnt_refit_img2 = 0
        self._thr_has_accepted_fit_img2 = False
        self._threshold_health_img2 = {'state': 'init', 'reason': '',
                                       'source': None, 'spread': None,
                                       'updated_iso': None}
        self._img2_model = None
        self._img2_logicals_source = None
        self._display_proba2 = None
        self._proba_img2_to_save = []
        self._save_two_array = False
        self._display_image = None
        self._display_intensities = None
        self._display_logicals = None
        self._display_image2 = None
        self._display_intensities2 = None
        self._display_logicals2 = None
        self._display_image_mid = None
        self._display_intensities_mid = None
        self._display_logicals_mid = None
        self._intensity_accum = []
        self.live_hist_data = self.live_gauss_fits = None
        self.live_thresholds = self.live_infidelities = None
        self._thr_place_ratio = None
        self._thr_clamp_lo = self._thr_clamp_hi = None
        self._thr_fit_anchor = None
        self._thr_fit_med_below = None
        self._thr_fit_med_above = None
        self._thr_has_accepted_fit = False
        self._threshold_health = {'state': 'init', 'reason': '', 'source': None,
                                  'spread': None, 'updated_iso': None}
        # HDF5 save health (isInit scans still create the file + save images).
        self._save_health = {'state': 'ok', 'reason': '', 'lost_seqs': 0,
                             'lost_seq_ids': [], 'last_error': '',
                             'updated_iso': None}
        self._dropped_seqs = 0
        self._dropped_seq_ids = []
        self._seq_ids_max_seen = 0
        self._seq_gap_count = 0
        self._seq_gap_ids = []
        self._accum_pattern_name = None
        self._active_defocus = None
        self._hist_version = 0
        self._hist_rep_sites = []
        self.loading_rates = np.array([])
        self.grid_shift_history = []
        self.grid_shift_heatmap = None
        self._scan_logicals = []
        self._last_batch_seq_ids = []
        self._seq_targets = {}
        self._diag_since = 0
        self._diag_pull_cnt = 0
        self._diag_empty_streak = 0
        self._param_indices = None
        self._scan_params = None
        self._scan_dims = None
        self._scan_name = None
        self._scan_param_path = None
        self._plot_scale = 1.0
        self._imgs_to_process = []
        self._seq_ids_to_process = []
        self._imgs_to_save = []
        self._logicals_to_save = []
        self._intensities_to_save = []
        self._seq_ids_to_save = []
        self._save_lock = threading.Lock()
        self._file_created = False
        self._frame_size_fixed = False
        # NOTE: don't reset self._day_dir — it's set in __init__ before this call
        self.img_buffer = self.log_buffer = None
        self._img_cnt_grid = self._img_cnt_refit = self._img_cnt_loading = 0
        self._seq_total = 0   # per-run shot stamp (get_plot_data: shots_this_run)

    def _load_from_disk(self):
        """Load LOADED state: try day folder first, then scan config."""
        from yb_analysis.io.preload import (
            _parse_hist_data_struct, _parse_gauss_fits_struct
        )
        from scipy.io import loadmat

        # Grid from config
        grid_x = _vector(self.config.get('initGridLocationsX', []))
        grid_y = _vector(self.config.get('initGridLocationsY', []))
        self.grid_locations = np.column_stack([grid_y, grid_x]) if len(grid_x) > 0 else np.zeros((0, 2))
        self.num_sites = len(grid_x) if len(grid_x) > 0 else 0

        # Defaults from config
        self.loaded_thresholds = _vector(self.config.get('initThresholds', []))
        self.loaded_infidelities = _vector(self.config.get('initInfidelities',
                                           np.full(self.num_sites, np.nan)))
        self.loaded_gauss_fits = None

        # Try day folder grid
        grid_file = os.path.join(self._day_dir, 'gridLocations.txt')
        if os.path.isfile(grid_file):
            try:
                g = np.loadtxt(grid_file, skiprows=1)
                if g.ndim == 1:
                    g = g.reshape(1, -1)
                if g.shape[1] >= 2 and g.shape[0] > 0:
                    self.grid_locations = g[:, :2]   # Y, X — ignore extra columns (e.g. Site_Index)
                    self.num_sites = len(g)
                    logger.info('Loaded grid from day folder (%d sites)', self.num_sites)
            except Exception as e:
                logger.debug('Could not load day-folder grid: %s', e)

        # Try day folder thresholds
        thresh_file = os.path.join(self._day_dir, 'threshold.mat')
        if os.path.isfile(thresh_file):
            try:
                td = loadmat(thresh_file, squeeze_me=True)
                t = np.asarray(td['thresholds'], dtype=np.float64).ravel()
                inf = np.asarray(td['infidelities'], dtype=np.float64).ravel()
                if len(t) == self.num_sites:
                    self.loaded_thresholds = t
                    self.loaded_infidelities = inf
                    logger.info('Loaded thresholds from day folder')
                gf = _parse_gauss_fits_struct(td.get('gaussFitsStruct'))
                if gf and len(gf) == self.num_sites:
                    self.loaded_gauss_fits = gf
                    logger.info('Loaded gauss fits from day folder')
            except Exception as e:
                logger.debug('Could not load day-folder thresholds: %s', e)

        # Fallback gauss fits from config
        if self.loaded_gauss_fits is None:
            raw_gf = self.config.get('gaussFits')
            if raw_gf is not None:
                self.loaded_gauss_fits = _parse_gauss_fits_struct(raw_gf)

        # Two-array mode (isGrid2=1): load image-2 calibration from day folder.
        # Offline analysis (load_data, unpack) handles this layout; see plan.
        if self.is_two_array:
            self._load_img2_calibration()

    def _load_img2_calibration(self):
        """Load image-2 grid/thresholds/histData from the day folder.

        Falls back to single-array mode (self.is_two_array = False) on any
        missing or invalid file, rather than crashing the scan.
        """
        from yb_analysis.io.preload import (
            _parse_hist_data_struct, _parse_gauss_fits_struct
        )
        from scipy.io import loadmat

        grid_path = os.path.join(self._day_dir, 'gridLocations_img2.txt')
        if not os.path.isfile(grid_path):
            logger.warning('isGrid2=1 but %s not found; falling back to '
                           'single-array mode', grid_path)
            self.is_two_array = False
            return

        try:
            g = np.loadtxt(grid_path, skiprows=1)
            if g.ndim == 1:
                g = g.reshape(1, -1)
            if g.shape[1] < 2 or g.shape[0] == 0:
                raise ValueError('grid file has no usable rows')
            self.grid_locations_img2 = g[:, :2]
            self.num_sites_img2 = len(g)
            logger.info('Loaded img2 grid (%d sites)', self.num_sites_img2)
        except Exception as e:
            logger.warning('Failed to load %s (%s); single-array fallback',
                           grid_path, e)
            self.is_two_array = False
            self.grid_locations_img2 = None
            self.num_sites_img2 = 0
            return

        thresh_path = os.path.join(self._day_dir, 'threshold_img2.mat')
        if not os.path.isfile(thresh_path):
            logger.warning('isGrid2=1 but %s not found; single-array fallback',
                           thresh_path)
            self.is_two_array = False
            self.grid_locations_img2 = None
            self.num_sites_img2 = 0
            return
        try:
            td = loadmat(thresh_path, squeeze_me=True)
            t = np.asarray(td['thresholds'], dtype=np.float64).ravel()
            if len(t) != self.num_sites_img2:
                raise ValueError(f'thresholds_img2 length {len(t)} != '
                                 f'num_sites_img2 {self.num_sites_img2}')
            self.loaded_thresholds_img2 = t
            self.loaded_infidelities_img2 = np.asarray(
                td.get('infidelities', np.full(self.num_sites_img2, np.nan)),
                dtype=np.float64).ravel()
            self.loaded_gauss_fits_img2 = _parse_gauss_fits_struct(
                td.get('gaussFitsStruct'))
        except Exception as e:
            logger.warning('Failed to load %s (%s); single-array fallback',
                           thresh_path, e)
            self.is_two_array = False
            self.grid_locations_img2 = None
            self.num_sites_img2 = 0
            self.loaded_thresholds_img2 = None
            self.loaded_infidelities_img2 = None
            self.loaded_gauss_fits_img2 = None
            return

        # histData_img2 is optional — missing/invalid is non-fatal
        hist_path = os.path.join(self._day_dir, 'histData_img2.mat')
        if os.path.isfile(hist_path):
            try:
                hd = loadmat(hist_path, squeeze_me=True)
                self.loaded_hist_data_img2 = _parse_hist_data_struct(
                    hd.get('histData'), self.num_sites_img2)
            except Exception as e:
                logger.debug('Could not load %s: %s', hist_path, e)

        logger.info('Two-array mode active: img1=%d sites, img2=%d sites',
                    self.num_sites, self.num_sites_img2)

    # --- Loading-pattern grids (affine migration) ---

    def _image_pattern_specs(self):
        """Per-image loading-pattern specs, or None for legacy behaviour.

        Priority:
          1. config['imagePatternsJson'] — JSON list of dicts
             {name, base_phase_path, zernike?, order?, legacy_zerniked?,
              baked_zernike?}; entry k -> camera frame k (last entry reused
              if the list is shorter than NumImages).
          2. warmup_kwargs (rearrange scans): initial_phase -> frame 0,
             final_phase -> final frame; extras.*_phase_zernike give the
             baked Zernike to strip for extraction.
        Returns a length-NumImages list (each entry a spec dict or None), or
        None when the scan declares no loading pattern.
        """
        import json as _json
        cfg = self.config
        pSeq = max(1, self.num_images_per_seq)

        def _norm(p):
            s = _mat_str(p)
            return s.replace('\\', '/') if s else None

        def _zern_list(z):
            if z is None:
                return None
            try:
                zl = _vector(z).tolist()
            except Exception:
                return None
            return zl if any(c != 0.0 for c in zl) else None

        def _planes_list(z):
            # Axial layer depths (ANSI 2*rho^2-1 radians) for a 3-D pattern.
            # Unlike a Zernike, z=0 is a meaningful layer depth, so DON'T drop
            # all-zero; only an empty/absent list means "2-D" -> None.
            if z is None:
                return None
            try:
                zl = [float(c) for c in _vector(z).tolist()]
            except Exception:
                return None
            return zl if zl else None

        raw = cfg.get('imagePatternsJson')
        if raw is not None:
            s = _mat_str(raw)
            try:
                items = _json.loads(s) if s else None
            except Exception:
                items = None
            if items:
                specs = [None] * pSeq
                for k in range(pSeq):
                    it = items[k] if k < len(items) else items[-1]
                    if it and it.get('name') and it.get('base_phase_path'):
                        bz = _zern_list(it.get('baked_zernike'))
                        specs[k] = {
                            'name': str(it['name']),
                            'base_phase_path': _norm(it['base_phase_path']),
                            'zernike': it.get('zernike'),
                            'order': it.get('order', 'col'),
                            'legacy_zerniked': bool(it.get('legacy_zerniked',
                                                           bz is not None)),
                            'baked_zernike': bz,
                            # OPTIONAL 3-D layer depths; None -> legacy 2-D.
                            'planes_z_rad': _planes_list(it.get('planes_z_rad')),
                            # OPTIONAL per-spec extraction-threshold override; None
                            # -> fall back to the pattern's registry-record value,
                            # then the 0.30 default (see _build_pattern_grids).
                            'threshold': it.get('threshold'),
                        }
                if any(specs):
                    return specs

        wk = cfg.get('warmup_kwargs')
        if isinstance(wk, dict):
            init_p = _norm(wk.get('initial_phase'))
            final_p = _norm(wk.get('final_phase'))
            extras = wk.get('extras') if isinstance(wk.get('extras'), dict) else {}

            # Optional per-phase (or shared) 3-D layer depths from extras;
            # absent -> 2-D, unchanged. The actual rearrange-protocol 3-D
            # wiring lives elsewhere; here we only thread the depths so the
            # detection grid is derived per-plane when declared.
            shared_planes = extras.get('planes_z_rad')

            def _spec(path, zern, planes):
                bz = _zern_list(zern)
                return {'name': os.path.splitext(os.path.basename(path))[0],
                        'base_phase_path': path, 'zernike': None, 'order': 'col',
                        'legacy_zerniked': bz is not None, 'baked_zernike': bz,
                        'planes_z_rad': _planes_list(
                            planes if planes is not None else shared_planes)}
            if init_p:
                specs = [None] * pSeq
                specs[0] = _spec(init_p, extras.get('initial_phase_zernike'),
                                 extras.get('initial_phase_planes_z_rad'))
                if final_p and pSeq >= 2:
                    specs[pSeq - 1] = _spec(
                        final_p, extras.get('final_phase_zernike'),
                        extras.get('final_phase_planes_z_rad'))
                return specs
        return None

    def _default_thresholds(self, n):
        """Per-site initial thresholds when a pattern has no saved per-pattern
        thresholds yet.

        If the loaded thresholds already have the right site count, KEEP them
        PER-SITE (a same-count warm start) — this is the common case where the
        day-folder / frame-0 thresholds match the pattern, and it must NOT be
        flattened (flattening to the median was the bug that made img1 and img2
        diverge for the same pattern). Only when the site count genuinely
        differs (a different grid) fall back to broadcasting the median. Live
        Gaussian refitting refines per-site within ~200 shots either way."""
        base = np.asarray(self.loaded_thresholds, dtype=np.float64).ravel()
        base = base[np.isfinite(base)]
        if base.size == int(n):
            return base.copy()   # same site count -> keep PER-SITE thresholds
        val = float(np.median(base)) if base.size else 0.0
        return np.full(int(n), val, dtype=np.float64)

    def _build_pattern_grids(self):
        """Replace the day-folder grid with per-image loading-pattern grids:
        each pattern's simulated knm positions mapped through the global
        affine and the per-scan crop ROI. Sets frame-0 -> grid_locations and
        the final frame -> grid_locations_img2 so the EXISTING per-frame
        selection + live drift correction apply unchanged. No-ops (keeps the
        day grid) if no pattern/ROI/affine is available."""
        self._pattern_knm = {}   # (re)populated below; the live affine update reads it
        self._pattern_phase_status = {}  # name -> 'ok'|'missing'|'unreachable' (health chip)
        roi = self.config.get('roi')
        roi = _vector(roi) if roi is not None else None
        if roi is None or roi.size < 4:
            return
        self._roi = [float(v) for v in roi[:4]]

        specs = self._image_pattern_specs()
        if not specs or not any(specs):
            return

        import yb_analysis.analysis.affine_transform as aff
        import yb_analysis.analysis.pattern_registry as reg
        from yb_analysis.slm_sync.client import SlmPhaseNotFound
        self._pattern_names = {k: s['name'] for k, s in enumerate(specs) if s}

        A = aff.load_matrix()
        if A is None:
            logger.warning('loading pattern(s) %s declared but no affine '
                           'committed; using day-folder grid',
                           list(self._pattern_names.values()))
            return

        grids = {}
        for k, s in enumerate(specs):
            if not s:
                continue
            try:
                # Per-pattern extraction threshold precedence: an explicit per-spec
                # override (imagePatternsJson "threshold") wins; else the value
                # stored on the pattern's registry record; else the 0.30 default.
                # This lets a pattern that needs a higher cut to reject spurious
                # low-amplitude edge ghosts (e.g. 33x33_uniform_centered_level ->
                # 0.40, which otherwise leaks 8 phantom spots, 4 off-sensor) carry
                # that in its registry record WITHOUT every scan having to pass it,
                # while a scan can still override per-shot. NOTE: this is the
                # lab-side DETECTION-grid threshold, distinct from the rearrange
                # server-side warmup_kwargs.derive_threshold (the SLM rearrange grid).
                _existing = reg.get_pattern(s['name'])
                _thr = s.get('threshold')
                if _thr is None and _existing is not None:
                    _thr = _existing.get('threshold')
                _thr = 0.30 if _thr is None else float(_thr)
                rec = reg.fetch_or_refresh_pattern(
                    s['name'], base_phase_path=s['base_phase_path'],
                    default_loading_zernike=s.get('zernike'),
                    order=s.get('order', 'col'),
                    threshold=_thr,
                    legacy_zerniked=s.get('legacy_zerniked', False),
                    baked_zernike=s.get('baked_zernike'),
                    planes_z_rad=s.get('planes_z_rad'))
            except SlmPhaseNotFound:
                # The phase file the scan asked for does not exist on the SLM
                # server (almost always a misspelled name). Record it for the
                # "phase" health chip and skip this frame's grid (-> day grid).
                self._pattern_phase_status[s['name']] = 'missing'
                logger.warning('pattern %s: phase file %s NOT FOUND on SLM '
                               'server', s['name'], s['base_phase_path'])
                continue
            except Exception as e:
                self._pattern_phase_status.setdefault(s['name'], 'unreachable')
                logger.warning('pattern %s fetch failed (%s); trying cache',
                               s['name'], e)
                rec = reg.get_pattern(s['name'])
            if not rec or not rec.get('knm'):
                logger.warning('no registry record for pattern %s', s['name'])
                continue
            # A usable record (fresh from the server or a cache hit) means the
            # phase exists -- a transient fetch error that still found a cache
            # is not a "missing phase".
            self._pattern_phase_status[s['name']] = 'ok'
            knm = np.asarray(rec['knm'], dtype=np.float64)
            self._pattern_knm[k] = knm
            grids[k] = aff.apply_affine_cropped(aff._knm_to_xy(knm), A, self._roi)
        if not grids:
            return
        self._pattern_grids = grids

        pSeq = max(1, self.num_images_per_seq)
        if 0 in grids:
            self.grid_locations = np.ascontiguousarray(grids[0])
            self.num_sites = len(grids[0])
            self._affine_grid0 = grids[0].copy()
            (self.loaded_thresholds, self.loaded_infidelities,
             self.loaded_gauss_fits) = self._pattern_thresholds(
                self._pattern_names[0], self.num_sites)
        last = pSeq - 1
        if last >= 1 and last in grids:
            self.is_two_array = True
            self.grid_locations_img2 = np.ascontiguousarray(grids[last])
            self.num_sites_img2 = len(grids[last])
            (self.loaded_thresholds_img2, self.loaded_infidelities_img2,
             self.loaded_gauss_fits_img2) = self._pattern_thresholds(
                self._pattern_names[last], self.num_sites_img2)
        logger.info('Loading-pattern grids active: %s (sites %s); affine '
                    'mapped + ROI-cropped', self._pattern_names,
                    {k: len(v) for k, v in grids.items()})

    # --- Loading-phase health chip ("phase" status tile) ---

    def _active_by_pattern(self):
        """The expConfig ``ByPattern`` table from the scan's embedded config
        snapshot ({} if absent / not a dict). This is the consts that actually
        ran (scan_summary embeds ``SeqConfig.consts`` as ``config['expConfig']``),
        so it is authoritative for "does this pattern have a per-array overlay"."""
        ec = self.config.get('expConfig') if isinstance(self.config, dict) else None
        if not isinstance(ec, dict):
            return {}
        bp = ec.get('ByPattern')
        return bp if isinstance(bp, dict) else {}

    def _probe_phase_status(self, spec):
        """Existence check for one loading phase when _build_pattern_grids did
        not get to it (e.g. no affine committed). Cache-first, so it adds no SLM
        round-trip in the normal path. Returns 'ok'|'missing'|'unreachable'."""
        import yb_analysis.analysis.pattern_registry as reg
        from yb_analysis.slm_sync.client import SlmPhaseNotFound
        try:
            rec = reg.fetch_or_refresh_pattern(
                spec['name'], base_phase_path=spec['base_phase_path'],
                order=spec.get('order', 'col'),
                legacy_zerniked=spec.get('legacy_zerniked', False),
                baked_zernike=spec.get('baked_zernike'),
                planes_z_rad=spec.get('planes_z_rad'))
            return 'ok' if rec else 'unreachable'
        except SlmPhaseNotFound:
            return 'missing'
        except Exception:  # noqa: BLE001 — network/HTTP -> treat as unverified
            return 'unreachable'

    def _compute_pattern_health(self):
        """Build the Live "phase" status-chip payload: for every loading phase
        the scan declares, whether the phase file EXISTS on the SLM server and
        whether its pattern has an expConfig (``ByPattern``) entry.

          red  'fail' -> a declared phase file is MISSING on the SLM server
                         (HTTP 404). The "misspelled phase name" case this chip
                         exists to catch.
          yellow 'warn'-> a pattern has no ByPattern entry while the table is in
                         use (so per-array config is configured and this one was
                         likely mistyped), or the SLM couldn't be reached to verify.
          green 'ok'  -> every declared phase exists (+ has a ByPattern entry
                         when the table is populated).
          grey 'none' -> the scan declares no loading pattern.
        """
        specs = [s for s in (self._image_pattern_specs() or []) if s]
        if not specs:
            self._pattern_health = None
            return

        by_pattern = self._active_by_pattern()
        bp_populated = bool(by_pattern)

        patterns, missing, unreachable, no_cfg, seen = {}, [], [], [], set()
        for s in specs:
            name = s['name']
            if name in seen:
                continue
            seen.add(name)
            ph = self._pattern_phase_status.get(name) or self._probe_phase_status(s)
            self._pattern_phase_status[name] = ph
            has_cfg = name in by_pattern
            patterns[name] = {'phase_path': s.get('base_phase_path'),
                              'phase': ph, 'expconfig': bool(has_cfg)}
            if ph == 'missing':
                missing.append(name)
            elif ph == 'unreachable':
                unreachable.append(name)
            # expConfig check fires only when SOME pattern carries a ByPattern
            # overlay (an empty table = the per-array overlay system is unused,
            # so no pattern "should" have one and a warning would be pure noise).
            if bp_populated and not has_cfg:
                no_cfg.append(name)

        if missing:
            state, reason = 'fail', 'phase file not on SLM server: ' + ', '.join(missing)
        elif no_cfg or unreachable:
            bits = []
            if no_cfg:
                bits.append('no expConfig (ByPattern) entry: ' + ', '.join(no_cfg))
            if unreachable:
                bits.append('could not reach SLM to verify: ' + ', '.join(unreachable))
            state, reason = 'warn', ' · '.join(bits)
        else:
            state, reason = 'ok', ''

        from datetime import datetime as _dt
        self._pattern_health = {
            'state': state,
            'reason': reason,
            'phase_missing': missing,
            'unreachable': unreachable,
            'no_expconfig': no_cfg,
            'bypattern_populated': bp_populated,
            'patterns': patterns,
            'updated_iso': _dt.now().isoformat(timespec='seconds'),
        }

    def _pattern_thresholds(self, name, n):
        """Per-pattern detection thresholds for a grid of n sites: from the
        pattern's threshold store when it matches, else placeholders (the
        live Gaussian refit replaces these within ~200 shots and saves back
        per-pattern via _save_threshold). Returns (thresholds, infidelities,
        gauss_fits)."""
        try:
            import yb_analysis.analysis.pattern_registry as reg
            td = reg.load_pattern_thresholds(name)
        except Exception:
            td = None
        if td and len(np.ravel(td.get('thresholds', []))) == n:
            return (np.asarray(td['thresholds'], dtype=np.float64).ravel(),
                    np.asarray(td['infidelities'], dtype=np.float64).ravel(),
                    td.get('gauss_fits'))
        return self._default_thresholds(n), np.full(n, np.nan), None

    def _schedule_affine_update(self):
        """After a scan with a declared loading pattern, update the global
        affine from the observed drift (background thread). Mirrors the live
        grid-shift tracker. Off if affine_autoupdate is False or no pattern."""
        if not getattr(type(self), 'affine_autoupdate', True):
            return
        if self._pattern_grids is None or self._roi is None:
            return
        name = self._pattern_names.get(0)
        if not name or self.img_buffer is None or self.img_buffer.size() < 10:
            return
        try:
            n = min(self.img_buffer.size(), UPDATE_GRID_BATCH_SIZE)
            imgs = self.img_buffer.get_last_n(n).astype(np.float64)
        except Exception:
            return
        threading.Thread(
            target=self._affine_update_worker,
            args=(name, np.asarray(self._roi, dtype=np.float64), imgs),
            name='affine-update-%s' % self.scan_id, daemon=True).start()

    def _affine_update_worker(self, name, roi, imgs):
        try:
            import yb_analysis.analysis.pattern_registry as reg
            import yb_analysis.analysis.affine_transform as aff
            with _AFFINE_UPDATE_LOCK:
                rec = reg.get_pattern(name)
                if not rec or not rec.get('knm'):
                    return
                cand = aff.propose_scan_update(
                    imgs, np.asarray(rec['knm'], dtype=np.float64), roi,
                    self.mask_mat, str(self.scan_id))
                if cand.get('accept'):
                    aff.commit_update(cand, ema_weight=aff.SHIFT_EMA_WEIGHT)
                    logger.info('Affine drift-updated from scan %s: shift '
                                '(dy=%s, dx=%s) prominence %.1f', self.scan_id,
                                cand.get('shift_dy'), cand.get('shift_dx'),
                                cand.get('prominence'))
                else:
                    logger.info('Affine update skipped for scan %s (%s)',
                                self.scan_id, cand.get('reason'))
        except Exception as e:
            logger.warning('Affine update failed for scan %s: %s',
                           self.scan_id, e)

    # --- Live target-aware survival: pull per-shot diag target sets ---

    def _per_shot_survival_series(self, max_shots=400):
        """Per-shot SURVIVAL in time order, for the 0d Scan-Curve timeseries
        (which otherwise shows loading). Target-aware (TP at this run's diag
        targets) when targets are known for the shot, else per-site survival
        (matched-index logic1&logic2 / loaded). None for 1-image scans (no
        img2). Bounded to the last ``max_shots`` so the snapshot stays small."""
        # Survival only when there's a 2nd image (NumImages >= 2 AND logic2
        # present); 1-image scans keep the loading timeseries.
        if self.num_images_per_seq < 2:
            return None
        sl = self._scan_logicals
        if not sl or sl[0][2] is None:
            return None
        st = self._seq_targets
        recent = sl[-max_shots:]
        vals = []
        target_aware = False
        for seq_id, l1, l2 in recent:
            if l2 is None:
                vals.append(None)
                continue
            a1 = np.asarray(l1, dtype=bool)
            a2 = np.asarray(l2, dtype=bool)
            tgt = st.get(int(seq_id))
            if tgt is not None and len(tgt):
                t = np.asarray(tgt, dtype=int)
                t = t[(t >= 0) & (t < a2.shape[0])]
                vals.append(float(a2[t].sum()) / t.size if t.size else None)
                if t.size:
                    target_aware = True
            elif a1.shape[0] != a2.shape[0]:
                # Cross-grid run (init pattern != target pattern): img1 and img2
                # are detected on DIFFERENT grids, so matched-index per-site
                # survival (a1 & a2) is undefined and would raise. Leave None
                # until this shot's diag targets arrive (the target-aware branch
                # above then fills it with per-shot TP).
                vals.append(None)
            else:
                loaded = int(a1.sum())
                vals.append(float((a1 & a2).sum()) / loaded if loaded > 0 else None)
        if not any(v is not None for v in vals):
            return None
        return {'values': vals, 'target_aware': target_aware}

    def _pull_live_targets(self):
        """Background: refresh ``self._seq_targets`` (seq_id → target lab-site
        indices) from the SLM diag for this scan, incrementally. Fully guarded:
        any failure / SLM-offline is a no-op (the live curve falls back to
        per-site survival). Never blocks acquisition (own daemon thread)."""
        try:
            from yb_analysis.slm_sync.client import SlmSyncClient
            cli = SlmSyncClient()
            diag = cli.get_diag(str(self.scan_id),
                                since_seq_id=(self._diag_since or None))
            # None = SLM unreachable / busy (gate 503) / timeout — TRANSIENT
            # during an active rearrangement scan. Do NOT count it toward the
            # give-up streak; just retry next interval.
            if diag is None:
                return
            entries = diag.get('entries') or []
            if not entries:
                # Only give up when the SLM AFFIRMATIVELY has no diag for this
                # scan_id (count==0) and we've never seen any — i.e. a genuine
                # non-rearrangement run. An empty incremental page when we
                # already have targets is normal (no new shots yet).
                if int(diag.get('count', 0)) == 0 and not self._seq_targets:
                    self._diag_empty_streak += 1
                return
            new_targets = dict(self._seq_targets)
            last = self._diag_since or 0
            added = 0
            for r in entries:
                sid = r.get('seq_id')
                if sid is None:
                    continue
                d = r.get('diag') or {}
                idx = d.get('target_site_indices')
                if not idx:
                    idx = d.get('target_paired')
                if idx:
                    arr = np.asarray(idx, dtype=np.int64).ravel()
                    new_targets[int(sid)] = np.unique(arr[arr >= 0])
                    added += 1
                last = max(last, int(sid))
            # Atomic ref swap (readers use .get(), safe under the GIL).
            self._seq_targets = new_targets
            self._diag_since = last
            self._diag_empty_streak = 0
            if added:
                logger.info('live target-aware survival: %d/%d shots have diag '
                            'targets (scan %s)', len(new_targets),
                            len(self._scan_logicals), self.scan_id)
        except Exception as e:
            logger.debug('live target pull failed (scan %s): %s', self.scan_id, e)
        finally:
            self._diag_pull_running = False

    # --- Live per-N-shots self-calibration (loading-pattern scans) ---

    def _schedule_affine_live(self):
        """Every N loading shots, kick a background thread that nudges the GLOBAL
        affine TRANSLATION toward the recent atoms (EWMA) and re-derives the live
        detection grid from it. Background so the ~0.5-1 s cross-correlation never
        stalls the acquisition loop; a guard prevents overlapping runs."""
        if not getattr(type(self), 'affine_autoupdate', True):
            return
        if self._affine_update_running:
            return
        knm = self._pattern_knm.get(0)
        name = self._pattern_names.get(0)
        if name is None or knm is None or self.img_buffer is None:
            return
        if self.img_buffer.size() < min(AFFINE_LIVE_BATCH, 5):
            return
        try:
            n = min(self.img_buffer.size(), AFFINE_LIVE_BATCH)
            imgs = self.img_buffer.get_last_n(n).astype(np.float64)
        except Exception:
            return
        self._affine_update_running = True
        threading.Thread(
            target=self._affine_live_worker,
            args=(name, np.asarray(knm, dtype=np.float64), imgs, int(self._seq_total)),
            name='affine-live-%s' % self.scan_id, daemon=True).start()

    def _affine_live_worker(self, name, knm, imgs, seq_no):
        try:
            import yb_analysis.analysis.affine_transform as aff
            from yb_analysis.analysis import update_log
            with _AFFINE_UPDATE_LOCK:
                roi = np.asarray(self._roi, dtype=np.float64)
                cand = aff.propose_scan_update(
                    imgs, knm, roi, self.mask_mat, str(self.scan_id),
                    search_range=AFFINE_LIVE_SEARCH_RANGE)
                rec = {'scan_id': str(self.scan_id), 'seq_no': int(seq_no),
                       'shift_dy': cand.get('shift_dy'), 'shift_dx': cand.get('shift_dx'),
                       'snr': cand.get('snr')}
                # Safety (railed shift / low SNR / no affine): log + skip, don't move.
                if not cand.get('accept'):
                    rec.update(accepted=False, reason=cand.get('reason'))
                    update_log.append('affine.jsonl', rec)
                    logger.info('Affine live update skipped @shot %d (%s)',
                                seq_no, cand.get('reason'))
                    return
                entry = aff.commit_update(cand, ema_weight=AFFINE_LIVE_EMA)
                A = np.asarray(entry['A'], dtype=np.float64).reshape(2, 3)
                self._refresh_grids_from_affine(A)
                self.grid_shift_history.append(
                    (cand.get('shift_dy', 0), cand.get('shift_dx', 0)))
                rec.update(accepted=True, ema=AFFINE_LIVE_EMA,
                           ty=float(A[0, 2]), tx=float(A[1, 2]),
                           rotation_deg=entry.get('rotation_deg'),
                           scale_x=entry.get('scale_x'), scale_y=entry.get('scale_y'))
                update_log.append('affine.jsonl', rec)
                logger.info('Affine live-updated @shot %d: shift (dy=%s, dx=%s) '
                            'snr=%.1f -> t=[%.2f, %.2f]', seq_no,
                            cand.get('shift_dy'), cand.get('shift_dx'),
                            cand.get('snr') or 0.0, A[0, 2], A[1, 2])
        except Exception as e:
            logger.warning('Affine live update failed @shot %d: %s', seq_no, e)
        finally:
            self._affine_update_running = False

    def _refresh_grids_from_affine(self, A):
        """Re-derive the live detection grid(s) from affine ``A`` (2x3) + ROI.
        Whole-array reference swaps (atomic under the GIL) so a concurrent
        detect_atom read sees either the old or new grid, never a torn one."""
        import yb_analysis.analysis.affine_transform as aff
        roi = self._roi
        if 0 in self._pattern_knm:
            g0 = aff.apply_affine_cropped(
                aff._knm_to_xy(self._pattern_knm[0]), A, roi)
            self.grid_locations = np.ascontiguousarray(g0)
            self._affine_grid0 = g0.copy()
        last = max(1, self.num_images_per_seq) - 1
        if self.is_two_array and last in self._pattern_knm:
            gl = aff.apply_affine_cropped(
                aff._knm_to_xy(self._pattern_knm[last]), A, roi)
            self.grid_locations_img2 = np.ascontiguousarray(gl)

    def _placement_ratio(self, fits, thres):
        """Per-site r = (thr - mu_empty)/(mu_atom - mu_empty) from a full Gaussian
        fit — where the principled threshold sits between the two peaks. The cheap
        inter-fit tracker holds this ratio as the populations drift. NaN/degenerate
        sites -> 0.5 (midpoint)."""
        n = int(self.num_sites)
        r = np.full(n, 0.5, dtype=np.float64)
        for s in range(min(n, len(fits))):
            p = fits[s].get('params') if isinstance(fits[s], dict) else None
            if p is None:
                continue
            mu_e, mu_a = float(p[0]), float(p[3])
            if mu_a - mu_e > 1e-9:
                r[s] = float(np.clip((thres[s] - mu_e) / (mu_a - mu_e), 0.0, 1.0))
        return r

    def _valley_clamp_band(self, fits):
        """Per-site [lo, hi] the cheap tracker must keep each threshold within,
        from an ACCEPTED full fit: [mu_e + m*sep, mu_a - m*sep] with
        m = THRES_LIVE_VALLEY_MARGIN. A threshold can then never drift onto or
        past either peak (the spread-explosion mechanism). Degenerate/failed
        sites -> (nan, nan): the cheap tracker skips them (no anchor)."""
        n = int(self.num_sites)
        lo = np.full(n, np.nan, dtype=np.float64)
        hi = np.full(n, np.nan, dtype=np.float64)
        m = THRES_LIVE_VALLEY_MARGIN
        for s in range(min(n, len(fits))):
            p = fits[s].get('params') if isinstance(fits[s], dict) else None
            if p is None:
                continue
            mu_e, mu_a = float(p[0]), float(p[3])
            sep = mu_a - mu_e
            if sep > 1e-9:
                lo[s] = mu_e + m * sep
                hi[s] = mu_a - m * sep
        return lo, hi

    def _validate_full_fit(self, fits, thres, all_i, num_sites=None):
        """Structural guard: decide whether a full Gaussian refit is trustworthy.
        Rejects two failure modes: (1) low-loading / near-unimodal data that
        collapses both Gaussians onto one peak (peaks not credibly separated) and
        (2) blank/dropped frames (whole-frame ~0) polluting the buffer, which the
        fit reads as a perfectly-separated empty(~0)-vs-atom(pedestal) pair — the
        2026-06-19 corruption that passed the separation/loading checks. Returns
        (ok: bool, reason: str, stats: dict)."""
        seps, sig_es = [], []
        for f in fits:
            p = f.get('params') if isinstance(f, dict) else None
            if p is None:
                continue
            seps.append(float(p[3]) - float(p[0]))
            sig_es.append(abs(float(p[1])))
        n_conv = len(seps)
        ns = int(self.num_sites if num_sites is None else num_sites)
        stats = {'n_converged': int(n_conv), 'n_sites': ns}
        if n_conv == 0:
            return False, 'no per-site fit converged', stats
        # Blank-frame backstop: even though the ingestion filter keeps blanks out of
        # the accumulators, reject defensively if a meaningful fraction of the fit
        # buffer is whole-frame ~0 (e.g. a buffer carried in before the filter
        # existed). A blank-vs-pedestal split looks like a flawless discriminator,
        # so it would otherwise sail through the separation/loading checks below.
        blank_frac = (float(np.mean(np.nanmax(all_i, axis=1) < BLANK_FRAME_FLOOR))
                      if all_i.size else 0.0)
        stats['blank_frac'] = round(blank_frac, 4)
        if blank_frac > THRES_FIT_MAX_BLANK_FRAC:
            return False, ('%.0f%% of accumulated shots are blank/dropped frames '
                           '(whole-frame ~0)' % (100 * blank_frac)), stats
        seps = np.asarray(seps, dtype=np.float64)
        sig_e = float(np.median(sig_es)) if sig_es else 0.0
        min_sep = max(THRES_FIT_MIN_SEP_ABS, THRES_FIT_MIN_SEP_SIGMA * sig_e)
        frac_sep = float(np.mean(seps >= min_sep))
        loaded_frac = float(np.mean(all_i > thres[None, :]))
        stats.update({'min_sep': round(min_sep, 3),
                      'frac_separated': round(frac_sep, 3),
                      'loaded_frac': round(loaded_frac, 4),
                      'median_sep': round(float(np.median(seps)), 3),
                      'sigma_empty': round(sig_e, 3)})
        if frac_sep < THRES_FIT_MIN_SEP_FRAC:
            return False, ('only %.0f%% of sites have separated peaks '
                           '(need >=%.0f%%, min sep %.2f ADU)'
                           % (100 * frac_sep, 100 * THRES_FIT_MIN_SEP_FRAC,
                              min_sep)), stats
        if loaded_frac < THRES_FIT_MIN_LOADING:
            return False, ('pooled loaded fraction %.1f%% too low for a bimodal '
                           'fit' % (100 * loaded_frac)), stats
        return True, 'ok', stats

    def _validate_loaded_thresholds(self):
        """On-load sanity of the per-site thresholds we start with (from the
        per-pattern store, or day folder for a scan that declares no pattern).
        Sets self._threshold_health and, when the loaded store is HEALTHY with
        matching Gaussian fits, primes the cheap-tracker anchor so it can track
        from the start. A DEGRADED per-pattern store is NOT swapped for day-folder
        thresholds (a different pattern's thresholds are useless) — the cheap
        tracker holds and the first ACCEPTED full fit re-anchors. Warns loudly."""
        n = int(self.num_sites)
        if n <= 0:
            return
        name = self._pattern_names.get(0)
        src = ('pattern:%s' % name) if name else (
            _mat_str(self.config.get('calibrationSource')) or 'day-folder')
        thr = np.asarray(self.loaded_thresholds, dtype=np.float64).ravel()
        iso = _now_iso()
        if thr.size != n or not np.isfinite(thr).any():
            logger.warning('No usable loaded thresholds for %s (n=%d) — will '
                           'calibrate from live data', src, n)
            self._threshold_health = {
                'state': 'unknown', 'reason': 'no usable loaded thresholds',
                'source': src, 'spread': None, 'updated_iso': iso}
            return
        spread = float(np.nanstd(thr))
        # fit-aware: fraction of sites whose loaded threshold sits OUTSIDE its own
        # [mu_empty, mu_atom] — a direct degeneracy signal.
        frac_outside = None
        gf = self.loaded_gauss_fits
        if gf and len(gf) == n:
            out, cnt = 0, 0
            for s in range(n):
                p = gf[s].get('params') if isinstance(gf[s], dict) else None
                pr = np.ravel(p) if p is not None else None
                if pr is None or pr.size < 4:
                    continue
                cnt += 1
                mu_e, mu_a = float(pr[0]), float(pr[3])
                if mu_a - mu_e <= 1e-9 or not (mu_e <= thr[s] <= mu_a):
                    out += 1
            if cnt:
                frac_outside = out / cnt
        outside_bad = (frac_outside is not None
                       and frac_outside > (1.0 - THRES_FIT_MIN_SEP_FRAC))
        if spread > THRES_LOADED_MAX_SPREAD or outside_bad:
            reasons = []
            if spread > THRES_LOADED_MAX_SPREAD:
                reasons.append('threshold spread %.2f ADU (> %.2f)'
                               % (spread, THRES_LOADED_MAX_SPREAD))
            if outside_bad:
                reasons.append('%.0f%% of cuts outside their own peaks'
                               % (100 * frac_outside))
            msg = '; '.join(reasons)
            logger.warning('Loaded thresholds for %s look DEGRADED (%s) — holding '
                           'and re-anchoring from live data; NOT using day folder',
                           src, msg)
            self._threshold_health = {
                'state': 'degraded', 'reason': msg, 'source': src,
                'spread': spread, 'frac_outside': frac_outside,
                'updated_iso': iso}
            return
        # Healthy loaded store: prime the cheap-tracker anchor from its fits so it
        # can track immediately (re-anchored by the first accepted full fit).
        anchor_primed = False
        if gf and len(gf) == n:
            self._thr_place_ratio = self._placement_ratio(gf, thr)
            self._thr_clamp_lo, self._thr_clamp_hi = self._valley_clamp_band(gf)
            if float(np.isfinite(self._thr_clamp_lo).mean()) >= THRES_FIT_MIN_SEP_FRAC:
                self._thr_has_accepted_fit = True
                self._thr_fit_anchor = thr.copy()   # fixed cut for the cheap tracker
                # NOTE: the cheap tracker's median references come only from a live
                # full fit (raw intensities aren't available on load), so it HOLDS
                # the loaded thresholds until the first accepted full fit sets them.
                anchor_primed = True
        if not name:
            logger.warning('Scan declares no loading pattern — using %s thresholds '
                           '(NOT per-pattern); spread %.2f ADU', src, spread)
        self._threshold_health = {
            'state': 'ok' if name else 'unknown_pattern',
            'reason': '' if name else 'no loading pattern declared; using day folder',
            'source': src, 'spread': spread, 'frac_outside': frac_outside,
            'anchor_primed': anchor_primed, 'updated_iso': iso}

    def _update_thresholds_live_cheap(self):
        """Cheap (<50 ms) per-site threshold tracker between full fits. Each cut is
        held at the last ACCEPTED full-fit value shifted by the measured drift of
        BOTH peaks since that fit, so it tracks TOWARD the next full fit (the goal)
        rather than drifting away:
            new = fit_cut + (1-r)*Δμ_empty + r*Δμ_atom
        where r is the fit's placement ratio (the cut sits at fraction r between the
        peaks) and Δμ_{empty,atom} are the drifts of each peak's robust centre. The
        centres are MEDIANS of the recent window split by the FIXED fit cut (below =
        empty, above = atom), and the fit-time references (``_thr_fit_med_below/
        above``) are computed with the SAME median estimator on the fit window — so
        the truncation bias of a one-sided median cancels in the difference and only
        the true drift remains. Splitting by the fixed cut (not the drifting one)
        also kills the positive-feedback ratchet. Net: unbiased + no overshoot +
        tracks both peaks. GUARDS: holds until the first accepted full fit sets the
        references; needs enough samples on BOTH sides; clamps inside the valley
        band so a cut can never sit on/past a peak."""
        name = self._pattern_names.get(0)
        if not name:
            return
        if not self._thr_has_accepted_fit:
            return
        anchor = self._thr_fit_anchor       # last accepted full-fit cut (fixed)
        r = self._thr_place_ratio           # cut's fractional position between peaks
        mb0 = self._thr_fit_med_below       # fit-window median below the cut (empty ref)
        ma0 = self._thr_fit_med_above       # fit-window median above the cut (atom ref)
        lo = self._thr_clamp_lo
        hi = self._thr_clamp_hi
        n = int(self.num_sites)
        if any(x is None or np.asarray(x).size != n
               for x in (anchor, r, mb0, ma0, lo, hi)):
            return  # references not set yet (no accepted full fit) -> hold
        accum = self._intensity_accum
        if len(accum) < max(2 * THRES_LIVE_MIN_PER_SIDE, 10):
            return
        thr = np.asarray(self.thresholds, dtype=np.float64).ravel()
        if thr.size != n:
            return
        W = np.asarray(accum[-THRES_LIVE_WINDOW:], dtype=np.float64)
        if W.ndim != 2 or W.shape[1] != n:
            return
        anchor = np.asarray(anchor, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        mb0 = np.asarray(mb0, dtype=np.float64)
        ma0 = np.asarray(ma0, dtype=np.float64)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        below = W < anchor[None, :]
        above = W > anchor[None, :]
        cb = below.sum(0)
        ca = above.sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # empty cols -> nan (gated out)
            mbn = np.nanmedian(np.where(below, W, np.nan), axis=0)
            man = np.nanmedian(np.where(above, W, np.nan), axis=0)
        # Drift of each peak since the fit (same estimator -> bias cancels); the cut
        # moves with the weighted drift of the two peaks it sits between.
        de = mbn - mb0
        da = man - ma0
        cand = anchor + (1.0 - r) * de + r * da
        valid = ((cb >= THRES_LIVE_MIN_PER_SIDE) & (ca >= THRES_LIVE_MIN_PER_SIDE)
                 & np.isfinite(cand) & np.isfinite(mb0) & np.isfinite(ma0)
                 & np.isfinite(lo) & np.isfinite(hi))
        new = thr.copy()
        a = THRES_LIVE_EMA
        new[valid] = (1.0 - a) * thr[valid] + a * cand[valid]
        # Structural valley clamp: never on/past a peak.
        new[valid] = np.clip(new[valid], lo[valid], hi[valid])
        self.live_thresholds = new
        self._save_threshold()
        self._log_threshold_update(name, 'cheap', int(valid.sum()))

    def _fit_median_refs(self, all_i, thres):
        """Per-site fit-window medians of the values below / above the fit cut — the
        SAME estimator the cheap tracker uses live, so its one-sided truncation bias
        cancels when it differences (now - fit). Empty cols -> NaN (cheap skips)."""
        thr = np.asarray(thres, dtype=np.float64).ravel()
        below = all_i < thr[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mb = np.nanmedian(np.where(below, all_i, np.nan), axis=0)
            ma = np.nanmedian(np.where(~below, all_i, np.nan), axis=0)
        return mb, ma

    def _maybe_refit_img2(self):
        """Independent full Gaussian refit for the img2 (final) frame when it is
        a DISTINCT loading pattern from img1. Mirrors the img1 200-shot refit —
        SAME degenerate-fit guard — but on the img2 accumulator, writing the
        result to img2's per-pattern store. Fires every UPDATE_THRES_INTERVAL
        img2 shots once enough have accumulated (across short runs via the
        namespaced cross-run accumulator). Between fits img2 HOLDS the last
        accepted fit (no cheap inter-fit drift tracker)."""
        name2 = self._img2_pattern_name()
        n2 = len(self._intensity_accum_img2)
        if (self._img_cnt_refit_img2 < UPDATE_THRES_INTERVAL
                or n2 < UPDATE_THRES_INTERVAL):
            return
        self._img_cnt_refit_img2 = 0
        # Reset the cross-run shots-since-attempt for img2's (namespaced) pattern.
        akey = self._img2_accum_key()
        if akey:
            pa = _get_pattern_accum(akey, self.num_sites_img2)
            with _pattern_accum_lock:
                pa['n_since_attempt'] = 0
        all_i = np.array(self._intensity_accum_img2)
        hist = self._compute_hist_data(all_i, self.num_sites_img2)
        self.live_hist_data_img2 = hist
        fits, thres, inf = self._fit_gaussians(all_i, hist_data=hist)
        ok, reason, stats = self._validate_full_fit(
            fits, thres, all_i, num_sites=self.num_sites_img2)
        iso = _now_iso()
        if ok:
            logger.info('Accepted img2 Gaussian refit for %s (%d shots): %s',
                        name2, n2, stats)
            self.live_gauss_fits_img2 = fits
            self.live_thresholds_img2 = thres
            self.live_infidelities_img2 = inf
            self._thr_has_accepted_fit_img2 = True
            self._threshold_health_img2 = {
                'state': 'ok', 'reason': '', 'source': 'fit',
                'spread': float(np.nanstd(thres)),
                'mean_infidelity': (float(np.nanmean(inf)) if inf.size else None),
                'updated_iso': iso, 'stats': stats}
            self._save_threshold_img2()
            self._log_threshold_update(
                name2, 'fit', int(self.num_sites_img2), log_infidelities=True,
                thresholds=thres, infidelities=inf, seq_no=self._seq_total)
        else:
            logger.warning('REJECTED degenerate img2 threshold refit for %s '
                           '(%d shots): %s %s — img2 thresholds unchanged, not '
                           'saved', name2, n2, reason, stats)
            held = (self.live_thresholds_img2
                    if self.live_thresholds_img2 is not None
                    else self.loaded_thresholds_img2)
            held = (np.asarray(held, dtype=np.float64)
                    if held is not None else np.array([]))
            self._threshold_health_img2 = {
                'state': 'degraded', 'reason': reason, 'source': 'fit_rejected',
                'spread': (float(np.nanstd(held)) if held.size else None),
                'attempted_spread': float(np.nanstd(thres)),
                'updated_iso': iso, 'stats': stats}
            self._log_threshold_rejected(
                name2, reason, stats, thres, current_thres=held,
                seq_no=self._seq_total)

    def _log_threshold_update(self, name, source, n_updated, *,
                              log_infidelities=False, thresholds=None,
                              infidelities=None, seq_no=None):
        """Append one shot-stamped threshold record (full per-site vector) to the
        per-pattern audit log so the drift can be analysed offline. Defaults to
        the img1 EFFECTIVE thresholds; the img2 refit passes its own vectors so
        the img2 pattern's log records img2's calibration."""
        if not name:
            return
        try:
            from yb_analysis.analysis import update_log
            import yb_analysis.analysis.pattern_registry as reg
            thr = np.asarray(self.thresholds if thresholds is None else thresholds,
                             dtype=np.float64).ravel()
            rec = {'scan_id': str(self.scan_id),
                   'seq_no': int(self._seq_total if seq_no is None else seq_no),
                   'pattern': name, 'source': source, 'n_updated': int(n_updated),
                   'mean_thr': float(np.nanmean(thr)) if thr.size else None,
                   'std_thr': float(np.nanstd(thr)) if thr.size else None,
                   'thresholds': thr.tolist()}
            if log_infidelities:
                inf = np.asarray(self.infidelities if infidelities is None
                                 else infidelities, dtype=np.float64).ravel()
                rec['infidelities'] = inf.tolist()
                rec['mean_infidelity'] = (float(np.nanmean(inf))
                                          if inf.size else None)
            update_log.append('thresholds/%s.jsonl' % reg._sanitize_name(name), rec)
        except Exception as e:  # noqa: BLE001
            logger.debug('threshold log failed: %s', e)

    def _log_threshold_rejected(self, name, reason, stats, attempted_thres, *,
                                current_thres=None, seq_no=None):
        """Append a 'fit_rejected' record to the per-pattern audit log when a full
        refit is rejected as degenerate, so the dashboard threshold tab can show
        the rejection (with its reason) and the attempted spread. The per-site
        thresholds are NOT logged (they were not applied). ``current_thres``
        defaults to the img1 EFFECTIVE thresholds; the img2 refit passes its
        own held thresholds."""
        if not name:
            return
        try:
            from yb_analysis.analysis import update_log
            import yb_analysis.analysis.pattern_registry as reg
            cur = np.asarray(self.thresholds if current_thres is None
                             else current_thres, dtype=np.float64).ravel()
            att = np.asarray(attempted_thres, dtype=np.float64).ravel()
            rec = {'scan_id': str(self.scan_id),
                   'seq_no': int(self._seq_total if seq_no is None else seq_no),
                   'pattern': name, 'source': 'fit_rejected', 'n_updated': 0,
                   'reason': str(reason),
                   'mean_thr': float(np.nanmean(cur)) if cur.size else None,
                   'std_thr': float(np.nanstd(cur)) if cur.size else None,
                   'attempted_mean_thr': float(np.nanmean(att)) if att.size else None,
                   'attempted_spread': float(np.nanstd(att)) if att.size else None,
                   'stats': stats}
            update_log.append('thresholds/%s.jsonl' % reg._sanitize_name(name), rec)
        except Exception as e:  # noqa: BLE001
            logger.debug('threshold reject-log failed: %s', e)

    # --- EFFECTIVE properties ---

    @property
    def thresholds(self):
        return self.live_thresholds if self.live_thresholds is not None else self.loaded_thresholds

    def _img2_pattern_name(self):
        """Loading-pattern name of the FINAL (img2) frame, or None."""
        return self._pattern_names.get(max(1, self.num_images_per_seq) - 1)

    def _img2_accum_key(self):
        """In-memory cross-run accumulator key for the img2 frame. Namespaced
        away from the bare pattern name (which the SAME pattern uses as an img1
        LOADING frame in other scans) so img2's post-protocol intensities never
        contaminate that pattern's loading-frame full fit. This key is only ever
        a dict key in _pattern_accum — it is never written to disk."""
        name2 = self._img2_pattern_name()
        return ('%s\x00img2' % name2) if name2 else None

    def _img2_refit_active(self):
        """True when img2 should get its OWN threshold refit: it is a declared
        loading pattern DISTINCT from img1's. When the two frames share a
        pattern, img2 keeps using img1's single live refit (via
        :meth:`_effective_thresholds`), so no separate img2 refit is needed."""
        if not self.is_two_array or self.num_sites_img2 <= 0:
            return False
        p2 = self._img2_pattern_name()
        return bool(p2) and (p2 != self._pattern_names.get(0))

    def _resolve_img2_model(self):
        """Load the img2 spot-shape detector when enabled + available. Sets
        ``self._img2_model`` (an ``ssm`` model dict or None) and the provenance
        tag. The model is the img2 detector whenever we'll save img2 frames;
        falls back to threshold detection if disabled
        (``IMG2_SHAPE_MODEL_VARIANT=''``) or the artifact is missing."""
        self._img2_model = None
        self._img2_logicals_source = None
        variant = (IMG2_SHAPE_MODEL_VARIANT or '').strip()
        if not variant:
            return
        # ONLY for runs where img2 is a DISTINCT loading pattern from img1
        # (same condition as the img2 threshold refit). When both frames share
        # the pattern, img2 keeps using img1's threshold detection unchanged.
        if not (self._img2_refit_active() and self.num_images_per_seq >= 2):
            return
        m = ssm.load_model(variant)
        if not m:
            return
        self._img2_model = m
        self._img2_logicals_source = m['tag']
        logger.info('img2 detection uses spot-shape model %s (box=%d) for '
                    'pattern %s (%d sites)', m['tag'], m['box_size'],
                    self._img2_pattern_name(), self.num_sites_img2)

    def _effective_thresholds(self, pattern_name, loaded):
        """Per-pattern effective thresholds for a frame whose loading pattern
        is ``pattern_name`` and whose stored per-site thresholds are ``loaded``.

        A frame detects with the live thresholds for ITS pattern:
          * the frame-0 (loading) pattern -> img1's live refit, and
          * a DISTINCT img2 pattern -> img2's own live refit (set by
            :meth:`_maybe_refit_img2`).
        Any other pattern (or before its first accepted fit) uses its stored
        per-site thresholds. This keeps thresholds per-pattern and per-site;
        when both frames share the loading pattern a single (img1-driven) refit
        serves both."""
        if (self.live_thresholds is not None and pattern_name is not None
                and pattern_name == self._pattern_names.get(0)):
            return self.live_thresholds
        if (self.live_thresholds_img2 is not None and pattern_name is not None
                and pattern_name == self._img2_pattern_name()
                and len(self.live_thresholds_img2) == len(loaded)):
            return self.live_thresholds_img2
        return loaded

    @property
    def infidelities(self):
        return self.live_infidelities if self.live_infidelities is not None else self.loaded_infidelities

    @property
    def gauss_fits(self):
        return self.live_gauss_fits if self.live_gauss_fits is not None else self.loaded_gauss_fits

    # --- Data flow ---

    def store_new_data(self, info):
        pSeq = max(1, self.num_images_per_seq)
        imgs_in = info['imgs']
        sids_in = info['seq_ids']
        # imgs and seq_ids are paired 1:1 per sequence by the wire decoder;
        # guard the (pathological) length-mismatch case anyway.
        n_seq = min(len(imgs_in), len(sids_in))
        kept = 0
        for i in range(n_seq):
            img3d = imgs_in[i]  # (rows, cols, n_frames) for this sequence
            # Auto-detect frame_size from first image (config may have swapped W/H)
            actual_shape = img3d.shape[:2]
            if actual_shape != self.frame_size and not self._frame_size_fixed:
                logger.info('Fixing frame_size: config=%s, actual=%s', self.frame_size, actual_shape)
                self.frame_size = actual_shape
                buf_size = max(UPDATE_THRES_BATCH_SIZE, UPDATE_GRID_BATCH_SIZE)
                self.img_buffer = RingBuffer(buf_size, self.frame_size, dtype='int16')
                self._frame_size_fixed = True
                # Retry HDF5 creation — initial attempt fails when config has frameSize=(0,0)
                if not self._file_created and self.num_sites > 0:
                    try:
                        create_scan_file(self.fname, self.config, self.frame_size, self.num_sites)
                        self._file_created = True
                        logger.info('HDF5 file created after frame_size fix')
                    except Exception as e:
                        logger.warning('HDF5 retry failed: %s', e)

            # --- Frame-drop SAFETY -----------------------------------------
            # Each sequence must deliver exactly pSeq frames. A short (or extra)
            # block means a camera/compute stall dropped (or duplicated) a frame
            # for THIS sequence. The img1/img2 split downstream is purely by
            # frame parity (idx % pSeq in process_data, [0::2]/[1::2] in
            # save_data), so a single off-count sequence would phase-flip that
            # split for EVERY later sequence and silently scramble survival
            # pairing (img1[k] vs img2[k] taken from different shots), as seen
            # on scan 20260608_111039 (img1=244, img2=186). Drop the whole
            # incomplete sequence — its seq_id too — so frames<->seq_ids stay
            # 1:1 and the parity split stays valid. The dropped shot is already
            # unusable for survival (missing its final image) anyway.
            n_frames = img3d.shape[2]
            if n_frames != pSeq:
                self._dropped_seqs += 1
                if len(self._dropped_seq_ids) < 1000:
                    self._dropped_seq_ids.append(int(sids_in[i]))
                logger.warning(
                    'Dropping seq_id=%s: got %d frame(s), expected pSeq=%d '
                    '(frame drop / stall). %d sequence(s) dropped this scan.',
                    int(sids_in[i]), n_frames, pSeq, self._dropped_seqs)
                continue
            # ----------------------------------------------------------------

            for p in range(pSeq):
                self._imgs_to_process.append(img3d[:, :, p])
            if self.img_buffer is not None:
                self.img_buffer.push(img3d[:, :, 0].astype(np.int16))
            sid = int(sids_in[i])
            if self._seq_ids_max_seen > 0 and sid > self._seq_ids_max_seen + 1:
                gap_start = self._seq_ids_max_seen + 1
                gap_end = sid - 1
                gap_size = gap_end - gap_start + 1
                self._seq_gap_count += gap_size
                if len(self._seq_gap_ids) < 200:
                    self._seq_gap_ids.extend(
                        range(gap_start,
                              min(gap_start + (200 - len(self._seq_gap_ids)),
                                  gap_end + 1)))
                logger.warning(
                    'ZMQ delivery gap: seq_ids %d..%d (%d shot(s)) missing '
                    '(total missing this scan: %d)',
                    gap_start, gap_end, gap_size, self._seq_gap_count)
            if sid > self._seq_ids_max_seen:
                self._seq_ids_max_seen = sid
            self._seq_ids_to_process.append(sid)
            kept += 1

        # Refit / loading / drift cadence counts only the COMPLETE sequences
        # actually stored (not the raw arrivals), so a burst of drops doesn't
        # prematurely trip a refit on too-few real shots.
        self._img_cnt_grid += kept
        self._img_cnt_refit += kept
        self._img_cnt_refit_img2 += kept
        self._img_cnt_loading += kept
        self._img_cnt_affine += kept
        self._img_cnt_thres_live += kept
        self._diag_pull_cnt += kept

    def process_data(self):
        if not self._imgs_to_process:
            return
        if self.is_init:
            for img in self._imgs_to_process:
                self._imgs_to_save.append(img.astype(np.int16))
            # Show the first image of the latest sequence on the dashboard
            pSeq = self.num_images_per_seq
            n_imgs = len(self._imgs_to_process)
            last_seq_start = (n_imgs - 1) // pSeq * pSeq
            self._display_image = self._imgs_to_process[last_seq_start].astype(np.int16)
            self._seq_ids_to_save.extend(self._seq_ids_to_process)
            self._imgs_to_process.clear()
            self._seq_ids_to_process.clear()
            return

        n_new_seqs = 0
        pSeq = self.num_images_per_seq
        seq_logic_buf = []  # collect logicals within one sequence
        batch_sids = []  # seq_ids completed in THIS batch (for "current" highlight)
        for idx, img in enumerate(self._imgs_to_process):
            frame_idx = idx % pSeq            # 0 = initial, pSeq-1 = final
            is_first = frame_idx == 0
            is_last  = pSeq >= 2 and frame_idx == pSeq - 1
            is_mid   = (not is_first) and (not is_last)   # only when pSeq >= 3
            # Two-array mode: the second array layout applies to the FINAL
            # frame (post-rearrangement / post-pushout). Middle frames are
            # still in the same configuration as the initial image, so they
            # use grid_locations.
            proba_vec = None   # img2 per-site posterior for this frame (model path)
            if self.is_two_array and is_last:
                grid_i = self.grid_locations_img2
                # Per-pattern, per-site thresholds. A frame detects with ITS
                # pattern's thresholds. The live Gaussian refit only ever
                # updates the LOADING (frame-0) pattern — img2 is AFTER the
                # protocol and must never drive a refit — so the final frame
                # gets the live thresholds exactly when its pattern IS that
                # loading pattern, otherwise its own stored per-site thresholds.
                thr_i = self._effective_thresholds(
                    self._pattern_names.get(pSeq - 1), self.loaded_thresholds_img2)
                fr = img.astype(np.float64)
                if self._img2_model is not None:
                    # Spot-SHAPE GMM detector (distinct-pattern img2 only):
                    # logicals from the model, not the intensity threshold.
                    # Intensities are still the production masked sum (for
                    # storage / histograms / unchanged analysis), computed in the
                    # SAME vectorised pass. proba_vec = per-site P(loaded).
                    res = None
                    try:
                        res = ssm.detect_frame(self._img2_model, fr, grid_i,
                                               self.mask_mat)
                    except Exception as e:  # noqa: BLE001
                        logger.debug('img2 shape-model detect failed (%s); '
                                     'falling back to threshold', e)
                    if res is not None:
                        logicals, proba_vec, intensities = res
                    else:
                        logicals, intensities = detect_atom(
                            fr, grid_i, thr_i, self.mask_mat)
                else:
                    logicals, intensities = detect_atom(
                        fr, grid_i, thr_i, self.mask_mat)
            else:
                grid_i = self.grid_locations
                thr_i = self.thresholds
                logicals, intensities = detect_atom(
                    img.astype(np.float64), grid_i, thr_i, self.mask_mat)
            self._logicals_to_save.append(logicals)
            self._intensities_to_save.append(intensities)
            self._imgs_to_save.append(img.astype(np.int16))
            seq_logic_buf.append(logicals)

            # On first image of each sequence: accumulate for histograms + display
            if is_first:
                # Blank/dropped frames (whole-frame ~0 — camera/ZMQ returned no real
                # image) carry no atom info; keep them OUT of the threshold-fit
                # accumulators (live + cross-run) and the histograms, else the
                # double-Gaussian refit reads the 0-vs-pedestal toggle as empty-vs-atom
                # and corrupts both the thresholds and the histogram binning.
                if _is_blank_intensities(intensities):
                    self._blank_shot_count += 1
                    if self._blank_shot_count % 20 == 1:
                        logger.warning('Blank/dropped frame skipped from threshold '
                                       'accumulators (max intensity < %.1f); %d so far '
                                       'this scan', BLANK_FRAME_FLOOR,
                                       self._blank_shot_count)
                else:
                    self._intensity_accum.append(intensities.copy())
                    # Also feed the cross-run accumulator for this pattern so the
                    # 200-shot full fit can fire over several short runs (in-memory,
                    # age-windowed; lost on restart).
                    if self._accum_pattern_name:
                        pa = _get_pattern_accum(self._accum_pattern_name, self.num_sites)
                        if pa is not None and intensities.size == pa['num_sites']:
                            with _pattern_accum_lock:
                                pa['entries'].append((time.time(), intensities.copy()))
                                pa['n_since_attempt'] += 1
                                _prune_pattern_accum(pa, time.time())
                self.log_buffer.push(logicals.astype(np.float64))
                record_loading(logicals)
                # Always display image-1 (loading image, not pushout)
                self._display_image = img.astype(np.int16)
                self._display_intensities = intensities.copy()
                self._display_logicals = logicals.copy()
            # Middle frame(s): only meaningful when pSeq >= 3 (e.g. the
            # two-round SLM rearrangement after round 1 but before round
            # 2). Only the most recent middle frame is kept for display.
            if is_mid:
                self._display_image_mid = img.astype(np.int16)
                self._display_intensities_mid = intensities.copy()
                self._display_logicals_mid = logicals.copy()
            # Final image of each sequence: feeds the "image 2" display
            # slot and (for is_two_array) the second log buffer.
            if is_last:
                self._display_image2 = img.astype(np.int16)
                self._display_intensities2 = intensities.copy()
                self._display_logicals2 = logicals.copy()
                self._display_proba2 = (proba_vec.copy()
                                        if proba_vec is not None else None)
                if self.is_two_array and self.log_buffer_img2 is not None:
                    self.log_buffer_img2.push(logicals.astype(np.float64))
                # When the shape model is the img2 detector, buffer its per-site
                # posterior 1:1 with the final-frame logicals for HDF5 (NaN for
                # any frame that fell back to the threshold so the certainties
                # dataset stays shape-aligned with logicals_img2).
                if self._img2_model is not None and self._save_two_array:
                    if (proba_vec is not None
                            and np.size(proba_vec) == self.num_sites_img2):
                        self._proba_img2_to_save.append(
                            np.asarray(proba_vec, dtype=np.float64))
                    else:
                        self._proba_img2_to_save.append(
                            np.full(self.num_sites_img2, np.nan))
                # Accumulate img2 intensities for its OWN Gaussian refit, but
                # only when img2 is a DISTINCT loading pattern from img1 AND the
                # shape model is NOT the active img2 detector (the model
                # supersedes the threshold refit — running it anyway would just
                # log redundant degenerate-fit rejections). Mirrors the img1
                # accumulation, incl. the img2-NAMESPACED cross-run accumulator
                # (_img2_accum_key) so the SAME pattern's post-protocol (img2)
                # intensities never pollute its loading-frame fit elsewhere.
                if (self._img2_model is None and self._img2_refit_active()
                        and intensities.size == self.num_sites_img2
                        and not _is_blank_intensities(intensities)):
                    self._intensity_accum_img2.append(intensities.copy())
                    akey = self._img2_accum_key()
                    if akey:
                        pa = _get_pattern_accum(akey, self.num_sites_img2)
                        if pa is not None and intensities.size == pa['num_sites']:
                            with _pattern_accum_lock:
                                pa['entries'].append((time.time(), intensities.copy()))
                                pa['n_since_attempt'] += 1
                                _prune_pattern_accum(pa, time.time())

            # On last image of each sequence: accumulate for scan curve.
            # logic2 always refers to the FINAL frame's logicals so the
            # survival curve compares initial vs final regardless of pSeq.
            if frame_idx == pSeq - 1:
                seq_idx = idx // pSeq
                sid = int(self._seq_ids_to_process[seq_idx]) if seq_idx < len(self._seq_ids_to_process) else 0
                logic1 = seq_logic_buf[0]
                logic2 = seq_logic_buf[-1] if len(seq_logic_buf) >= 2 else None
                self._scan_logicals.append((sid, logic1.copy(), logic2.copy() if logic2 is not None else None))
                batch_sids.append(sid)
                seq_logic_buf.clear()
                n_new_seqs += 1
        self._last_batch_seq_ids = batch_sids
        self._seq_total += n_new_seqs

        # Rebin histograms from accumulated intensities (cheap: ~0.5ms)
        if len(self._intensity_accum) >= 1:
            self._rebin_histograms()

        # Check 2000-shot rotation
        if len(self._intensity_accum) >= UPDATE_HIST_BATCH_SIZE:
            logger.info('2000-shot cycle — rotating to background')
            self.loaded_gauss_fits = self.live_gauss_fits or self.loaded_gauss_fits
            self.loaded_thresholds = self.thresholds.copy()
            self.loaded_infidelities = self.infidelities.copy()
            self._intensity_accum.clear()
            self.live_hist_data = None
            self.live_gauss_fits = None
            self.live_thresholds = None
            self.live_infidelities = None

        # Same rotation for the img2 accumulator (distinct-pattern refit).
        if len(self._intensity_accum_img2) >= UPDATE_HIST_BATCH_SIZE:
            if self.live_gauss_fits_img2 is not None:
                self.loaded_gauss_fits_img2 = self.live_gauss_fits_img2
            if self.live_thresholds_img2 is not None:
                self.loaded_thresholds_img2 = self.live_thresholds_img2.copy()
            if self.live_infidelities_img2 is not None:
                self.loaded_infidelities_img2 = self.live_infidelities_img2.copy()
            self._intensity_accum_img2.clear()
            self.live_hist_data_img2 = None
            self.live_gauss_fits_img2 = None
            self.live_thresholds_img2 = None
            self.live_infidelities_img2 = None

        self._seq_ids_to_save.extend(self._seq_ids_to_process)
        self._imgs_to_process.clear()
        self._seq_ids_to_process.clear()

    @staticmethod
    def _compute_hist_data(all_i, num_sites):
        """Per-site density histograms ({counts, bin_centers}) over an
        (N, num_sites) intensity stack. Shared by the img1 and img2 refits AND
        read by the full fit as its per-site bins. Two robustness measures:
          * only the most recent THRES_FIT_MAX_SHOTS shots are binned (bounds the
            per-rebuild cost; the cross-run buffer can reach 2000), and
          * each site's 50-bin RANGE is clipped to the HIST_DISPLAY_CLIP_PCT central
            percentile band, so one hot pixel / outlier shot can't stretch the bins
            so wide the empty/atom doublet collapses into a single bar."""
        hist_data = []
        if all_i.shape[0] > THRES_FIT_MAX_SHOTS:
            all_i = all_i[-THRES_FIT_MAX_SHOTS:]
        lo_pct, hi_pct = HIST_DISPLAY_CLIP_PCT
        for s in range(int(num_sites)):
            col = all_i[:, s]
            if col.size:
                lo, hi = np.percentile(col, [lo_pct, hi_pct])
                if not (hi > lo):           # degenerate (all-equal) site
                    lo, hi = float(col.min()) - 0.5, float(col.max()) + 0.5
            else:
                lo, hi = 0.0, 1.0
            counts, edges = np.histogram(col, bins=50, range=(lo, hi), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist_data.append({'counts': counts, 'bin_centers': centers})
        return hist_data

    def _rebin_histograms(self):
        all_i = np.array(self._intensity_accum)  # (N, num_sites)
        self.live_hist_data = self._compute_hist_data(all_i, self.num_sites)

    def update_data(self):
        if self.is_init:
            return

        pattern_active = (self._pattern_grids is not None and self._roi is not None
                          and self._pattern_names.get(0) is not None)

        # --- POSITIONS ---
        # Loading-pattern scan: every N shots nudge the GLOBAL affine TRANSLATION
        # (EWMA, ~100-shot memory) in a background thread and re-derive the live
        # detection grid from it. Legacy day-folder scan: the local-grid drift
        # tracker (full integer shift, every 50) is unchanged.
        if pattern_active:
            if self._img_cnt_affine >= AFFINE_LIVE_INTERVAL:
                self._img_cnt_affine = 0
                self._schedule_affine_live()

        # --- Live target-aware survival: pull per-shot diag targets ---
        # Every ~25 shots, refresh seq_id->target-sites from the SLM diag in a
        # background thread so the live scan curve can show TP (matches the
        # Analysis tab). Backs off after a few empty pulls (non-rearrange runs
        # have no diag), so it's a no-op cost there.
        if (self._diag_pull_cnt >= DIAG_PULL_INTERVAL
                and self._diag_empty_streak < DIAG_PULL_MAX_EMPTY
                and not self._diag_pull_running):
            self._diag_pull_cnt = 0
            self._diag_pull_running = True
            threading.Thread(target=self._pull_live_targets,
                             name='diag-targets-%s' % self.scan_id,
                             daemon=True).start()
        elif self._img_cnt_grid >= UPDATE_GRID_INTERVAL:
            if self.img_buffer and self.img_buffer.size() >= UPDATE_GRID_BATCH_SIZE:
                self._img_cnt_grid = 0
                n = min(self.img_buffer.size(), UPDATE_GRID_BATCH_SIZE)
                imgs = self.img_buffer.get_last_n(n).astype(np.float64)
                grid, _, dy, dx, heatmap = locate_atom_update(
                    imgs, self.grid_locations, 10, self.mask_mat
                )
                self.grid_locations = grid
                self.grid_shift_history.append((dy, dx))
                self.grid_shift_heatmap = heatmap
                # Two-array: inherit the same camera-frame shift on img2 grid
                if self.is_two_array and self.grid_locations_img2 is not None:
                    self.grid_locations_img2 = (
                        self.grid_locations_img2 + np.array([dy, dx]))
                self._save_grid()
                logger.info('Grid updated (dy=%d, dx=%d)', dy, dx)

        # --- THRESHOLDS (cheap, every N shots) ---
        # Track the per-site threshold between full fits: keep it at the last
        # fit's placement ratio between the (drifting) empty/atom population
        # means, EWMA-blended (~100-shot memory). <50 ms; saves + logs each time.
        if pattern_active and self._img_cnt_thres_live >= THRES_LIVE_INTERVAL:
            self._img_cnt_thres_live = 0
            self._update_thresholds_live_cheap()

        # --- THRESHOLDS (full Gaussian refit, every 200 seq) ---
        # The authoritative re-anchor: refits the double Gaussians (true
        # infidelities), replaces the live thresholds, and refreshes the
        # placement ratio + valley clamp band the cheap tracker rides between
        # fits. n_accum counts shots accumulated for THIS pattern, including any
        # carried in from recent runs (cross-run accumulator), so the fit fires
        # over several short runs. A DEGENERATE fit (low-loading / near-unimodal
        # data) is REJECTED: thresholds unchanged, nothing saved, health degraded.
        n_accum = len(self._intensity_accum)
        if self._img_cnt_refit >= UPDATE_THRES_INTERVAL and n_accum >= UPDATE_THRES_INTERVAL:
            self._img_cnt_refit = 0
            if self._accum_pattern_name:   # reset cross-run shots-since-attempt
                pa = _get_pattern_accum(self._accum_pattern_name, self.num_sites)
                with _pattern_accum_lock:
                    pa['n_since_attempt'] = 0
            all_i = np.array(self._intensity_accum)
            fits, thres, inf = self._fit_gaussians(all_i)
            ok, reason, stats = self._validate_full_fit(fits, thres, all_i)
            iso = _now_iso()
            if ok:
                logger.info('Accepted Gaussian refit (%d shots): %s', n_accum, stats)
                self.live_gauss_fits = fits
                self.live_thresholds = thres
                self.live_infidelities = inf
                self._thr_place_ratio = self._placement_ratio(fits, thres)
                self._thr_clamp_lo, self._thr_clamp_hi = self._valley_clamp_band(fits)
                self._thr_fit_anchor = np.asarray(thres, dtype=np.float64).copy()
                self._thr_fit_med_below, self._thr_fit_med_above = \
                    self._fit_median_refs(all_i, thres)
                self._thr_has_accepted_fit = True
                self._threshold_health = {
                    'state': 'ok', 'reason': '', 'source': 'fit',
                    'spread': float(np.nanstd(thres)),
                    'mean_infidelity': (float(np.nanmean(inf))
                                        if inf.size else None),
                    'updated_iso': iso, 'stats': stats}
                self._hist_version += 1
                self._hist_rep_sites = self._pick_rep_sites()
                self._save_threshold()
                self._save_histdata()
                if pattern_active:
                    self._log_threshold_update(
                        self._pattern_names.get(0), 'fit', int(self.num_sites),
                        log_infidelities=True)
            else:
                logger.warning('REJECTED degenerate threshold refit (%d shots): %s '
                               '%s — thresholds unchanged, not saved',
                               n_accum, reason, stats)
                self._threshold_health = {
                    'state': 'degraded', 'reason': reason, 'source': 'fit_rejected',
                    'spread': (float(np.nanstd(self.thresholds))
                               if self.thresholds.size else None),
                    'attempted_spread': float(np.nanstd(thres)),
                    'updated_iso': iso, 'stats': stats}
                if pattern_active:
                    self._log_threshold_rejected(
                        self._pattern_names.get(0), reason, stats, thres)

        # --- THRESHOLDS for img2 (independent full refit, every 200 seq) ---
        # When img2 is a DISTINCT loading pattern from img1, refit ITS thresholds
        # from the final-frame intensities (same guard), saving back to img2's
        # per-pattern store. Skipped when the spot-shape model is img2's detector
        # (it supersedes thresholds) or when img2 shares img1's pattern.
        if self._img2_model is None and self._img2_refit_active():
            self._maybe_refit_img2()

        # Loading rates (every 5 seq)
        if self._img_cnt_loading >= UPDATE_LOADING_INTERVAL:
            if self.log_buffer and self.log_buffer.size() >= UPDATE_LOADING_INTERVAL:
                n = min(self.log_buffer.size(), 200)
                self.loading_rates = self.log_buffer.get_last_n(n).mean(axis=0)
                if (self.is_two_array
                        and self.log_buffer_img2 is not None
                        and self.log_buffer_img2.size() >= UPDATE_LOADING_INTERVAL):
                    n2 = min(self.log_buffer_img2.size(), 200)
                    self.loading_rates_img2 = (
                        self.log_buffer_img2.get_last_n(n2).mean(axis=0))
                self._img_cnt_loading = 0

    def _fit_gaussians(self, all_intensities, hist_data=None):
        from scipy.optimize import least_squares, minimize_scalar
        from scipy.special import ndtr   # standard-normal CDF; ~2.5x faster than norm.cdf
        from yb_analysis.detection.dynamical_threshold import _gauss_pdf

        # Per-site bin centres/counts to fit against. Defaults to the img1 live
        # histograms (self.live_hist_data); the img2 refit passes its own.
        hd = hist_data if hist_data is not None else self.live_hist_data

        # Fit only the most recent THRES_FIT_MAX_SHOTS shots — the per-site fit is the
        # dominant cost and a few hundred shots already resolve the doublet; this also
        # keeps the avg-prior / per-site-stat passes from scaling with the (up to
        # 2000-shot) cross-run buffer. Matches the cap _compute_hist_data uses for hd.
        if all_intensities.shape[0] > THRES_FIT_MAX_SHOTS:
            all_intensities = all_intensities[-THRES_FIT_MAX_SHOTS:]

        M = all_intensities.shape[1]
        fits, thres, inf = [], np.zeros(M), np.zeros(M)

        # Manual Gaussian PDF (no scipy.stats.norm input-validation overhead, paid on
        # every residual eval across thousands of sites) -- ~2.5x faster, identical math.
        def two_g(p, x):
            return p[2]*_gauss_pdf(x, p[0], p[1]) + p[5]*_gauss_pdf(x, p[3], p[4])

        # Step 1: Fit the AVERAGE histogram to get global priors
        all_vals = all_intensities.ravel()
        avg_ct, avg_edges = np.histogram(all_vals, bins=50, density=True)
        avg_bc = 0.5 * (avg_edges[:-1] + avg_edges[1:])
        avg_mn, avg_mx, avg_sd = all_vals.min(), all_vals.max(), all_vals.std()
        avg_p25, avg_p75 = np.percentile(all_vals, [25, 75])

        avg_x0 = [avg_p25, avg_sd/4, 0.3, avg_p75, avg_sd/4, 0.3]
        avg_lb = [avg_mn, avg_sd/50, 0.01, np.median(all_vals), avg_sd/50, 0.01]
        avg_ub = [np.median(all_vals), avg_sd*2, 1.0, avg_mx, avg_sd*2, 1.0]

        try:
            avg_res = least_squares(lambda p: two_g(p, avg_bc) - avg_ct,
                                     avg_x0, bounds=(avg_lb, avg_ub), method='trf')
            avg_p = avg_res.x
            if avg_p[0] > avg_p[3]:
                avg_p = np.array([avg_p[3], avg_p[4], avg_p[5], avg_p[0], avg_p[1], avg_p[2]])
            logger.info('Avg fit: mu_e=%.2f sig_e=%.2f mu_a=%.2f sig_a=%.2f',
                        avg_p[0], avg_p[1], avg_p[3], avg_p[4])
        except Exception:
            avg_p = None
            logger.warning('Average histogram fit failed, using percentile guesses')

        # Step 2: Fit each site using avg fit as initial guess
        for s in range(M):
            vals = all_intensities[:, s]
            mn, mx, sd = vals.min(), vals.max(), vals.std()
            if sd < 1e-10:
                fits.append({'params': None})
                thres[s], inf[s] = np.median(vals), np.nan
                continue

            # Use avg fit as initial guess (much better than per-site percentiles)
            if avg_p is not None:
                x0 = [avg_p[0], avg_p[1], avg_p[2], avg_p[3], avg_p[4], avg_p[5]]
                spread = avg_p[3] - avg_p[0]  # separation between peaks
                lb = [avg_p[0] - spread, avg_p[1]/5, 0.01, avg_p[3] - spread, avg_p[4]/5, 0.01]
                ub = [avg_p[0] + spread, avg_p[1]*5, 1.0, avg_p[3] + spread, avg_p[4]*5, 1.0]
            else:
                p25, p50, p75 = np.percentile(vals, [25, 50, 75])
                rng = mx - mn
                x0 = [p25, sd/4, 0.3, p75, sd/4, 0.3]
                lb = [mn - 0.1*rng, sd/50, 0.01, p50, sd/50, 0.01]
                ub = [p50, sd*2, 1.0, mx + 0.1*rng, sd*2, 1.0]

            h = hd[s] if (hd and s < len(hd)) else None
            if h:
                bc, ct = h['bin_centers'], h['counts']
            else:
                ct, edges = np.histogram(vals, bins=50, density=True)
                bc = 0.5 * (edges[:-1] + edges[1:])

            try:
                res = least_squares(lambda p: two_g(p, bc) - ct, x0, bounds=(lb, ub), method='trf')
                p = res.x
                if p[0] > p[3]:
                    p = np.array([p[3], p[4], p[5], p[0], p[1], p[2]])
                opt = minimize_scalar(
                    lambda xc: (1 - ndtr((xc - p[0]) / p[1])) + ndtr((xc - p[3]) / p[4]),
                    bounds=(p[0], p[3]), method='bounded'
                )
                fits.append({'params': p})
                thres[s], inf[s] = opt.x, opt.fun
            except Exception:
                fits.append({'params': None})
                thres[s], inf[s] = np.median(vals), np.nan

        return fits, thres, inf

    def _pick_rep_sites(self):
        if self.num_sites < 4:
            return list(range(self.num_sites))
        inf = self.infidelities.copy()
        inf[np.isnan(inf)] = np.inf
        best = int(np.argmin(inf))
        valid = self.infidelities[~np.isnan(self.infidelities)]
        worst = int(np.argmax(valid)) if len(valid) > 0 else 0
        others = [i for i in range(self.num_sites) if i not in (best, worst)]
        rng = np.random.default_rng()
        rand = rng.choice(others, size=min(2, len(others)), replace=False).tolist() if others else []
        sites = [best, worst] + rand
        while len(sites) < 4 and len(sites) < self.num_sites:
            for i in range(self.num_sites):
                if i not in sites:
                    sites.append(i)
                    break
        return sites

    # --- Save ---

    def _save_block(self, do_append, sids, n_frames, two_array=False):
        """Run the HDF5 append (on the daemon save thread), recording
        ``save_health`` on failure instead of letting the thread die silently.

        ``append_block`` already retries transient OS locks; this is the
        last-resort surface for when even the retries are exhausted — the block
        is lost, but the operator sees it (Live view turns red) rather than the
        old behaviour where the exception printed to the log and vanished.
        """
        # Fetch (lazily create) the health dict once — robust to any
        # construction path (e.g. tests that build a bare DM via __new__) and
        # keeps the except branch from double-faulting.
        sh = getattr(self, '_save_health', None)
        if not isinstance(sh, dict):
            sh = self._save_health = {'state': 'ok', 'reason': '', 'lost_seqs': 0,
                                      'lost_seq_ids': [], 'last_error': '',
                                      'updated_iso': None}
        try:
            with self._save_lock:
                do_append()
            logger.info('Saved %d frames%s', n_frames,
                        ' (two-array)' if two_array else '')
            # Writes are flowing again after an earlier failure: drop the active
            # 'fail' state but keep the lost tally so the warning persists.
            if sh.get('state') == 'fail':
                sh['state'] = 'recovered'
                sh['reason'] = ('HDF5 saves resumed; %d sequence(s) lost earlier'
                                % int(sh.get('lost_seqs', 0)))
                sh['updated_iso'] = _now_iso()
        except Exception as e:
            ids = [int(s) for s in list(sids)] if sids is not None else []
            n = len(ids)
            sh['state'] = 'fail'
            sh['lost_seqs'] = int(sh.get('lost_seqs', 0)) + n
            kept = sh.setdefault('lost_seq_ids', [])
            room = max(0, 1000 - len(kept))
            if room:
                kept.extend(ids[:room])
            sh['last_error'] = '%s: %s' % (type(e).__name__, e)
            sh['reason'] = ('HDF5 save failed — %d sequence(s) lost so far (%s)'
                            % (sh['lost_seqs'], sh['last_error']))
            sh['updated_iso'] = _now_iso()
            logger.error('HDF5 save FAILED for %d sequence(s) (seq_ids %s%s): %s '
                         '— this block is LOST.', n, ids[:12],
                         '...' if n > 12 else '', e)

    def save_data(self):
        if not self._imgs_to_save or not self._file_created:
            return self.fname
        imgs = np.array(self._imgs_to_save, dtype=np.int16)
        sids = np.array(self._seq_ids_to_save, dtype=np.int64)

        if self._save_two_array:
            # Demux the interleaved per-frame buffer into per-sequence rows:
            # img1 = first frame of each sequence ([0::pSeq]); img2 = FINAL
            # frame ([pSeq-1::pSeq]), matching process_data's is_first/is_last.
            # Middle frames (pSeq>=3, two-round rearrangement) stay in `imgs`
            # but are not part of the per-sequence logicals datasets. Keying on
            # pSeq (not a hard-coded 0::2 / 1::2) keeps img1/img2 paired for
            # pSeq>=3 too — the old 1::2 picked the MIDDLE frame as img2.
            pSeq = max(1, self.num_images_per_seq)
            # Safety: store_new_data only admits whole sequences, so the buffer
            # length should be a multiple of pSeq. If a partial sequence ever
            # slips in, trim the orphan tail (loudly) so the parity demux can't
            # phase-flip img1<->img2 for every subsequent block (the failure
            # mode behind scan 20260608_111039's img1=244 / img2=186 split).
            if self._logicals_to_save:
                n_full = (len(self._logicals_to_save) // pSeq) * pSeq
                if n_full != len(self._logicals_to_save):
                    logger.warning(
                        'save_data: %d orphan frame(s) in two-array buffer '
                        '(len=%d, pSeq=%d) — trimming tail to keep img1/img2 in '
                        'phase.', len(self._logicals_to_save) - n_full,
                        len(self._logicals_to_save), pSeq)
                logs_all = self._logicals_to_save[:n_full]
                ints_all = self._intensities_to_save[:n_full]
                imgs = imgs[:n_full]
            else:
                # No logicals (defensive) — let the zeros-branch below size
                # itself from `imgs`; never trim `imgs` to zero here.
                logs_all = []
                ints_all = []
            logs1 = (np.array(logs_all[0::pSeq], dtype=bool)
                     if logs_all
                     else np.zeros((len(imgs) // pSeq, max(self.num_sites, 1)),
                                   dtype=bool))
            logs2 = (np.array(logs_all[pSeq - 1::pSeq], dtype=bool)
                     if logs_all
                     else np.zeros((len(imgs) // pSeq,
                                    max(self.num_sites_img2, 1)), dtype=bool))
            ints1 = (np.array(ints_all[0::pSeq], dtype=np.float64)
                     if ints_all
                     else np.zeros_like(logs1, dtype=np.float64))
            ints2 = (np.array(ints_all[pSeq - 1::pSeq], dtype=np.float64)
                     if ints_all
                     else np.zeros_like(logs2, dtype=np.float64))
            # img2 per-site posterior "% certainty" from the shape model, one
            # row per sequence aligned 1:1 with logs2 (None when the threshold
            # detector was used -> no certainties dataset). Materialise before
            # the save thread so clearing the buffer below is safe.
            proba2 = None
            if self._proba_img2_to_save:
                k = logs2.shape[0]
                pr = self._proba_img2_to_save[:k]
                if 0 < len(pr) < k:        # defensive pad (shouldn't happen)
                    pr = pr + [np.full(self.num_sites_img2, np.nan)] * (k - len(pr))
                if pr:
                    proba2 = np.array(pr, dtype=np.float64)

            def _do():
                self._save_block(
                    lambda: append_block(
                        self.fname, imgs, logs1, ints1, sids,
                        logicals_img2_block=logs2,
                        intensities_img2_block=ints2,
                        proba_img2_block=proba2,
                    ),
                    sids, len(imgs), two_array=True)
        else:
            logs = (np.array(self._logicals_to_save, dtype=bool)
                    if self._logicals_to_save
                    else np.zeros((len(imgs), max(self.num_sites, 1)),
                                  dtype=bool))
            ints = (np.array(self._intensities_to_save, dtype=np.float64)
                    if self._intensities_to_save
                    else np.zeros_like(logs, dtype=np.float64))

            def _do():
                self._save_block(
                    lambda: append_block(self.fname, imgs, logs, ints, sids),
                    sids, len(imgs))

        threading.Thread(target=_do, daemon=True).start()
        self._imgs_to_save.clear()
        self._logicals_to_save.clear()
        self._intensities_to_save.clear()
        self._seq_ids_to_save.clear()
        self._proba_img2_to_save.clear()
        return self.fname

    def _schedule_slm_sync(self):
        """Phase 2: fire-and-forget background sync of the SLM PC's
        per-scan diag ledger + code-snapshot manifest into the scan's
        directory. Called once from `get_data_manager`'s eviction path
        when this scan_id is about to be discarded (scan finished).

        Runs only when ``sync_after_finish`` is True (the class default).
        Independent of the HDF5 final-save thread — different files,
        different network path.
        """
        if not self.sync_after_finish:
            return
        if not self.fname or not self._file_created:
            # Init scans / scans that never created an HDF5 don't get a
            # sidecar — there's no scan_dir to put it in.
            return
        scan_dir = os.path.dirname(self.fname)
        try:
            from yb_analysis.slm_sync import sync as _slm_sync_mod
            _slm_sync_mod.sync_scan_async(self.scan_id, scan_dir)
        except Exception:
            logger.exception('slm_sync schedule failed for scan %d',
                             self.scan_id)

    def _save_grid(self):
        try:
            np.savetxt(os.path.join(self._day_dir, 'gridLocations.txt'),
                       self.grid_locations, header='Y\tX', delimiter='\t', comments='')
        except Exception as e:
            logger.warning('Grid save failed: %s', e)
        if self.is_two_array and self.grid_locations_img2 is not None:
            try:
                np.savetxt(os.path.join(self._day_dir, 'gridLocations_img2.txt'),
                           self.grid_locations_img2, header='Y\tX',
                           delimiter='\t', comments='')
            except Exception as e:
                logger.warning('Grid img2 save failed: %s', e)

    def _save_threshold(self):
        try:
            from scipy.io import savemat
            # Preserve the best-available Gaussian fits: the live fit when present,
            # else the loaded (previous full-fit) ones. The cheap inter-fit
            # threshold update runs BEFORE the first full fit (live_gauss_fits is
            # None), and must NOT wipe the per-pattern gaussFitsStruct the
            # dashboard/analysis rely on -- so use self.gauss_fits, not the live
            # fits alone.
            gfsrc = self.gauss_fits
            gs = np.empty(self.num_sites, dtype=[('params', 'O')])
            for s in range(self.num_sites):
                p = (gfsrc[s].get('params')
                     if (gfsrc is not None and s < len(gfsrc)) else None)
                gs[s]['params'] = p if p is not None else np.array([])
            mat = {
                'thresholds': self.thresholds,
                'infidelities': self.infidelities,
                'gaussFitsStruct': gs,
            }
            savemat(os.path.join(self._day_dir, 'threshold.mat'), mat)
            # Per-pattern persistence (loading-pattern affine migration): save
            # the live-refined thresholds back to the frame-0 pattern so it
            # reloads them next time (self-calibrating per pattern; daily drift
            # absorbed by the live refit).
            if self._pattern_grids is not None and self._pattern_names.get(0):
                import yb_analysis.analysis.pattern_registry as reg
                reg.save_pattern_thresholds(self._pattern_names[0], mat)
        except Exception as e:
            logger.warning('Threshold save failed: %s', e)

    def _save_threshold_img2(self):
        """Persist img2's live-refined thresholds to img2's per-pattern store
        (and the day-folder threshold_img2.mat the legacy two-array path reads).
        Mirrors _save_threshold but for the img2 frame — it NEVER touches img1's
        day-folder threshold.mat or the frame-0 pattern store."""
        name2 = self._img2_pattern_name()
        n = int(self.num_sites_img2)
        if n <= 0:
            return
        try:
            from scipy.io import savemat
            thr = (self.live_thresholds_img2 if self.live_thresholds_img2 is not None
                   else self.loaded_thresholds_img2)
            inf = (self.live_infidelities_img2 if self.live_infidelities_img2 is not None
                   else self.loaded_infidelities_img2)
            gfsrc = (self.live_gauss_fits_img2 if self.live_gauss_fits_img2 is not None
                     else self.loaded_gauss_fits_img2)
            gs = np.empty(n, dtype=[('params', 'O')])
            for s in range(n):
                p = (gfsrc[s].get('params')
                     if (gfsrc is not None and s < len(gfsrc)) else None)
                gs[s]['params'] = p if p is not None else np.array([])
            mat = {
                'thresholds': np.asarray(thr, dtype=np.float64).ravel(),
                'infidelities': np.asarray(
                    inf if inf is not None else np.full(n, np.nan),
                    dtype=np.float64).ravel(),
                'gaussFitsStruct': gs,
            }
            savemat(os.path.join(self._day_dir, 'threshold_img2.mat'), mat)
            if name2:
                import yb_analysis.analysis.pattern_registry as reg
                reg.save_pattern_thresholds(name2, mat)
        except Exception as e:
            logger.warning('img2 threshold save failed: %s', e)

    def _save_histdata(self):
        if not self.live_hist_data:
            return
        try:
            from scipy.io import savemat
            hs = np.empty(self.num_sites, dtype=[('counts', 'O'), ('bin_centers', 'O')])
            for s in range(self.num_sites):
                if s < len(self.live_hist_data):
                    hs[s]['counts'] = self.live_hist_data[s]['counts']
                    hs[s]['bin_centers'] = self.live_hist_data[s]['bin_centers']
                else:
                    hs[s]['counts'] = hs[s]['bin_centers'] = np.array([])
            savemat(os.path.join(self._day_dir, 'histData.mat'), {'histData': hs})
        except Exception as e:
            logger.warning('HistData save failed: %s', e)

    # --- Plot data ---

    def get_plot_data(self):
        return {
            'scan_id': self.scan_id,
            'cur_image': self._display_image.astype(np.float64) if self._display_image is not None else None,
            'cur_intensities': self._display_intensities,
            'logicals': self._display_logicals,
            'cur_image2': self._display_image2.astype(np.float64) if self._display_image2 is not None else None,
            'cur_intensities2': self._display_intensities2,
            'logicals2': self._display_logicals2,
            # img2 detector provenance + mean per-site posterior "% certainty"
            # (only when the spot-shape GMM is img2's detector; None ->
            # thresholds). The full per-site certainties persist to HDF5
            # (certainties_img2); the live snapshot ships only the scalar mean.
            'logicals2_source': self._img2_logicals_source,
            'logicals2_certainty_mean': (
                float(np.nanmean(self._display_proba2))
                if self._display_proba2 is not None
                and np.size(self._display_proba2) else None),
            'cur_image_mid': self._display_image_mid.astype(np.float64) if self._display_image_mid is not None else None,
            'cur_intensities_mid': self._display_intensities_mid,
            'logicals_mid': self._display_logicals_mid,
            'num_images': self.num_images_per_seq,
            'is_two_array': self.is_two_array,
            'grid_locations_img2': self.grid_locations_img2.copy()
                if self.grid_locations_img2 is not None else None,
            'num_sites_img2': self.num_sites_img2,
            # img2 EFFECTIVE thresholds: the live refit (distinct-pattern img2)
            # when one has been accepted, else the loaded per-pattern store.
            'thresholds_img2': (self.live_thresholds_img2
                                if self.live_thresholds_img2 is not None
                                else self.loaded_thresholds_img2),
            'infidelities_img2': (self.live_infidelities_img2
                                  if self.live_infidelities_img2 is not None
                                  else self.loaded_infidelities_img2),
            'loaded_gauss_fits_img2': (self.live_gauss_fits_img2
                                       if self.live_gauss_fits_img2 is not None
                                       else self.loaded_gauss_fits_img2),
            'loaded_hist_data_img2': (self.live_hist_data_img2
                                      if self.live_hist_data_img2 is not None
                                      else self.loaded_hist_data_img2),
            'loading_rates_img2': self.loading_rates_img2.copy()
                if self.loading_rates_img2 is not None else None,
            # img2 independent-refit status (distinct loading pattern only).
            'active_pattern_img2': (self._img2_pattern_name()
                                    if self._img2_refit_active() else None),
            'threshold_health_img2': self._threshold_health_img2,
            'shots_since_fit_img2': int(self._img_cnt_refit_img2),
            'next_fit_in_img2': max(
                0, UPDATE_THRES_INTERVAL - int(self._img_cnt_refit_img2)),
            'grid_locations': self.grid_locations.copy() if len(self.grid_locations) > 0 else None,
            'box_size': self.mask_mat.shape[0],
            'num_sites': self.num_sites,
            'thresholds': self.thresholds.copy(),
            'infidelities': self.infidelities.copy(),
            'loading_rates': self.loading_rates.copy(),
            'loading_history': get_loading_history(),
            # Per-shot survival (TP) for the 0d Scan-Curve timeseries — shows
            # survival instead of loading, target-aware when diag targets known.
            'survival_history': self._per_shot_survival_series(),
            'loaded_gauss_fits': self.loaded_gauss_fits,
            'live_hist_data': self.live_hist_data,
            'live_gauss_fits': self.live_gauss_fits,
            'hist_version': self._hist_version,
            'hist_rep_sites': self._hist_rep_sites,
            'n_accum_shots': len(self._intensity_accum),
            # Per-RUN shot count for the status "shot #" display: resets each scan
            # (new DataManager). NOT len(_intensity_accum) above — that's the live
            # histogram / fit accumulator, which the cross-run seeding carries
            # between short runs of the same loading pattern (so it would otherwise
            # make the displayed shot # "keep ticking up" across rearrange runs).
            'shots_this_run': int(self._seq_total),
            # Detection-threshold calibration health + cadence (threshold tab).
            'threshold_health': self._threshold_health,
            # HDF5 save health (Live status strip): turns the save tile red when
            # an append_block ultimately failed and a block of shots was lost.
            'save_health': self._save_health,
            # Loading-phase health (Live status strip "phase" tile): red when a
            # declared loading phase file is missing on the SLM server, yellow
            # when a pattern lacks an expConfig (ByPattern) entry. None = the
            # scan declared no loading pattern.
            'pattern_health': self._pattern_health,
            'seq_reconciliation': {
                'max_seen': getattr(self, '_seq_ids_max_seen', 0),
                'gap_count': getattr(self, '_seq_gap_count', 0),
                'gap_ids': list(getattr(self, '_seq_gap_ids', [])[:50]),
            },
            'active_pattern': self._pattern_names.get(0),
            'active_defocus': self._active_defocus,
            'blank_shots': int(self._blank_shot_count),
            'shots_since_fit': int(self._img_cnt_refit),
            'next_fit_in': max(0, UPDATE_THRES_INTERVAL - int(self._img_cnt_refit)),
            'grid_shift_heatmap': self.grid_shift_heatmap.copy() if self.grid_shift_heatmap is not None else None,
            'grid_shift_history': list(self.grid_shift_history),
            'scan_curve': compute_scan_curve(
                self._scan_logicals, self._param_indices,
                self._scan_params, self.num_images_per_seq,
                scan_dims=self._scan_dims,
                is_two_array=self.is_two_array,
                recent_seq_ids=self._last_batch_seq_ids,
                seq_targets=self._seq_targets),
            'scan_name': self._scan_name,
            'scan_param_path': self._scan_param_path,
            'plot_scale': self._plot_scale,
            'scan_filename': os.path.basename(self.fname) if self.fname else None,
        }


def _extract_scan_title(config):
    """Extract the scan name from ScanName.scanname (MATLAB uint16 char array)."""
    sn = config.get('ScanName')
    if not isinstance(sn, dict):
        return None
    raw = sn.get('scanname')
    if raw is None:
        return None
    try:
        return ''.join(chr(int(c)) for c in np.asarray(raw).ravel() if c > 0)
    except Exception:
        return None


def _gaussian_mask(box_size, sigma):
    """Create Gaussian mask matching MATLAB's YbHistInit.m:
    maskMat = zeros(boxSize); maskMat(center,center) = 1; maskMat = imgaussfilt(maskMat, sigma);
    """
    from scipy.ndimage import gaussian_filter
    mask = np.zeros((box_size, box_size))
    center = box_size // 2
    mask[center, center] = 1.0
    mask = gaussian_filter(mask, sigma)
    return mask
