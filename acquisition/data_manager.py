"""DataManager: image acquisition, atom detection, histogram lifecycle, and saving.

State model (3 layers):
  LOADED  — from disk at startup, read-only during scan
  LIVE    — accumulates during scan, resets every 2000 shots
  EFFECTIVE — what detect_atom and the dashboard actually use
"""

import os
import logging
import threading
import collections
import numpy as np

from yb_analysis.config import (
    UPDATE_GRID_INTERVAL, UPDATE_GRID_BATCH_SIZE,
    UPDATE_THRES_INTERVAL, UPDATE_THRES_BATCH_SIZE,
    UPDATE_LOADING_INTERVAL, UPDATE_HIST_BATCH_SIZE,
)
from yb_analysis.detection.detect_atom import detect_atom
from yb_analysis.detection.scan_analysis import (
    extract_scan_params, extract_scan_params_h5, extract_scan_name,
    extract_scan_dims, extract_scan_dims_h5, compute_scan_curve,
)
from yb_analysis.detection.locate_atom import locate_atom_update
from yb_analysis.detection.buffers import RingBuffer
from yb_analysis.io.scan_directory import make_scan_dir, make_scan_fname, scan_id_to_stamps
from yb_analysis.io.hdf5_store import create_scan_file, append_block
from yb_analysis.io.mat_reader import load_scan_config_from_mat

logger = logging.getLogger(__name__)

_cache = {}
_cache_lock = threading.Lock()
# Serialises post-scan affine updates (one global affine file).
_AFFINE_UPDATE_LOCK = threading.Lock()

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
        self._roi = None                # [Xoff, Yoff, W, H] for this scan
        self._affine_grid0 = None       # frame-0 affine-predicted grid (pre-drift)

        # Paths
        date_stamp, time_stamp = scan_id_to_stamps(scan_id)
        self.dname, self.date, self.time = make_scan_dir(date_stamp, time_stamp)
        mat_fname, _, _ = make_scan_fname(date_stamp, time_stamp, self.dname)
        self.fname = os.path.splitext(mat_fname)[0] + '.h5'
        self._day_dir = os.path.dirname(self.dname)  # Data/YYYYMMDD/

        # Load scan config
        self.config = load_scan_config_from_mat(mat_fname)
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
        self._hist_version = 0              # incremented on each Gaussian refit
        self._hist_rep_sites = self._pick_rep_sites()

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
        if self.num_sites > 0:
            try:
                create_scan_file(
                    self.fname, self.config, self.frame_size, self.num_sites,
                    two_array=self._save_two_array,
                    num_sites_img2=self.num_sites_img2,
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
        self._hist_version = 0
        self._hist_rep_sites = []
        self.loading_rates = np.array([])
        self.grid_shift_history = []
        self.grid_shift_heatmap = None
        self._scan_logicals = []
        self._last_batch_seq_ids = []
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
                        }
                if any(specs):
                    return specs

        wk = cfg.get('warmup_kwargs')
        if isinstance(wk, dict):
            init_p = _norm(wk.get('initial_phase'))
            final_p = _norm(wk.get('final_phase'))
            extras = wk.get('extras') if isinstance(wk.get('extras'), dict) else {}

            def _spec(path, zern):
                bz = _zern_list(zern)
                return {'name': os.path.splitext(os.path.basename(path))[0],
                        'base_phase_path': path, 'zernike': None, 'order': 'col',
                        'legacy_zerniked': bz is not None, 'baked_zernike': bz}
            if init_p:
                specs = [None] * pSeq
                specs[0] = _spec(init_p, extras.get('initial_phase_zernike'))
                if final_p and pSeq >= 2:
                    specs[pSeq - 1] = _spec(
                        final_p, extras.get('final_phase_zernike'))
                return specs
        return None

    def _default_thresholds(self, n):
        """Length-n initial thresholds when no per-pattern thresholds exist
        yet (a new pattern with a different site count). Broadcasts the median
        of the loaded thresholds; live Gaussian refitting replaces these
        within ~200 shots. Per-pattern stored thresholds are a follow-up."""
        base = np.asarray(self.loaded_thresholds, dtype=np.float64).ravel()
        base = base[np.isfinite(base)]
        val = float(np.median(base)) if base.size else 0.0
        return np.full(int(n), val, dtype=np.float64)

    def _build_pattern_grids(self):
        """Replace the day-folder grid with per-image loading-pattern grids:
        each pattern's simulated knm positions mapped through the global
        affine and the per-scan crop ROI. Sets frame-0 -> grid_locations and
        the final frame -> grid_locations_img2 so the EXISTING per-frame
        selection + live drift correction apply unchanged. No-ops (keeps the
        day grid) if no pattern/ROI/affine is available."""
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
                rec = reg.fetch_or_refresh_pattern(
                    s['name'], base_phase_path=s['base_phase_path'],
                    default_loading_zernike=s.get('zernike'),
                    order=s.get('order', 'col'),
                    legacy_zerniked=s.get('legacy_zerniked', False),
                    baked_zernike=s.get('baked_zernike'))
            except Exception as e:
                logger.warning('pattern %s fetch failed (%s); trying cache',
                               s['name'], e)
                rec = reg.get_pattern(s['name'])
            if not rec or not rec.get('knm'):
                logger.warning('no registry record for pattern %s', s['name'])
                continue
            knm = np.asarray(rec['knm'], dtype=np.float64)
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

    # --- EFFECTIVE properties ---

    @property
    def thresholds(self):
        return self.live_thresholds if self.live_thresholds is not None else self.loaded_thresholds

    @property
    def infidelities(self):
        return self.live_infidelities if self.live_infidelities is not None else self.loaded_infidelities

    @property
    def gauss_fits(self):
        return self.live_gauss_fits if self.live_gauss_fits is not None else self.loaded_gauss_fits

    # --- Data flow ---

    def store_new_data(self, info):
        n_seq = len(info['seq_ids'])
        for i in range(n_seq):
            img3d = info['imgs'][i]  # (rows, cols, pSeq)
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
            for p in range(img3d.shape[2]):
                self._imgs_to_process.append(img3d[:, :, p])
            if self.img_buffer is not None:
                self.img_buffer.push(img3d[:, :, 0].astype(np.int16))
        self._seq_ids_to_process.extend(info['seq_ids'])
        self._img_cnt_grid += n_seq
        self._img_cnt_refit += n_seq
        self._img_cnt_loading += n_seq

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
            if self.is_two_array and is_last:
                grid_i = self.grid_locations_img2
                thr_i = self.loaded_thresholds_img2
            else:
                grid_i = self.grid_locations
                thr_i = self.thresholds
            logicals, intensities = detect_atom(
                img.astype(np.float64), grid_i, thr_i, self.mask_mat
            )
            self._logicals_to_save.append(logicals)
            self._intensities_to_save.append(intensities)
            self._imgs_to_save.append(img.astype(np.int16))
            seq_logic_buf.append(logicals)

            # On first image of each sequence: accumulate for histograms + display
            if is_first:
                self._intensity_accum.append(intensities.copy())
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
                if self.is_two_array and self.log_buffer_img2 is not None:
                    self.log_buffer_img2.push(logicals.astype(np.float64))

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

        self._seq_ids_to_save.extend(self._seq_ids_to_process)
        self._imgs_to_process.clear()
        self._seq_ids_to_process.clear()

    def _rebin_histograms(self):
        all_i = np.array(self._intensity_accum)  # (N, num_sites)
        hist_data = []
        for s in range(self.num_sites):
            counts, edges = np.histogram(all_i[:, s], bins=50, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist_data.append({'counts': counts, 'bin_centers': centers})
        self.live_hist_data = hist_data

    def update_data(self):
        if self.is_init:
            return

        # Grid (every 50 seq)
        if self._img_cnt_grid >= UPDATE_GRID_INTERVAL:
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

        # Gaussian refit (every 200 seq, NOT before 200)
        n_accum = len(self._intensity_accum)
        if self._img_cnt_refit >= UPDATE_THRES_INTERVAL and n_accum >= UPDATE_THRES_INTERVAL:
            self._img_cnt_refit = 0
            logger.info('Refitting Gaussians (%d shots)', n_accum)
            all_i = np.array(self._intensity_accum)
            fits, thres, inf = self._fit_gaussians(all_i)
            self.live_gauss_fits = fits
            self.live_thresholds = thres
            self.live_infidelities = inf
            self._hist_version += 1
            self._hist_rep_sites = self._pick_rep_sites()
            self._save_threshold()
            self._save_histdata()

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

    def _fit_gaussians(self, all_intensities):
        from scipy.optimize import least_squares, minimize_scalar
        from scipy.stats import norm

        M = all_intensities.shape[1]
        fits, thres, inf = [], np.zeros(M), np.zeros(M)

        def two_g(p, x):
            return p[2]*norm.pdf(x, p[0], p[1]) + p[5]*norm.pdf(x, p[3], p[4])

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

            h = self.live_hist_data[s] if self.live_hist_data else None
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
                    lambda xc: (1 - norm.cdf(xc, p[0], p[1])) + norm.cdf(xc, p[3], p[4]),
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

    def save_data(self):
        if not self._imgs_to_save or not self._file_created:
            return self.fname
        imgs = np.array(self._imgs_to_save, dtype=np.int16)
        sids = np.array(self._seq_ids_to_save, dtype=np.int64)

        if self._save_two_array:
            # Demux the interleaved per-frame lists into per-image-per-sequence
            # arrays. Frame indices 0,2,4,... are img1; 1,3,5,... are img2.
            # See plan: HDF5 two-array layout.
            logs1 = (np.array(self._logicals_to_save[0::2], dtype=bool)
                     if self._logicals_to_save
                     else np.zeros((len(imgs) // 2, max(self.num_sites, 1)),
                                   dtype=bool))
            logs2 = (np.array(self._logicals_to_save[1::2], dtype=bool)
                     if self._logicals_to_save
                     else np.zeros((len(imgs) // 2,
                                    max(self.num_sites_img2, 1)), dtype=bool))
            ints1 = (np.array(self._intensities_to_save[0::2], dtype=np.float64)
                     if self._intensities_to_save
                     else np.zeros_like(logs1, dtype=np.float64))
            ints2 = (np.array(self._intensities_to_save[1::2], dtype=np.float64)
                     if self._intensities_to_save
                     else np.zeros_like(logs2, dtype=np.float64))

            def _do():
                with self._save_lock:
                    append_block(
                        self.fname, imgs, logs1, ints1, sids,
                        logicals_img2_block=logs2,
                        intensities_img2_block=ints2,
                    )
                    logger.info('Saved %d frames (two-array)', len(imgs))
        else:
            logs = (np.array(self._logicals_to_save, dtype=bool)
                    if self._logicals_to_save
                    else np.zeros((len(imgs), max(self.num_sites, 1)),
                                  dtype=bool))
            ints = (np.array(self._intensities_to_save, dtype=np.float64)
                    if self._intensities_to_save
                    else np.zeros_like(logs, dtype=np.float64))

            def _do():
                with self._save_lock:
                    append_block(self.fname, imgs, logs, ints, sids)
                    logger.info('Saved %d frames', len(imgs))

        threading.Thread(target=_do, daemon=True).start()
        self._imgs_to_save.clear()
        self._logicals_to_save.clear()
        self._intensities_to_save.clear()
        self._seq_ids_to_save.clear()
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
            gs = np.empty(self.num_sites, dtype=[('params', 'O')])
            for s in range(self.num_sites):
                p = self.live_gauss_fits[s].get('params') if self.live_gauss_fits else None
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
            'cur_image_mid': self._display_image_mid.astype(np.float64) if self._display_image_mid is not None else None,
            'cur_intensities_mid': self._display_intensities_mid,
            'logicals_mid': self._display_logicals_mid,
            'num_images': self.num_images_per_seq,
            'is_two_array': self.is_two_array,
            'grid_locations_img2': self.grid_locations_img2.copy()
                if self.grid_locations_img2 is not None else None,
            'num_sites_img2': self.num_sites_img2,
            'thresholds_img2': self.loaded_thresholds_img2,
            'infidelities_img2': self.loaded_infidelities_img2,
            'loaded_gauss_fits_img2': self.loaded_gauss_fits_img2,
            'loaded_hist_data_img2': self.loaded_hist_data_img2,
            'loading_rates_img2': self.loading_rates_img2.copy()
                if self.loading_rates_img2 is not None else None,
            'grid_locations': self.grid_locations.copy() if len(self.grid_locations) > 0 else None,
            'box_size': self.mask_mat.shape[0],
            'num_sites': self.num_sites,
            'thresholds': self.thresholds.copy(),
            'infidelities': self.infidelities.copy(),
            'loading_rates': self.loading_rates.copy(),
            'loading_history': get_loading_history(),
            'loaded_gauss_fits': self.loaded_gauss_fits,
            'live_hist_data': self.live_hist_data,
            'live_gauss_fits': self.live_gauss_fits,
            'hist_version': self._hist_version,
            'hist_rep_sites': self._hist_rep_sites,
            'n_accum_shots': len(self._intensity_accum),
            'grid_shift_heatmap': self.grid_shift_heatmap.copy() if self.grid_shift_heatmap is not None else None,
            'grid_shift_history': list(self.grid_shift_history),
            'scan_curve': compute_scan_curve(
                self._scan_logicals, self._param_indices,
                self._scan_params, self.num_images_per_seq,
                scan_dims=self._scan_dims,
                is_two_array=self.is_two_array,
                recent_seq_ids=self._last_batch_seq_ids),
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
