"""DataManager: image acquisition, atom detection, histogram lifecycle, and saving.

State model (3 layers):
  LOADED  — from disk at startup, read-only during scan
  LIVE    — accumulates during scan, resets every 2000 shots
  EFFECTIVE — what detect_atom and the dashboard actually use
"""

import os
import logging
import threading
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


def get_data_manager(scan_id):
    with _cache_lock:
        if scan_id in _cache:
            return _cache[scan_id]
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


def _vector(val):
    """Flatten MATLAB vector to 1-D float64 array."""
    return np.asarray(val, dtype=np.float64).ravel()


class DataManager:

    def __init__(self, scan_id):
        self.scan_id = scan_id

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
        box_size = int(_scalar(self.config.get('boxSize', 11)))
        mask_sigma = float(_scalar(self.config.get('maskSigma', 2.0)))
        self.mask_mat = _gaussian_mask(box_size, mask_sigma)

        if self.is_init or self.is_hc:
            if self.is_hc:
                logger.info('High-coherence mode (isHC=1): no images via ZMQ')
            self._init_empty()
            return

        # --- LOADED state (from disk, read-only during scan) ---
        self.num_sites = 0
        self._load_from_disk()

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

        # --- Other state ---
        self.loading_rates = np.zeros(self.num_sites)
        self.grid_shift_history = []
        self.grid_shift_heatmap = None

        # --- Display state: always image-1 of the latest sequence ---
        self._display_image = None
        self._display_intensities = None
        self._display_logicals = None

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
        if self.num_sites > 0:
            try:
                create_scan_file(self.fname, self.config, self.frame_size, self.num_sites)
                self._file_created = True
            except Exception as e:
                logger.warning('Failed to create HDF5: %s', e)

    def _init_empty(self):
        """Initialize empty state for isInit scans."""
        self.num_sites = 0
        self.loaded_thresholds = np.array([])
        self.loaded_infidelities = np.array([])
        self.loaded_gauss_fits = None
        self.grid_locations = np.zeros((0, 2))
        self._intensity_accum = []
        self.live_hist_data = self.live_gauss_fits = None
        self.live_thresholds = self.live_infidelities = None
        self._hist_version = 0
        self._hist_rep_sites = []
        self.loading_rates = np.array([])
        self.grid_shift_history = []
        self.grid_shift_heatmap = None
        self._imgs_to_process = []
        self._seq_ids_to_process = []
        self._imgs_to_save = []
        self._logicals_to_save = []
        self._intensities_to_save = []
        self._seq_ids_to_save = []
        self._save_lock = threading.Lock()
        self._file_created = False
        self._day_dir = ''
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
                if g.shape[1] == 2 and g.shape[0] > 0:
                    self.grid_locations = g
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
            self._seq_ids_to_save.extend(self._seq_ids_to_process)
            self._imgs_to_process.clear()
            self._seq_ids_to_process.clear()
            return

        n_new_seqs = 0
        pSeq = self.num_images_per_seq
        seq_logic_buf = []  # collect logicals within one sequence
        for idx, img in enumerate(self._imgs_to_process):
            logicals, intensities = detect_atom(
                img.astype(np.float64), self.grid_locations,
                self.thresholds, self.mask_mat
            )
            self._logicals_to_save.append(logicals)
            self._intensities_to_save.append(intensities)
            self._imgs_to_save.append(img.astype(np.int16))
            seq_logic_buf.append(logicals)

            # On first image of each sequence: accumulate for histograms + display
            if idx % pSeq == 0:
                self._intensity_accum.append(intensities.copy())
                self.log_buffer.push(logicals.astype(np.float64))
                # Always display image-1 (loading image, not pushout)
                self._display_image = img.astype(np.int16)
                self._display_intensities = intensities.copy()
                self._display_logicals = logicals.copy()

            # On last image of each sequence: accumulate for scan curve
            if idx % pSeq == pSeq - 1:
                seq_idx = idx // pSeq
                sid = int(self._seq_ids_to_process[seq_idx]) if seq_idx < len(self._seq_ids_to_process) else 0
                logic1 = seq_logic_buf[0]
                logic2 = seq_logic_buf[1] if len(seq_logic_buf) >= 2 else None
                self._scan_logicals.append((sid, logic1.copy(), logic2.copy() if logic2 is not None else None))
                seq_logic_buf.clear()
                n_new_seqs += 1

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
        logs = np.array(self._logicals_to_save, dtype=bool) if self._logicals_to_save else np.zeros((len(imgs), max(self.num_sites, 1)), dtype=bool)
        ints = np.array(self._intensities_to_save, dtype=np.float64) if self._intensities_to_save else np.zeros_like(logs, dtype=np.float64)
        sids = np.array(self._seq_ids_to_save, dtype=np.int64)

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

    def _save_grid(self):
        try:
            np.savetxt(os.path.join(self._day_dir, 'gridLocations.txt'),
                       self.grid_locations, header='Y\tX', delimiter='\t', comments='')
        except Exception as e:
            logger.warning('Grid save failed: %s', e)

    def _save_threshold(self):
        try:
            from scipy.io import savemat
            gs = np.empty(self.num_sites, dtype=[('params', 'O')])
            for s in range(self.num_sites):
                p = self.live_gauss_fits[s].get('params') if self.live_gauss_fits else None
                gs[s]['params'] = p if p is not None else np.array([])
            savemat(os.path.join(self._day_dir, 'threshold.mat'), {
                'thresholds': self.thresholds,
                'infidelities': self.infidelities,
                'gaussFitsStruct': gs,
            })
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
            'cur_image': self._display_image.astype(np.float64) if self._display_image is not None else None,
            'cur_intensities': self._display_intensities,
            'logicals': self._display_logicals,
            'grid_locations': self.grid_locations.copy() if len(self.grid_locations) > 0 else None,
            'box_size': self.mask_mat.shape[0],
            'num_sites': self.num_sites,
            'thresholds': self.thresholds.copy(),
            'infidelities': self.infidelities.copy(),
            'loading_rates': self.loading_rates.copy(),
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
                scan_dims=self._scan_dims),
            'scan_name': self._scan_name,
            'scan_param_path': self._scan_param_path,
            'plot_scale': self._plot_scale,
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
