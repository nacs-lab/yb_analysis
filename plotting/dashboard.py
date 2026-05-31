"""Yb Tweezer Dashboard — Plotly Dash, single-callback architecture.

Layout:
  Row 1: [Tweezer Array]  [Atom Intensities]
  Row 2: [Atom Intensities (wide)]  [Loading Rate (live)]
  Row 3: [4 rep site histograms — stable between refits]
  Row 4: [Interactive site selector + histogram]
  /debug endpoint for state inspection
"""

import base64
import io
import json
import math
import os
import pickle
import time
import multiprocessing
import tempfile
import logging
import traceback
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from dash import Dash, html, dcc, Input, Output, State, no_update, Patch

from yb_analysis.config import DASH_IMAGE_MAX_DIM, DASH_IMAGE_PNG_COMPRESSION

logger = logging.getLogger(__name__)

# Theme
BG = '#0a0a16'
PANEL = '#0d1220'
TEXT = '#e0e0e0'
GRID = '#1a1a30'
_L = dict(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, size=10),
          margin=dict(l=40, r=15, t=35, b=30), uirevision='live')
_A = dict(gridcolor=GRID, zerolinecolor=GRID)

# Shared data file path
_DATA_FILE = os.path.join(tempfile.gettempdir(), 'yb_dash_data.pkl')
# Separate file for the scan-queue snapshot. Written by QueuePane at ~1Hz,
# read by both the Queue panel callback and the /api/queue endpoint, so the
# Dash subprocess doesn't need its own ZMQ socket to the runner.
_QUEUE_FILE = os.path.join(tempfile.gettempdir(), 'yb_dash_queue.pkl')
# SLM-proxy snapshot — written by yb_analysis.slm_proxy.SlmProxy on the main
# process, read here by the dashboard subprocess + the /api/slm/* routes.
_SLM_FILE = os.path.join(tempfile.gettempdir(), 'yb_dash_slm.pkl')
# Reverse-channel control file: written by the Dash subprocess (browser
# toggles) and read by the MAIN process. Carries the live-image downsample
# toggle. Tiny + atomic, like the other shared pickles.
_CONTROL_FILE = os.path.join(tempfile.gettempdir(), 'yb_dash_control.pkl')

# Opt-in performance logging. Set env YB_DASH_PROFILE=1 before launching
# run_monitor; the write path (update) and render path (refresh) then each
# emit one concise timing/size line per call, tagged DASHPROF.
_PROFILE = bool(os.environ.get('YB_DASH_PROFILE'))


class DashboardRenderer:
    """Runs the Dash web server in a **separate process** to avoid GIL
    starvation from the heavy image-processing thread.

    Data is shared via a pickle file: the main process writes it,
    the Dash process reads it on each callback tick.
    """

    def __init__(self, port=8050, host='127.0.0.1'):
        self._port = port
        self._host = host
        self._proc = None

    def start(self):
        if self._proc is None or not self._proc.is_alive():
            self._proc = multiprocessing.Process(
                target=_dash_main, args=(self._host, self._port, _DATA_FILE),
                daemon=True)
            self._proc.start()
            logger.info('Dashboard process started (pid=%d) at http://%s:%d',
                        self._proc.pid, self._host, self._port)

    def update(self, data):
        """Write plot data to the shared file.

        Pre-encodes each live frame to a downsampled PNG data URI (see
        ``_img_to_data_uri``) so the pickle stays small and the Dash callback
        (separate process) pays no image cost. Uses a double-buffer strategy
        to avoid Windows file-locking on the shared pickle.
        """
        t0 = time.perf_counter() if _PROFILE else 0.0
        d = dict(data)
        # Monotonic write counter: lets the Dash callback skip rebuilding all
        # figures on ticks where no new frame arrived (see _last_state gate).
        self._seq = getattr(self, '_seq', 0) + 1
        d['_write_seq'] = self._seq
        # Pre-encode image to a PNG data URI in the main process so the Dash
        # callback (separate process) doesn't pay the encode cost. Live-image
        # downsampling is toggleable from the dashboard (browser writes
        # _CONTROL_FILE; we read it here). Default ON — a full-sensor frame is
        # ~12 MB of base64 and freezes the browser; downsampling cuts it ~10x.
        ctrl = _read_control() or {}
        max_dim = DASH_IMAGE_MAX_DIM if ctrl.get('downsample', True) else None
        img = d.get('cur_image')
        if img is not None:
            uri, vlo, vhi = _img_to_data_uri(np.asarray(img, dtype=np.int16),
                                             max_dim=max_dim)
            d['_img_data_uri'] = uri
            d['_img_shape'] = img.shape
            d['_img_vlo'] = vlo
            d['_img_vhi'] = vhi
            d.pop('cur_image', None)  # don't pickle the raw image (18MB)
        img2 = d.get('cur_image2')
        if img2 is not None:
            uri2, vlo2, vhi2 = _img_to_data_uri(np.asarray(img2, dtype=np.int16),
                                                max_dim=max_dim)
            d['_img2_data_uri'] = uri2
            d['_img2_shape'] = img2.shape
            d['_img2_vlo'] = vlo2
            d['_img2_vhi'] = vhi2
            d.pop('cur_image2', None)
        img_mid = d.get('cur_image_mid')
        if img_mid is not None:
            uri_mid, vlo_mid, vhi_mid = _img_to_data_uri(
                np.asarray(img_mid, dtype=np.int16), max_dim=max_dim)
            d['_img_mid_data_uri'] = uri_mid
            d['_img_mid_shape'] = img_mid.shape
            d['_img_mid_vlo'] = vlo_mid
            d['_img_mid_vhi'] = vhi_mid
            d.pop('cur_image_mid', None)
        t_enc = time.perf_counter() if _PROFILE else 0.0
        # Write to alternating files to avoid read/write conflicts on Windows
        idx = getattr(self, '_write_idx', 0)
        target = _DATA_FILE + f'.{idx}'
        with open(target, 'wb') as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Update pointer (tiny file, fast write)
        with open(_DATA_FILE, 'w') as f:
            f.write(str(idx))
        self._write_idx = 1 - idx  # toggle 0 ↔ 1
        if _PROFILE:
            _log_update_profile(t0, t_enc, target, d, ctrl, max_dim)
        self.start()

    def update_queue(self, q):
        """Write the latest runner queue snapshot.

        Called by QueuePane after each successful queue_list poll (~1Hz).
        Used by both the in-browser Queue panel and /api/queue. Atomic
        rename so a partial write is never observed.
        """
        tmp = _QUEUE_FILE + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(q, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _QUEUE_FILE)
        self.start()

    def _terminate_proc(self, tag=''):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1)
            logger.info('Dashboard process stopped%s', f' ({tag})' if tag else '')
        self._proc = None

    def close(self):
        self._terminate_proc()
        for p in (_DATA_FILE, _DATA_FILE + '.0', _DATA_FILE + '.1',
                  _QUEUE_FILE, _QUEUE_FILE + '.tmp'):
            try:
                os.remove(p)
            except OSError:
                pass

    def restart(self):
        """Kill and immediately respawn the Dash subprocess.

        Keeps the shared data files in place so the new subprocess renders
        the current frame straight away (no 'waiting for data' gap).
        """
        self._terminate_proc(tag='restart')
        # Brief pause to let the OS release port 8050 before the new
        # subprocess tries to bind it. Without this the child can die
        # silently with WinError 10048.
        time.sleep(0.5)
        self.start()


def _read_data():
    """Read plot data from the shared pickle file (called in Dash process).

    Uses pointer file to find which buffer to read (avoids Windows lock conflicts).
    """
    try:
        with open(_DATA_FILE, 'r') as f:
            idx = f.read().strip()
        with open(_DATA_FILE + f'.{idx}', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, ValueError, pickle.UnpicklingError, OSError):
        return None


def _read_queue_data():
    """Read the latest scan-queue snapshot written by QueuePane."""
    try:
        with open(_QUEUE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError):
        return None


def _read_slm_data():
    """Read the latest SLM-proxy snapshot written by yb_analysis.slm_proxy.

    Returns None if the proxy was never started or the file is missing. The
    dashboard treats absence as "SLM panels disabled" and renders an offline
    badge.
    """
    try:
        with open(_SLM_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError):
        return None


def _read_control():
    """Read the dashboard control file (browser -> main reverse channel).

    Returns None when never written; callers treat absence as defaults.
    """
    try:
        with open(_CONTROL_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError):
        return None


def _write_control(state):
    """Atomically write the dashboard control file (called in Dash subprocess)."""
    tmp = _CONTROL_FILE + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _CONTROL_FILE)


# ---- Opt-in performance logging (env YB_DASH_PROFILE=1) --------------------

def _json_default(x):
    return _to_jsonable(x)


def _approx_json_size(obj):
    """Serialized byte size of a figure/patch payload (profiling only)."""
    try:
        return len(json.dumps(obj, default=_json_default))
    except (TypeError, ValueError):
        return -1


def _log_update_profile(t0, t_enc, target, d, ctrl, max_dim):
    """One line per write: image count/shapes, encode ms, pickle MB, total ms."""
    t_end = time.perf_counter()
    shapes = [d.get(k) for k in ('_img_shape', '_img2_shape', '_img_mid_shape')
              if d.get(k) is not None]
    try:
        sz = os.path.getsize(target) / 1e6
    except OSError:
        sz = -1.0
    logger.info('DASHPROF update: imgs=%d shapes=%s downsample=%s(max_dim=%s) '
                'encode=%.1fms pickle=%.2fMB total=%.1fms',
                len(shapes), shapes, ctrl.get('downsample', True), max_dim,
                (t_enc - t0) * 1e3, sz, (t_end - t0) * 1e3)


def _log_refresh_profile(n, t0, t_read, t_build, emitted):
    """One line per tick: read/build/emit ms, full/patch/no_update split,
    approximate payload MB (what the browser must parse), total ms."""
    t_emit = time.perf_counter()
    full = patch = noupd = 0
    payload = 0
    for o in emitted:
        if o is no_update:
            noupd += 1
        elif isinstance(o, Patch):
            patch += 1
            payload += max(0, _approx_json_size(o.to_plotly_json()))
        else:
            obj = o.to_plotly_json() if hasattr(o, 'to_plotly_json') else o
            full += 1
            payload += max(0, _approx_json_size(obj))
    logging.info('DASHPROF refresh n=%s: read=%.1fms build=%.1fms emit=%.1fms '
                 'kinds[full=%d patch=%d noupd=%d] payload=%.2fMB total=%.1fms',
                 n, (t_read - t0) * 1e3, (t_build - t_read) * 1e3,
                 (t_emit - t_build) * 1e3, full, patch, noupd,
                 payload / 1e6, (t_emit - t0) * 1e3)


def _to_jsonable(x):
    """Recursively convert numpy / bytes payloads to JSON-safe Python types.

    Used by the /api/* Flask routes — Flask's jsonify chokes on numpy
    arrays, np.int64, NaN/Inf, and bytes.
    """
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        f = float(x)
        return f if np.isfinite(f) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='replace')
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


# Keys served by each /api/* endpoint. Each maps to a subset of the
# dashboard's plot-data dict; everything that's downsampled-image-only
# is excluded (those live in _HEAVY_KEYS and are stripped from /snapshot).
_API_KEYS_STATUS = ('scan_id', 'scan_name', 'scan_filename', 'scan_param_path',
                    'num_sites', 'num_images', 'is_two_array', 'n_accum_shots',
                    'hist_version', '_dummy_mode')
_API_KEYS_GRID = ('grid_locations', 'thresholds', 'grid_locations_img2',
                  'thresholds_img2', 'num_sites', 'num_sites_img2', 'box_size',
                  'is_two_array')
_API_KEYS_LOADING = ('logicals', 'logicals2', 'cur_intensities',
                     'cur_intensities2', 'loading_history', 'loading_rates',
                     'loading_rates_img2', 'num_sites', 'n_accum_shots')
_API_KEYS_SCAN = ('scan_name', 'scan_filename', 'scan_param_path',
                  'scan_curve', 'plot_scale')
_API_KEYS_INFID = ('infidelities', 'infidelities_img2', 'num_sites')
_HEAVY_KEYS = ('_img_data_uri', '_img2_data_uri', '_img_mid_data_uri',
               '_img_shape', '_img2_shape', '_img_mid_shape',
               '_img_vlo', '_img_vhi', '_img2_vlo', '_img2_vhi',
               '_img_mid_vlo', '_img_mid_vhi')


def _resolve_scan_dir(scan_id):
    """Map a 14-digit scan_id to its data directory on disk.

    Mirrors `DataManager.__init__`'s path construction so the dashboard
    finds the same directory MATLAB writes into.
    """
    from yb_analysis.io.scan_directory import (
        make_scan_dir, make_scan_fname, scan_id_to_stamps)
    try:
        date_stamp, time_stamp = scan_id_to_stamps(int(scan_id))
        dname, _, _ = make_scan_dir(date_stamp, time_stamp)
        return dname
    except Exception:
        return None


def _runs_diag_response(scan_id):
    """Phase 2 helper: return per-scan diag rows.

    Strategy:
      1. If `<scan_dir>/slm_diag.h5` exists, read + return rows from it
         (fast, no network).
      2. Else passthrough to SLM PC's `/slm/runs/{scan_id}/diag` (live).
      3. Else 503.
    """
    from flask import jsonify
    scan_dir = _resolve_scan_dir(scan_id)
    if scan_dir:
        sidecar = os.path.join(scan_dir, 'slm_diag.h5')
        if os.path.exists(sidecar):
            try:
                return jsonify(_read_slm_diag_h5(sidecar, scan_id))
            except Exception as e:
                logger.warning('slm_diag.h5 read failed (%s): %s',
                               sidecar, e)
    # Live passthrough.
    try:
        from yb_analysis.slm_sync import SlmSyncClient
        client = SlmSyncClient()
        data = client.get_diag(scan_id)
        if data is None:
            return jsonify({'error': 'slm offline and no local sidecar'}), 503
        return jsonify(data)
    except Exception as e:
        logger.warning('runs_diag passthrough failed: %s', e)
        return jsonify({'error': str(e)}), 503


def _runs_code_response(scan_id):
    """Phase 2 helper: return code-snapshot manifest for a scan."""
    from flask import jsonify
    scan_dir = _resolve_scan_dir(scan_id)
    if scan_dir:
        sidecar = os.path.join(scan_dir, 'slm_code.json')
        if os.path.exists(sidecar):
            try:
                with open(sidecar, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                logger.warning('slm_code.json read failed (%s): %s',
                               sidecar, e)
    # Live passthrough.
    try:
        from yb_analysis.slm_sync import SlmSyncClient
        client = SlmSyncClient()
        data = client.get_code_manifest(scan_id)
        if data is None:
            return jsonify({'error': 'no code snapshot for this scan_id'}), 404
        return jsonify(data)
    except Exception as e:
        logger.warning('runs_code passthrough failed: %s', e)
        return jsonify({'error': str(e)}), 503


def _read_slm_diag_h5(path, scan_id):
    """Read all rows from a synced slm_diag.h5 sidecar.

    Returns the same shape the SLM PC's `/slm/runs/{id}/diag` returns,
    so the dashboard renderer treats local + remote responses identically.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        diag = f['diag']
        n = diag['seq_id'].shape[0]
        if n == 0:
            return {'scan_id': str(scan_id), 'count': 0,
                    'overflow': False, 'entries': []}
        # Pull the per-row JSON column — it carries the full original row.
        json_col = diag['diag_json'][:]
        entries = []
        for raw in json_col:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            try:
                entries.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                # Bad row — skip, but keep going.
                continue
    return {'scan_id': str(scan_id), 'count': n,
            'overflow': False, 'entries': entries}


def _register_api_routes(server):
    """Attach read-only JSON endpoints to the Flask app underlying Dash."""
    from flask import jsonify

    def _subset(keys):
        d = _read_data() or {}
        return jsonify(_to_jsonable({k: d.get(k) for k in keys}))

    _DESCRIPTIONS = {
        '/api/':            'this endpoint index',
        '/api':             'this endpoint index',
        '/api/endpoints':   'list of available endpoints (with descriptions)',
        '/api/status':      'scan_id, scan_name, scan_filename, n_accum_shots, num_sites, ...',
        '/api/grid':        'grid_locations, thresholds, box_size (+ *_img2 when two-array)',
        '/api/loading':     'logicals, cur_intensities, loading_history, loading_rates',
        '/api/scan':        'scan_curve (1-D or 2-D), scan_name, scan_filename',
        '/api/infidelities':'per-site discrimination infidelity',
        '/api/queue':       'runner queue {running, queued, history} (~1s stale)',
        '/api/snapshot':    'full plot-data dict minus heavy image blobs',
        '/api/slm':         'SLM-proxy passthrough index (health, lock, camera/png, ...)',
        '/api/slm/health':  'SLM PC /health passthrough (cached, ~2s stale)',
        '/api/slm/devices': 'SLM PC /devices passthrough (cached, ~10s stale)',
        '/api/slm/lock/status':     'SLM PC /lock/status passthrough (cached, ~2s stale)',
        '/api/slm/rearrange/diag':  'SLM PC /slm/rearrange_diag passthrough (cached, ~5s stale)',
        '/api/slm/camera/png':      'SLM PC /camera/capture_png passthrough (cached, ~2s stale)',
        '/api/slm/phase/png':       'SLM PC /slm/phase_png passthrough (cached, ~2s stale)',
        '/api/runs/<scan_id>/diag': 'per-scan SLM diag rows: synced sidecar first, then live SLM passthrough',
        '/api/runs/<scan_id>/code': 'per-scan code-snapshot manifest: synced sidecar first, then live SLM passthrough',
    }

    def _list_endpoints():
        seen = set()
        out = []
        for rule in server.url_map.iter_rules():
            path = str(rule)
            if not path.startswith('/api'):
                continue
            if path in seen:
                continue
            seen.add(path)
            out.append({'path': path,
                        'description': _DESCRIPTIONS.get(path, '')})
        out.sort(key=lambda r: r['path'])
        return out

    @server.route('/api/')
    @server.route('/api')
    def _api_index():
        return jsonify({
            'service': 'yb-tweezer-api',
            'note': 'read-only',
            'endpoints': _list_endpoints(),
        })

    @server.route('/api/endpoints')
    def _api_endpoints():
        return jsonify({'endpoints': _list_endpoints()})

    @server.route('/api/status')
    def _api_status():
        return _subset(_API_KEYS_STATUS)

    @server.route('/api/grid')
    def _api_grid():
        return _subset(_API_KEYS_GRID)

    @server.route('/api/loading')
    def _api_loading():
        return _subset(_API_KEYS_LOADING)

    @server.route('/api/scan')
    def _api_scan():
        return _subset(_API_KEYS_SCAN)

    @server.route('/api/infidelities')
    def _api_infid():
        return _subset(_API_KEYS_INFID)

    @server.route('/api/queue')
    def _api_queue():
        q = _read_queue_data()
        if q is None:
            return jsonify({'running': None, 'queued': [], 'history': []})
        return jsonify(_to_jsonable(q))

    @server.route('/api/snapshot')
    def _api_snapshot():
        d = _read_data() or {}
        return jsonify(_to_jsonable(
            {k: v for k, v in d.items() if k not in _HEAVY_KEYS}))

    # ---- SLM passthrough routes (read from yb_dash_slm.pkl) ----
    # All return cached SLM-PC data, polled by yb_analysis.slm_proxy on the
    # main process. Stale by up to the configured poll interval (typically
    # 2–10 s). When the proxy is disabled or the SLM PC is offline, returns
    # 503 with a JSON error body (no exception, no hang).

    from flask import Response

    def _slm_field(field, mime_type=None, default_status=503):
        """Helper: return field from yb_dash_slm.pkl as JSON, or 503."""
        slm = _read_slm_data()
        if slm is None:
            return jsonify({'error': 'slm proxy disabled or not started'}), 503
        if slm.get('slm_offline'):
            return jsonify({
                'error': 'slm offline',
                'last_error_msg': slm.get('last_error_msg', {}),
                'last_error_ts': slm.get('last_error_ts', {}),
            }), 503
        val = slm.get(field)
        if val is None:
            return jsonify({'error': f'no data for {field} yet'}), 503
        if mime_type:
            # PNG byte payload — return as raw response, not JSON.
            return Response(val, mimetype=mime_type)
        return jsonify(_to_jsonable(val))

    @server.route('/api/slm')
    @server.route('/api/slm/')
    def _api_slm_index():
        slm = _read_slm_data() or {}
        return jsonify({
            'service': 'yb-slm-proxy',
            'slm_url': slm.get('slm_url'),
            'slm_offline': slm.get('slm_offline', True),
            'last_poll_ts': slm.get('last_poll_ts', {}),
            'endpoints': [p for p in _DESCRIPTIONS if p.startswith('/api/slm/')],
        })

    @server.route('/api/slm/health')
    def _api_slm_health():
        return _slm_field('health')

    @server.route('/api/slm/devices')
    def _api_slm_devices():
        return _slm_field('devices')

    @server.route('/api/slm/lock/status')
    def _api_slm_lock():
        return _slm_field('lock_status')

    @server.route('/api/slm/rearrange/diag')
    def _api_slm_rearrange_diag():
        return _slm_field('rearrange_diag')

    @server.route('/api/slm/camera/png')
    def _api_slm_camera_png():
        return _slm_field('camera_png', mime_type='image/png')

    @server.route('/api/slm/phase/png')
    def _api_slm_phase_png():
        return _slm_field('phase_png', mime_type='image/png')

    # ---- Phase 2: per-scan diag / code retrieval (synced sidecar first,
    #      then SLM passthrough). These are used by the dashboard's
    #      Analysis tab and by ad-hoc curl/notebook clients.

    @server.route('/api/runs/<scan_id>/diag')
    def _api_runs_diag(scan_id):
        """Return the per-scan SLM diag rows.

        Prefers the locally-synced HDF5 sidecar (if Phase 2 sync has
        already run for this scan_id and `slm_diag.h5` exists).
        Otherwise passes through to the SLM PC's
        `/slm/runs/{scan_id}/diag` endpoint (~live data, ~5s stale).
        """
        return _runs_diag_response(scan_id)

    @server.route('/api/runs/<scan_id>/code')
    def _api_runs_code(scan_id):
        """Return the code-snapshot manifest for `scan_id`.

        Prefers `<scan_dir>/slm_code.json` if Phase 2 sync persisted
        it; otherwise passes through to the SLM PC's
        `/slm/runs/{scan_id}/code`.
        """
        return _runs_code_response(scan_id)


def _dash_main(host, port, data_file):
    """Entry point for the Dash subprocess."""
    # Reconfigure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [dash] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    app = _build_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ---- Partial-update (Patch) machinery --------------------------------------
# The single refresh callback used to return ~16 brand-new figures every tick,
# forcing the browser to run Plotly.react() on all of them at once — a
# multi-hundred-ms main-thread freeze that blocked hover. Instead we now build
# the full figure (reusing every _fig_* builder unchanged) and emit a Dash
# Patch carrying ONLY the leaf values that changed since the previous tick, so
# Plotly mutates traces/shapes in place. Panels whose data didn't change emit
# no_update and cost the browser nothing.
#
# Correctness model: `_last_figs` caches, per panel, the exact figure the
# connected browser is currently showing. A Patch is a diff applied on the
# client against that base, so the cache MUST track every emit:
#   * full figure -> cache = new      (client now shows new)
#   * patch        -> cache = new      (client applied the diff -> shows new)
#   * no_update    -> cache unchanged  (client still shows old)
# The initial callback (n_intervals == 0, fired once per page load) and a
# periodic keyframe always send the full figure, so a freshly (re)loaded tab
# resyncs. This assumes ONE active dashboard tab (the lab's single screen);
# multiple tabs loading at the same instant could briefly desync until the
# next keyframe — acceptable here, and a reload always fixes it.

_PANEL_IDS = ('array', 'array_mid', 'array2', 'intens', 'loadlive',
              'load', 'infid', 'shift', 'scan', 'avghist',
              'rep0', 'rep1', 'rep2', 'rep3')

# Send a full (un-patched) figure every this-many ticks as a resync keyframe.
# At the 3 s tick that's ~one full re-render every 10 min — imperceptible.
_KEYFRAME_EVERY = 200

# Per-panel snapshot of the figure the browser currently shows. Lives only in
# the Dash subprocess (single process). Keyed by _PANEL_IDS + 'site'.
_last_figs = {}

_MISSING = object()


class _Structural(Exception):
    """Raised mid-diff when a change can't be safely expressed as an in-place
    patch (a trace ``type`` flip, or a dropped key); the caller then falls back
    to sending the whole figure, which resyncs the client."""


def _vals_equal(a, b):
    """Leaf equality tolerant of numpy arrays/scalars, nested lists, and NaN."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            aa, bb = np.asarray(a), np.asarray(b)
            if aa.shape != bb.shape:
                return False
            if aa.dtype.kind == 'f' or bb.dtype.kind == 'f':
                return bool(np.array_equal(aa, bb, equal_nan=True))
            return bool(np.array_equal(aa, bb))
        except (TypeError, ValueError):
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_vals_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (np.integer, np.floating)):
        a = a.item()
    if isinstance(b, (np.integer, np.floating)):
        b = b.item()
    if (isinstance(a, float) and isinstance(b, float)
            and math.isnan(a) and math.isnan(b)):
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _is_dict_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict)


def _diff_into(node, old, new):
    """Record old->new changes into the Patch proxy ``node``.

    Returns True if anything changed. Raises ``_Structural`` when the change
    needs a full-figure replacement instead of an in-place patch.
    """
    if not (isinstance(old, dict) and isinstance(new, dict)):
        raise _Structural()           # only ever called on dict nodes
    if set(old) - set(new):           # a key disappeared -> can't patch safely
        raise _Structural()
    changed = False
    for k, nv in new.items():
        ov = old.get(k, _MISSING)
        if ov is _MISSING:            # brand-new key -> assign it
            node[k] = _to_jsonable(nv)
            changed = True
            continue
        if k == 'type' and ov != nv:  # trace type flip (scatter<->heatmap)
            raise _Structural()
        if isinstance(ov, dict) and isinstance(nv, dict):
            changed = _diff_into(node[k], ov, nv) or changed
        elif _is_dict_list(ov) or _is_dict_list(nv):
            # list of dicts: traces / shapes / annotations / images
            if (not isinstance(ov, list) or not isinstance(nv, list)
                    or len(ov) != len(nv)):
                node[k] = _to_jsonable(nv)         # length change -> replace
                changed = True
            else:
                for i in range(len(nv)):
                    if isinstance(ov[i], dict) and isinstance(nv[i], dict):
                        changed = _diff_into(node[k][i], ov[i], nv[i]) or changed
                    elif not _vals_equal(ov[i], nv[i]):
                        node[k][i] = _to_jsonable(nv[i])
                        changed = True
        else:
            if not _vals_equal(ov, nv):
                node[k] = _to_jsonable(nv)
                changed = True
    return changed


def _emit(pid, fig, force_full):
    """Return ``fig`` as a full object, a minimal Patch, or no_update.

    Keeps ``_last_figs[pid]`` equal to whatever the browser now shows.
    """
    if fig is _SKIP:                 # gated panel whose inputs didn't change
        return no_update
    new = fig.to_plotly_json() if hasattr(fig, 'to_plotly_json') else fig
    old = _last_figs.get(pid)
    if force_full or old is None:
        _last_figs[pid] = new
        return fig
    try:
        patch = Patch()
        changed = _diff_into(patch, old, new)
    except _Structural:
        _last_figs[pid] = new
        return fig                    # whole figure resyncs the client
    except Exception:
        logging.exception('patch diff failed for panel %s; sending full figure',
                          pid)
        _last_figs[pid] = new
        return fig
    _last_figs[pid] = new
    return patch if changed else no_update


def _force_full(n):
    """First call after a page load (n==0) and periodic keyframes go full."""
    return n == 0 or (_KEYFRAME_EVERY and n % _KEYFRAME_EVERY == 0)


# ---- Build gating ----------------------------------------------------------
# At large site counts (thousands of tweezers) the dominant cost is *building*
# the figures every tick — Plotly validates thousands-element arrays per trace
# (profiling showed ~0.7-0.9 s/tick even when nothing changed). Two gates avoid
# that work:
#   * Global: the main process stamps each write with a monotonic _write_seq.
#     If the seq (and slider/colorbar inputs) is unchanged since last tick, no
#     panel can have changed -> skip building entirely and return no_update.
#   * Per-panel: the cumulative 2-D maps (loading / infidelity / grid-shift)
#     only refit every 50-200 shots, so even on a fresh frame their data is
#     usually identical -> skip rebuilding just those.
# Page-load and keyframe ticks (force_full) bypass both gates so a client
# always resyncs.

_SKIP = object()                 # sentinel: gated panel unchanged -> no_update
_last_sig = {}                   # per-panel input signature (gated panels)
_last_state = {'key': None}      # global (write_seq, marker_size, cbar_scale)

# Above this site count, render the per-site overlays with WebGL (scattergl)
# instead of SVG. SVG creates one DOM node per site — fine for a few hundred,
# catastrophic at thousands; WebGL draws them all in one cheap pass.
_GL_SITES = 400


def _h(x):
    """Cheap, collision-safe-enough hash of an array/list/scalar for change
    detection. Hashes the raw bytes of numpy arrays (microseconds even at a
    few thousand elements)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return (x.shape, x.dtype.str, hash(x.tobytes()))
    if isinstance(x, (list, tuple)):
        return tuple(_h(v) for v in x)
    return x


def _sig_load(d, marker_size):
    return ('load', marker_size, bool(d.get('_dummy_mode')),
            _h(d.get('loading_rates')), _h(d.get('grid_locations')))


def _sig_infid(d, marker_size):
    return ('infid', marker_size, bool(d.get('_dummy_mode')),
            _h(d.get('infidelities')), _h(d.get('grid_locations')))


def _sig_shift(d):
    return ('shift', bool(d.get('_dummy_mode')),
            _h(d.get('grid_shift_heatmap')),
            tuple(map(tuple, d.get('grid_shift_history') or [])))


def _gated(pid, sig, builder, force_full):
    """Build the panel only if its signature changed (or force_full); else
    return _SKIP so _emit yields no_update without building."""
    if not force_full and _last_sig.get(pid) == sig:
        return _SKIP
    _last_sig[pid] = sig
    return builder()


def _build_app():
    app = Dash(__name__, title='Yb Tweezer Dashboard')

    # ---- Read-only JSON endpoints (piggyback on Dash's Flask server) ----
    # Lets external clients (e.g. the SLM server) poll experiment state over
    # the LAN. All GET, no writes. Bound to the same port as the dashboard.
    _register_api_routes(app.server)

    # Force crisp pixel rendering on zoomed images. Plotly recreates SVG
    # elements on each update, so CSS alone doesn't stick. A MutationObserver
    # re-applies the style whenever a new <image> element appears.
    app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
/* Sleek iOS-style toggle switch (used for the colorbar autoscale control). */
input.yb-switch {
    -webkit-appearance: none; appearance: none; margin: 0 6px 0 0;
    position: relative; width: 30px; height: 16px; flex: none;
    background: #3a3a52; border-radius: 8px; cursor: pointer;
    transition: background .15s ease; outline: none;
}
input.yb-switch::before {
    content: ''; position: absolute; top: 2px; left: 2px;
    width: 12px; height: 12px; border-radius: 50%;
    background: #d8d8e0; transition: transform .15s ease;
}
input.yb-switch:checked { background: #2a7fff; }
input.yb-switch:checked::before { transform: translateX(14px); background: #fff; }
</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
new MutationObserver(function(mutations) {
    document.querySelectorAll('image').forEach(function(el) {
        if (el.style.imageRendering !== 'pixelated') {
            el.style.imageRendering = 'pixelated';
        }
    });
    /* Bind plotly_click on load/infid graphs → update site dropdown.
       Re-binds after each DOM mutation so it survives figure refreshes. */
    ['load', 'infid'].forEach(function(gid) {
        var el = document.getElementById(gid);
        if (el && el.classList.contains('js-plotly-plot') && !el._ybClick) {
            el._ybClick = true;
            el.on('plotly_click', function(evtData) {
                if (!evtData || !evtData.points || !evtData.points.length) return;
                var pt = evtData.points[0];
                /* customdata is [site, value] on the 2-D maps (scalar on older
                   figures); take element 0 when it's an array. */
                var cd = pt.customdata;
                var site = Array.isArray(cd) ? cd[0]
                         : (cd != null) ? cd
                         : (pt.pointIndex != null) ? pt.pointIndex + 1 : null;
                if (site == null) return;
                /* Dash stores component props on the React fiber. We can update
                   the dropdown by finding its Dash component and calling setProps. */
                var dd = document.getElementById('site-dd');
                if (!dd) return;
                var key = Object.keys(dd).find(function(k) {
                    return k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$');
                });
                if (key) {
                    var fiber = dd[key];
                    /* Walk up to find the Dash component with setProps */
                    var node = fiber;
                    for (var i = 0; i < 30 && node; i++) {
                        if (node.memoizedProps && typeof node.memoizedProps.setProps === 'function') {
                            node.memoizedProps.setProps({value: site});
                            break;
                        }
                        node = node.return;
                    }
                }
            });
        }
    });
}).observe(document.body, {childList: true, subtree: true});
</script></body></html>'''

    app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh',
        'fontFamily': '"Segoe UI", sans-serif', 'color': TEXT, 'padding': '10px'}, children=[
        html.H1('Yb Tweezer Dashboard', style={'textAlign': 'center', 'color': '#e94560',
            'margin': '5px 0 10px 0', 'fontSize': '24px'}),
        # Row 1: image1 | middle | image2 | scan curve — up to four
        # equal-width 670px panels (per-shot live data: gets the most
        # vertical real estate). The middle-frame panel is wrapped in a
        # Div whose `display` is toggled by the "Middle frame" switch
        # overlaid on the array2 panel; when hidden, the remaining three
        # panels expand to fill the row.
        _row([
            # Image-1 panel + live-image downsample toggle overlaid top-right.
            # When ON (default) the main process ships a downsampled frame so
            # the browser doesn't choke on full-sensor images; turn OFF to see
            # full resolution (heavier — only for momentary inspection).
            html.Div(style={'flex': '1', 'minWidth': '0', 'position': 'relative'}, children=[
                dcc.Graph(id='array', figure=_waiting(''), style={'height': '670px'},
                          config={'displayModeBar': False}),
                html.Div(style={'position': 'absolute', 'top': '9px', 'right': '70px',
                                'zIndex': '5', 'display': 'flex', 'alignItems': 'center'},
                    children=[
                        dcc.Checklist(id='downsample',
                            options=[{'label': 'Downsample', 'value': 'ds'}],
                            value=['ds'], inline=True, inputClassName='yb-switch',
                            style={'fontSize': '11px', 'color': '#ffffff'},
                            labelStyle={'display': 'flex', 'alignItems': 'center',
                                        'cursor': 'pointer', 'margin': '0', 'color': '#ffffff'}),
                    ]),
            ]),
            html.Div(id='array-mid-wrapper',
                     style={'flex': '1', 'minWidth': '0', 'display': 'flex'},
                     children=[_graph('array_mid', 670)]),
            # array2 panel + middle-frame visibility toggle overlaid top-right.
            html.Div(style={'flex': '1', 'minWidth': '0', 'position': 'relative'}, children=[
                dcc.Graph(id='array2', figure=_waiting(''),
                          style={'height': '670px'},
                          config={'displayModeBar': False}),
                html.Div(style={'position': 'absolute', 'top': '9px', 'right': '70px',
                                'zIndex': '5', 'display': 'flex', 'alignItems': 'center'},
                    children=[
                        dcc.Checklist(id='show-mid',
                            options=[{'label': 'Middle frame', 'value': 'mid'}],
                            value=['mid'], inline=True, inputClassName='yb-switch',
                            style={'fontSize': '11px', 'color': '#ffffff'},
                            labelStyle={'display': 'flex', 'alignItems': 'center',
                                        'cursor': 'pointer', 'margin': '0', 'color': '#ffffff'}),
                    ]),
            ]),
            # Scan panel; colorbar-scale toggle overlaid INSIDE the panel (top-right).
            html.Div(style={'flex': '1', 'minWidth': '0', 'position': 'relative'}, children=[
                dcc.Graph(id='scan', figure=_waiting(''), style={'height': '670px'},
                          config={'displayModeBar': False}),
                # Toggle switch: on → autoscale colorbar to data; off → fixed 0–1.
                html.Div(style={'position': 'absolute', 'top': '9px', 'right': '70px',
                                'zIndex': '5', 'display': 'flex', 'alignItems': 'center'},
                    children=[
                        dcc.Checklist(id='cbar-scale',
                            options=[{'label': 'Autoscale', 'value': 'auto'}],
                            value=[], inline=True, inputClassName='yb-switch',
                            style={'fontSize': '11px', 'color': '#ffffff'},
                            labelStyle={'display': 'flex', 'alignItems': 'center',
                                        'cursor': 'pointer', 'margin': '0', 'color': '#ffffff'}),
                    ]),
            ]),
        ]),
        # Row 2: Atom Intensities (wide) + live Loading Rate panel
        # (per-shot data; intens gets 3x the width since it scales with #sites)
        _row([_graph('intens', 320, flex=3), _graph('loadlive', 320, flex=1)]),
        # Row 3: Avg Histogram + Rep site histograms (refit every 200 shots)
        _row([_graph('avghist', 240)] + [_graph(f'rep{i}', 240) for i in range(4)]),
        # Row 4: Loading | Infidelities | Site selector + Site Hist | Grid Shift
        # Tall enough for the 2-D site maps to read as roughly square on large arrays.
        _row([
            _graph('load', 600),
            _graph('infid', 600),
            html.Div(style={'flex': '1', 'minWidth': '0', 'display': 'flex', 'gap': '8px'}, children=[
                # Left: dropdown + parameters
                html.Div(style={'width': '140px', 'flexShrink': '0'}, children=[
                    html.Label('Site:', style={'fontSize': '12px'}),
                    dcc.Dropdown(id='site-dd', options=[], value=1, clearable=False,
                                 style={'backgroundColor': '#2b2b4a', 'color': '#222', 'marginBottom': '8px'}),
                    html.Div(id='site-info', style={'fontSize': '11px', 'color': '#bbb',
                        'lineHeight': '1.6'}),
                    # Slider controls marker size for the site-resolved
                    # scatter plots (load / infid) so they read well at
                    # any array spacing.
                    html.Div(style={'marginTop': '14px', 'paddingTop': '10px',
                        'borderTop': '1px solid #333'}, children=[
                        html.Label('Marker size:', style={'fontSize': '11px', 'color': '#bbb'}),
                        dcc.Slider(id='marker-size', min=2, max=40, step=1, value=12,
                            marks={2: {'label': '2', 'style': {'fontSize': '9px', 'color': '#888'}},
                                   20: {'label': '20', 'style': {'fontSize': '9px', 'color': '#888'}},
                                   40: {'label': '40', 'style': {'fontSize': '9px', 'color': '#888'}}},
                            tooltip={'placement': 'top', 'always_visible': False}),
                    ]),
                ]),
                # Right: histogram
                _graph('site', 590),
            ]),
            _graph('shift', 600),
        ]),
        # Row 5: SLM hardware (filled by the slm callback below). Phase 0 lands
        # these as a row; Phase 4 moves them into a dedicated tab.
        html.Div(id='slm-panel', style={
            'backgroundColor': PANEL, 'padding': '10px 14px',
            'marginTop': '10px', 'borderRadius': '4px',
            'fontFamily': '"Segoe UI", sans-serif', 'fontSize': '12px',
            'color': TEXT}),
        # Row 6: Scan queue (filled by the queue callback below)
        html.Div(id='queue-panel', style={
            'backgroundColor': PANEL, 'padding': '10px 14px',
            'marginTop': '10px', 'borderRadius': '4px',
            'fontFamily': '"Segoe UI", sans-serif', 'fontSize': '12px',
            'color': TEXT}),
        # Debug
        html.Details([
            html.Summary('Debug Info', style={'cursor': 'pointer', 'color': '#888', 'fontSize': '11px'}),
            html.Pre(id='debug-pre', style={'fontSize': '10px', 'color': '#aaa',
                'maxHeight': '300px', 'overflow': 'auto', 'whiteSpace': 'pre-wrap'}),
        ], style={'marginTop': '10px'}),
        dcc.Interval(id='tick', interval=3000, n_intervals=0),
        # Holds the downsample-toggle state; the callback's real job is the
        # side effect of writing _CONTROL_FILE for the main process to read.
        dcc.Store(id='downsample-state'),
    ])

    # --- Single callback for all panels ---
    outputs = ([Output('array', 'figure'), Output('array_mid', 'figure'),
                 Output('array2', 'figure'),
                 Output('intens', 'figure'), Output('loadlive', 'figure'),
                 Output('load', 'figure'), Output('infid', 'figure'),
                 Output('shift', 'figure'), Output('scan', 'figure'),
                 Output('avghist', 'figure')]
               + [Output(f'rep{i}', 'figure') for i in range(4)]
               + [Output('site-dd', 'options'), Output('debug-pre', 'children')])

    @app.callback(outputs, [Input('tick', 'n_intervals'),
                            Input('marker-size', 'value'),
                            Input('cbar-scale', 'value')])
    def refresh(_n, marker_size, cbar_toggle):
        # Guard against the slider returning None during first render.
        marker_size = int(marker_size) if marker_size else 12
        # Checklist returns a list; 'auto' present → autoscale, else fixed 0–1.
        cbar_scale = 'auto' if (cbar_toggle and 'auto' in cbar_toggle) else '01'
        t0 = time.perf_counter() if _PROFILE else 0.0
        d = _read_data()
        t_read = time.perf_counter() if _PROFILE else 0.0
        debug_lines = []

        if d is None:
            # No data: send full placeholders and drop the cache so the next
            # real tick re-establishes a full figure for every panel.
            for pid in _PANEL_IDS:
                _last_figs.pop(pid, None)
            debug_lines.append('No data yet (pickle file not found)')
            empty = [_waiting(t) for t in [
                'Tweezer Array (img 1)', 'Tweezer Array (middle)',
                'Tweezer Array (img 2)', 'Intensities', 'Loading Rate',
                'Loading', 'Infidelities', 'Grid Shift', 'Scan Curve',
                'Avg Histogram']]
            _last_sig.clear()
            _last_state['key'] = None
            return empty + [_waiting('Site Hist')]*4 + [[], '\n'.join(debug_lines)]

        # Global gate: if no new frame was written since the last tick (and the
        # slider/colorbar inputs are unchanged), nothing can have changed — skip
        # the whole rebuild. force_full (page load / keyframe) bypasses this.
        full = _force_full(_n)
        seq = d.get('_write_seq')
        state_key = (seq, marker_size, cbar_scale)
        if not full and seq is not None and _last_state.get('key') == state_key:
            if _PROFILE:
                logging.info('DASHPROF refresh n=%s: no new frame -> all '
                             'no_update (read=%.1fms)', _n,
                             (t_read - t0) * 1e3)
            return [no_update] * 16
        _last_state['key'] = state_key

        try:
            has_img = d.get('_img_data_uri') is not None
            has_img_mid = d.get('_img_mid_data_uri') is not None
            has_img2 = d.get('_img2_data_uri') is not None
            n = d.get('num_sites', 0)
            v = d.get('hist_version', 0)
            n_acc = d.get('n_accum_shots', 0)

            num_images = int(d.get('num_images', 1) or 1)
            img2_no_data_msg = ('No image 2 (NumImages = 1)' if num_images < 2
                                else 'Waiting for data...')
            img_mid_no_data_msg = ('No middle frame (NumImages < 3)'
                                   if num_images < 3 else 'Waiting for data...')
            # Two-array grid only applies to the final frame; the middle
            # frame is still in the initial array layout when pSeq >= 3.
            img2_grid_key = ('grid_locations_img2' if d.get('is_two_array')
                             else 'grid_locations')
            # Dummy mode: live panels (image, intensities, loading-rate trace)
            # still reflect the current frame, but cumulative panels carry
            # over stale values from the last real scan. Blank those out and
            # label them so the user isn't misled by frozen data.
            is_dummy = bool(d.get('_dummy_mode'))
            dummy_msg = 'Dummy mode'

            def _stale(title, builder):
                return _waiting(title, dummy_msg) if is_dummy else builder()

            figs = [
                _fig_array(d) if has_img else _waiting('Tweezer Array (img 1)'),
                _fig_array(d, img_key='_img_mid_data_uri',
                           shape_key='_img_mid_shape',
                           vlo_key='_img_mid_vlo', vhi_key='_img_mid_vhi',
                           logicals_key='logicals_mid',
                           grid_key='grid_locations',
                           title='Tweezer Array (middle)')
                    if has_img_mid else _waiting('Tweezer Array (middle)',
                                                 img_mid_no_data_msg),
                _fig_array(d, img_key='_img2_data_uri', shape_key='_img2_shape',
                           vlo_key='_img2_vlo', vhi_key='_img2_vhi',
                           logicals_key='logicals2', grid_key=img2_grid_key,
                           title='Tweezer Array (img 2)')
                    if has_img2 else _waiting('Tweezer Array (img 2)',
                                              img2_no_data_msg),
                _fig_intens(d),
                _fig_loading_live(d),
                # Cumulative 2-D maps: gated — rebuilt only when their own data
                # changed (they refit every 50-200 shots, not every frame).
                _gated('load', _sig_load(d, marker_size),
                       lambda: _stale('Loading Rates',
                                      lambda: _fig_loading(d, marker_size=marker_size)),
                       full),
                _gated('infid', _sig_infid(d, marker_size),
                       lambda: _stale('Infidelities',
                                      lambda: _fig_infid(d, marker_size=marker_size)),
                       full),
                _gated('shift', _sig_shift(d),
                       lambda: _stale('Grid Shift', lambda: _fig_shift(d)),
                       full),
                _stale('Scan Curve', lambda: _fig_scan_curve(d, cbar_scale=cbar_scale)),
                _stale('Avg Histogram', lambda: _fig_avghist(d)),
            ]

            reps = ([_waiting('Site Hist', dummy_msg)] * 4
                    if is_dummy else _figs_reps(d))
            opts = [{'label': f'Site {i+1}', 'value': i+1} for i in range(n)]

            lh = d.get('live_hist_data')
            lf = d.get('live_gauss_fits')
            ldf = d.get('loaded_gauss_fits')
            debug_lines.append(f'sites={n} accum={n_acc} hist_v={v}')
            debug_lines.append(f'live_hist: {"list["+str(len(lh))+"]" if isinstance(lh, list) else type(lh).__name__}')
            debug_lines.append(f'live_fits: {"list["+str(len(lf))+"]" if isinstance(lf, list) else type(lf).__name__}')
            debug_lines.append(f'loaded_fits: {"list["+str(len(ldf))+"]" if isinstance(ldf, list) else type(ldf).__name__}')
            debug_lines.append(f'img={has_img} img_mid={has_img_mid} img2={has_img2} rep_sites={d.get("hist_rep_sites")}')
            # scan_curve diagnostic: tells whether compute_scan_curve has
            # produced a dict at all and what mode/dims it picked.
            sc = d.get('scan_curve')
            if sc is None:
                debug_lines.append('scan_curve: None  (compute_scan_curve returned None — empty scan_logicals or missing scan_params/param_indices)')
            else:
                sc_mode = sc.get('mode', '?')
                sc_ndim = sc.get('ndim', 1)
                n_reps_arr = sc.get('n_reps')
                n_reps_total = int(np.asarray(n_reps_arr).sum()) if n_reps_arr is not None else 0
                debug_lines.append(f'scan_curve: mode={sc_mode} ndim={sc_ndim} reps_total={n_reps_total} num_images={d.get("num_images")}')

            # Emit minimal Patches (or no_update) instead of replacing every
            # figure wholesale — this is what keeps hover responsive. Building
            # the full figures above is cheap (separate Dash process); the
            # browser only pays for the panels that actually changed.
            t_build = time.perf_counter() if _PROFILE else 0.0
            emitted = [_emit(pid, f, full)
                       for pid, f in zip(_PANEL_IDS, figs + reps)]
            if _PROFILE:
                _log_refresh_profile(_n, t0, t_read, t_build, emitted)
            return emitted + [opts, '\n'.join(debug_lines)]

        except Exception:
            tb = traceback.format_exc()
            logging.error('Dashboard render error:\n%s', tb)
            return [no_update] * 16

    @app.callback([Output('site', 'figure'), Output('site-info', 'children')],
                  [Input('site-dd', 'value'), Input('tick', 'n_intervals')])
    def site_hist(val, _n):
        d = _read_data()
        if d is None or val is None:
            _last_figs.pop('site', None)
            return _waiting('Site Histogram'), ''
        if d.get('_dummy_mode'):
            _last_figs.pop('site', None)
            return _waiting('Site Histogram', 'Dummy mode'), ''
        fig, info = _fig_site(d, int(val) - 1)
        # Patch the figure in place (the site-info text is tiny — sent whole).
        return _emit('site', fig, _force_full(_n)), info

    # Click on loading-rate or infidelity 2D plot → select site in dropdown
    # is handled entirely in JavaScript (index_string) via plotly_click +
    # React fiber setProps. This bypasses Dash's callback system which can
    # lose clickData when the refresh callback replaces figures every 3s.

    @app.callback(Output('downsample-state', 'data'),
                  Input('downsample', 'value'))
    def set_downsample(val):
        # Browser -> main-process reverse channel. The main process reads
        # _CONTROL_FILE in DashboardRenderer.update() to decide max_dim.
        # Fires on page load too, so the control file always reflects the UI.
        ds = bool(val and 'ds' in val)
        try:
            _write_control({'downsample': ds})
        except OSError:
            logging.exception('failed to write dashboard control file')
        return ds

    @app.callback(Output('array-mid-wrapper', 'style'),
                  Input('show-mid', 'value'))
    def toggle_array_mid(val):
        # Hide the whole wrapper (and the array_mid panel inside it) when
        # the toggle is off; the remaining three row-1 panels expand to
        # fill the freed flex space.
        show = bool(val and 'mid' in val)
        return {'flex': '1', 'minWidth': '0',
                'display': 'flex' if show else 'none'}

    @app.callback(Output('queue-panel', 'children'),
                  Input('tick', 'n_intervals'))
    def refresh_queue(_n):
        return _render_queue_panel(_read_queue_data())

    @app.callback(Output('slm-panel', 'children'),
                  Input('tick', 'n_intervals'))
    def refresh_slm(_n):
        return _render_slm_panel(_read_slm_data())

    return app


# ---- Helpers ----

def _row(children):
    return html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}, children=children)

def _graph(id, h, flex=1):
    # Set initial "waiting" figure so Plotly has a uirevision baseline.
    # Without this, Plotly may not re-render when the callback first returns
    # a figure with uirevision='live' (no prior value to compare against).
    return dcc.Graph(id=id, figure=_waiting(''),
                     style={'flex': f'{flex}', 'minWidth': '0', 'height': f'{h}px'},
                     config={'displayModeBar': False})

def _waiting(title, message='Waiting for data...'):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=14, color='#666'))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, size=10),
                      margin=dict(l=40, r=15, t=35, b=30), uirevision='waiting',
                      title=title)
    return fig


# ---- Figure builders ----

def _img_to_data_uri(img, max_dim=DASH_IMAGE_MAX_DIM,
                     compression=DASH_IMAGE_PNG_COMPRESSION):
    """Convert an int16 image to a PNG data URI for the Plotly background.

    Downsamples to ``max_dim`` px on the long edge before encoding: the array
    panel is only ~670 px wide, so shipping a full-sensor frame is wasteful and
    freezes the browser on decode (~12 MB base64 for a 4096x2304 frame). The
    caller keeps the ORIGINAL shape for the figure extent, so the smaller
    raster is stretched back and the site-box overlays (in original pixel
    coords) stay aligned. ``vlo``/``vhi`` are the full-image 2/98 percentiles,
    returned for the colorbar and computed before downsampling.
    """
    import cv2
    vlo, vhi = float(np.percentile(img, 2)), float(np.percentile(img, 98))
    gray = np.clip((img.astype(np.float32) - vlo) / max(vhi - vlo, 1) * 255,
                   0, 255).astype(np.uint8)
    if max_dim:
        h, w = gray.shape
        scale = max(h, w) / float(max_dim)
        if scale > 1.0:
            gray = cv2.resize(
                gray, (max(1, round(w / scale)), max(1, round(h / scale))),
                interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode('.png', gray, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    b64 = base64.b64encode(enc.tobytes()).decode()
    return f'data:image/png;base64,{b64}', vlo, vhi


def _fig_array(d, img_key='_img_data_uri', shape_key='_img_shape',
               vlo_key='_img_vlo', vhi_key='_img_vhi',
               logicals_key='logicals', grid_key='grid_locations',
               title='Tweezer Array (img 1)'):
    data_uri = d.get(img_key)
    shape = d.get(shape_key)
    if data_uri is None or shape is None:
        return _waiting(title)
    H, W = shape
    fig = go.Figure()
    fig.add_layout_image(
        source=data_uri, xref='x', yref='y',
        x=0, y=0, sizex=W, sizey=H,
        sizing='stretch', layer='below',
    )
    # Colorbar via invisible scatter + autorange anchor at image corners
    vlo = d.get(vlo_key, 0)
    vhi = d.get(vhi_key, 255)
    fig.add_trace(go.Scatter(
        x=[0, W, 0, W], y=[0, 0, H, H], mode='markers',
        marker=dict(size=0.1, opacity=0, color=[vlo, vhi, vlo, vhi],
                    colorscale='gray', cmin=vlo, cmax=vhi, showscale=True,
                    colorbar=dict(title='Counts', len=0.9)),
        hoverinfo='skip', showlegend=False))

    # Site occupancy overlay (green = loaded, red = empty).
    grid = d.get(grid_key)
    logicals = d.get(logicals_key)
    box = d.get('box_size', 11)
    n = len(grid) if grid is not None else 0
    if grid is not None and n > 0:
        if logicals is not None and len(logicals) >= n:
            occ = np.asarray(logicals[:n], dtype=float)
        else:
            occ = np.zeros(n)
        if n > _GL_SITES:
            # WebGL boxes drawn as DATA-coordinate line outlines (not markers):
            # markers are a fixed pixel size that smears into a blob when zoomed
            # out, whereas these rectangles scale with zoom exactly like the old
            # SVG shapes — but render in two cheap WebGL traces (loaded / empty)
            # instead of thousands of SVG nodes. Each site contributes a closed
            # 5-corner loop plus a NaN to break the line between boxes.
            half = box / 2.0
            ys = grid[:, 0].astype(float)
            xs = grid[:, 1].astype(float)
            ox = np.array([-half, half, half, -half, -half, np.nan])
            oy = np.array([-half, -half, half, half, -half, np.nan])
            bx = xs[:, None] + ox[None, :]          # (n, 6)
            by = ys[:, None] + oy[None, :]
            loaded = occ.astype(bool)
            for sel, color in ((loaded, '#00ff88'), (~loaded, '#ff4444')):
                fig.add_trace(go.Scattergl(
                    x=bx[sel].ravel(), y=by[sel].ravel(), mode='lines',
                    line=dict(color=color, width=1.5),
                    hoverinfo='skip', showlegend=False))
        else:
            # Small arrays: data-coord rectangles (scale with zoom, crisper).
            half = box / 2
            shapes = []
            for i in range(n):
                y0, x0 = grid[i]
                c = '#00ff88' if occ[i] else '#ff4444'
                shapes.append(dict(type='rect', x0=x0-half, y0=y0-half,
                                   x1=x0+half, y1=y0+half,
                                   line=dict(color=c, width=2)))
            fig.update_layout(shapes=shapes)
        if n <= 200:
            # Text labels only for small arrays
            fig.add_trace(go.Scatter(
                x=grid[:, 1], y=grid[:, 0] - box / 2 - 3, mode='text',
                text=[str(i+1) for i in range(n)],
                textfont=dict(color='#ffdd44', size=7),
                hoverinfo='skip', showlegend=False))

    fig.update_layout(**_L, title=title,
                      xaxis=dict(range=[0, W], showgrid=False, zeroline=False, **_A),
                      yaxis=dict(range=[H, 0], scaleanchor='x', scaleratio=1,
                                 showgrid=False, zeroline=False, **_A))
    return fig


def _fig_intens(d):
    t = d.get('thresholds')
    if t is None or len(t) == 0:
        return _waiting('Intensities')
    n = len(t)
    sites = list(range(1, n+1))
    # Marker size shrinks as the array grows so dots don't overlap on dense
    # arrays but stay readable for small ones (~13px @ n<=140, ~8px @ n=225).
    cur_size = float(np.clip(1800.0 / n, 6, 13))
    thr_size = max(4.0, cur_size - 2)
    fig = go.Figure()
    # WebGL scatter — at thousands of sites SVG markers are a major render cost.
    fig.add_trace(go.Scattergl(x=sites, y=t.tolist(), mode='markers', name='Threshold',
                              marker=dict(size=thr_size, color='#777', symbol='circle', line=dict(width=1, color='#999'))))
    ymin, ymax = float(t.min()), float(t.max())
    ci = d.get('cur_intensities')
    if ci is not None:
        logicals = d.get('logicals')
        # Numeric occupancy + 2-stop colorscale (green/red) instead of a list of
        # thousands of hex strings — smaller payload, faster to build.
        if logicals is not None and len(logicals) >= n:
            occ = np.asarray(logicals[:n], dtype=float)
        else:
            occ = np.zeros(n)
        fig.add_trace(go.Scattergl(x=sites, y=ci.tolist(), mode='markers', name='Current',
                                  marker=dict(size=cur_size, symbol='circle',
                                              color=occ, colorscale=[[0, '#e44'], [1, '#0c6']],
                                              cmin=0, cmax=1, line=dict(width=1, color='white'))))
        ymin = min(ymin, float(ci.min()))
        ymax = max(ymax, float(ci.max()))
    # Mean line + 68% (±1σ) band for loaded / empty sites + distance annotation
    if ci is not None and logicals is not None:
        mask = np.array(logicals[:n], dtype=bool) if len(logicals) >= n else np.zeros(n, dtype=bool)

        def _band(values, color, fill, label, yanchor):
            mu = float(values.mean())
            sd = float(values.std())
            # ±1σ band ≈ central 68% of a normal distribution
            fig.add_shape(type='rect', xref='paper', x0=0, x1=1, y0=mu-sd, y1=mu+sd,
                          fillcolor=fill, line=dict(width=0), layer='below')
            fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=mu, y1=mu,
                          line=dict(color=color, width=1.5, dash='dash'))
            fig.add_annotation(text=f'{label}: {mu:.1f} ± {sd:.1f}', xref='paper', y=mu,
                               x=0.99, showarrow=False, xanchor='right', yanchor=yanchor,
                               font=dict(color=color, size=10), bgcolor='rgba(20,20,40,0.6)')
            return mu, sd

        if mask.any():
            mu_loaded, sd_loaded = _band(ci[mask], '#0c6', 'rgba(0,204,102,0.12)', 'Loaded', 'bottom')
            ymin = min(ymin, mu_loaded - sd_loaded)
            ymax = max(ymax, mu_loaded + sd_loaded)
        else:
            mu_loaded = None
        if (~mask).any():
            mu_empty, sd_empty = _band(ci[~mask], '#e44', 'rgba(238,68,68,0.12)', 'Empty', 'top')
            ymin = min(ymin, mu_empty - sd_empty)
            ymax = max(ymax, mu_empty + sd_empty)
        else:
            mu_empty = None
        if mu_loaded is not None and mu_empty is not None:
            delta = mu_loaded - mu_empty
            fig.add_annotation(text=f'Δ = {delta:.2f}', xref='paper', yref='paper',
                               x=0.5, y=1.0, showarrow=False,
                               font=dict(size=12, color='#ffdd44', family='monospace'),
                               bgcolor='rgba(20,20,40,0.8)')

    pad = max((ymax - ymin) * 0.2, 1)
    fig.update_layout(**_L, title='Atom Intensities', xaxis=dict(title='Site', dtick=max(1, n//20), **_A),
                      yaxis=dict(title='Intensity', range=[ymin-pad, ymax+pad], **_A),
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.3)'))
    return fig


def _fig_loading_live(d):
    hist = d.get('loading_history')
    if hist is None or len(hist) == 0:
        return _waiting('Loading Rate')
    hist = np.asarray(hist, dtype=float)
    logicals = d.get('logicals')
    cur = float(np.asarray(logicals).mean()) if logicals is not None and len(logicals) > 0 else None
    # Average over the displayed history window (always populated, unlike
    # loading_rates which only refreshes every UPDATE_LOADING_INTERVAL shots).
    avg = float(hist.mean())

    n = len(hist)
    x = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hist.tolist(), mode='lines+markers',
                              line=dict(color='#0c6', width=1.5),
                              marker=dict(size=4, color='#0c6'),
                              name='Per-shot', hoverinfo='y'))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=avg, y1=avg,
                  line=dict(color='#ffdd44', width=1.5, dash='dash'))
    fig.add_annotation(text=f'Avg: {avg:.1%}', xref='paper', y=avg,
                       x=0.99, showarrow=False, xanchor='right', yanchor='bottom',
                       font=dict(color='#ffdd44', size=10))
    if cur is not None:
        fig.add_annotation(text=f'Current: {cur:.1%}', xref='paper', yref='paper',
                           x=0.5, y=1.0, showarrow=False,
                           font=dict(size=18, color='#0c6', family='monospace'),
                           bgcolor='rgba(20,20,40,0.8)')
    fig.update_layout(**_L, title='Loading Rate (last 100)',
                      xaxis=dict(title='Shot # (oldest → latest)', **_A),
                      yaxis=dict(title='Fraction loaded', autorange=True,
                                 tickformat='.0%', **_A),
                      showlegend=False)
    return fig


def _fig_loading(d, marker_size=12):
    grid, rates = d.get('grid_locations'), d.get('loading_rates')
    if grid is None or rates is None or len(grid) == 0:
        return _waiting('Loading Rates')
    n = len(grid)
    sz = marker_size
    if n < 100:
        mode = 'markers+text'
        text = [f'{r:.0%}' for r in rates]
        tfont = dict(size=7, color='black')
    else:
        mode = 'markers'
        text = None
        tfont = None
    # WebGL + hovertemplate: at thousands of sites, SVG markers and a per-point
    # hovertext string list are both heavy. customdata = [site, rate] feeds both
    # the hover and the click-to-select JS (which reads customdata[0]).
    customdata = np.column_stack([np.arange(1, n + 1), np.asarray(rates)])
    fig = go.Figure(go.Scattergl(
        x=grid[:,1], y=grid[:,0], mode=mode,
        marker=dict(size=sz, color=rates.tolist(), colorscale='RdYlGn', cmin=0, cmax=1,
                    colorbar=dict(title='Rate', len=0.9), line=dict(width=0.5, color='white')),
        text=text, textfont=tfont, textposition='middle center',
        customdata=customdata,
        hovertemplate='Site %{customdata[0]}: %{customdata[1]:.1%}<extra></extra>'))
    fig.update_layout(**_L, title=f'Loading Rates ({n} sites)', clickmode='event',
                      yaxis=dict(autorange='reversed', scaleanchor='x', scaleratio=1,
                                 visible=False, **_A),
                      xaxis=dict(visible=False, **_A))
    return fig


def _fig_infid(d, marker_size=12):
    grid, inf = d.get('grid_locations'), d.get('infidelities')
    if grid is None or inf is None or len(grid) == 0:
        return _waiting('Infidelities')
    n = len(grid)
    log_inf = np.log10(np.clip(inf, 1e-6, 1.0))
    sz = marker_size
    if n < 100:
        mode = 'markers+text'
        text = [f'{v:.0e}' for v in inf]
        tfont = dict(size=6, color='white')
    else:
        mode = 'markers'
        text = None
        tfont = None
    # WebGL + hovertemplate; customdata = [site, infidelity] (the marker colour
    # is log10, so the real value rides in customdata for a readable hover).
    customdata = np.column_stack([np.arange(1, n + 1), np.asarray(inf)])
    fig = go.Figure(go.Scattergl(
        x=grid[:,1], y=grid[:,0], mode=mode,
        marker=dict(size=sz, color=log_inf.tolist(), colorscale='Magma_r', cmin=-4, cmax=-0.3,
                    colorbar=dict(title='log10', len=0.9), line=dict(width=0.5, color='white')),
        text=text, textfont=tfont, textposition='middle center',
        customdata=customdata,
        hovertemplate='Site %{customdata[0]}: %{customdata[1]:.2e}<extra></extra>'))
    fig.update_layout(**_L, title=f'Discrimination Infidelities ({n} sites)', clickmode='event',
                      yaxis=dict(autorange='reversed', scaleanchor='x', scaleratio=1,
                                 visible=False, **_A),
                      xaxis=dict(visible=False, **_A))
    return fig


def _fig_shift(d):
    hm = d.get('grid_shift_heatmap')
    if hm is None:
        return _waiting('Grid Shift')
    R = (hm.shape[0]-1)//2
    fig = go.Figure(go.Heatmap(z=hm, x0=-R, dx=1, y0=-R, dy=1, colorscale='Viridis',
                                showscale=True, colorbar=dict(len=0.9)))
    hist = d.get('grid_shift_history', [])
    title = 'Grid Shift Heatmap'
    if hist:
        dy, dx = hist[-1]
        fig.add_trace(go.Scatter(x=[dx], y=[dy], mode='markers',
                                 marker=dict(symbol='cross', size=14, color='red', line=dict(width=2)),
                                 showlegend=False))
        title = f'Grid Shift (dy={dy}, dx={dx})'
    fig.update_layout(**_L, title=title, xaxis=dict(title='dx', **_A),
                      yaxis=dict(title='dy', autorange='reversed', **_A))
    return fig


def _fig_scan_curve(d, cbar_scale='01'):
    sc = d.get('scan_curve')
    if sc is None or sc.get('mode') == 'undefined':
        return _waiting('Scan Curve')

    # --- 2-D heatmap ---
    if sc.get('ndim', 1) >= 2:
        return _fig_scan_2d(d, sc, cbar_scale=cbar_scale)

    # --- 1-D scatter with error bars ---
    x = sc['scan_x']
    y = sc['y_mean']
    err = sc['y_sem']
    n_reps = sc['n_reps']
    mode = sc['mode']
    mask = n_reps > 0
    if not np.any(mask):
        return _waiting('Scan Curve')
    x, y, err, n_reps = x[mask], y[mask], err[mask], n_reps[mask]

    scale = d.get('plot_scale', 1)
    if scale and scale != 0 and scale != 1:
        x_disp = x * scale
    else:
        x_disp = x

    scan_name = d.get('scan_name', 'Scan')
    x_label = d.get('scan_param_path') or scan_name
    if mode == 'survival':
        y_label = 'Survival'
    elif mode == 'rearrangement':
        y_label = 'Rearrangement Success (mean of logic2)'
    else:
        y_label = 'Loading Rate'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_disp, y=y, error_y=dict(type='data', array=err, visible=True, thickness=1.5),
        mode='markers', marker=dict(size=6, color='#44aaff'),
        hoverinfo='text', hovertext=[f'{x_label}={xi:.4g}, {y_label}={yi:.3f}+/-{ei:.3f} (n={ni})'
                                      for xi, yi, ei, ni in zip(x_disp, y, err, n_reps)]))
    title_text = _scan_title(f'{scan_name} ({int(n_reps.mean())} reps/pt)',
                             d.get('scan_filename'))
    fig.update_layout(**_L, title=title_text,
                      xaxis=dict(title=x_label, **_A),
                      yaxis=dict(title=y_label, range=[-0.05, 1.05], **_A))
    return fig


def _scan_title(main, fname):
    """Scan-panel title with the run/folder name shown inline and clearly
    (bright, larger) next to the main title rather than as a faint subtitle."""
    if not fname:
        return main
    run = fname[:-3] if fname.endswith('.h5') else fname  # strip .h5 → run folder
    return f'{main}  <span style="font-size:14px;color:#7cc4ff">— {run}</span>'


def _fmt_tick(v):
    """Compact axis label: SI suffix for big magnitudes, else general format."""
    av = abs(float(v))
    if av >= 1e9:
        return f'{v/1e9:g}G'
    if av >= 1e6:
        return f'{v/1e6:g}M'
    if av >= 1e3:
        return f'{v/1e3:g}k'
    return f'{v:g}'


def _tickset(vals):
    """Tick indices + labels for an equal-step (index) axis, thinned to ~12
    ticks max so labels don't crowd on long scans."""
    n = len(vals)
    step = max(1, int(np.ceil(n / 12)))
    idx = list(range(0, n, step))
    return idx, [_fmt_tick(vals[i]) for i in idx]


def _fig_scan_2d(d, sc, cbar_scale='01'):
    """Render a 2-D scan as a survival/loading heatmap."""
    heatmap = sc.get('heatmap')
    n_reps = sc.get('n_reps')
    if heatmap is None:
        return _waiting('Scan 2D')

    # Mask cells with no data
    mask = (n_reps is not None) and np.any(n_reps > 0)
    if not mask:
        return _waiting('Scan 2D')

    x_vals = sc['x_values']
    y_vals = sc['y_values']
    x_name = sc.get('x_name', 'dim0')
    y_name = sc.get('y_name', 'dim1')
    mode = sc.get('mode', 'survival')
    scan_name = d.get('scan_name', 'Scan')

    scale = d.get('plot_scale', 1)
    if scale and scale != 0 and scale != 1:
        x_disp = x_vals * scale
    else:
        x_disp = x_vals

    z = np.where(n_reps > 0, heatmap, np.nan)
    y_label = 'Survival' if mode == 'survival' else 'Loading'
    avg_reps = int(n_reps[n_reps > 0].mean()) if np.any(n_reps > 0) else 0

    # Colorbar z-range: 'auto' lets Plotly autoscale to the data, '01' pins 0–1.
    if cbar_scale == 'auto':
        zmin = zmax = None
    else:
        zmin, zmax = 0, 1

    # Plot against equal-step indices so EVERY cell is the same size even when
    # the scan values are unevenly spaced; the real values are restored as tick
    # labels and ride in customdata for the hover (along with reps + error).
    nx, ny = len(x_disp), len(y_vals)
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    xv = np.asarray(x_disp, dtype=float)
    yv = np.asarray(y_vals, dtype=float)
    Xv = np.broadcast_to(xv.reshape(1, nx), (ny, nx))   # actual x per cell
    Yv = np.broadcast_to(yv.reshape(ny, 1), (ny, nx))   # actual y per cell

    sem = sc.get('sem')
    if sem is not None:
        # customdata: [x_val, y_val, reps, error] per cell
        customdata = np.dstack([Xv, Yv, n_reps, sem])
        hovertemplate = (f'{x_name}=%{{customdata[0]:.4g}}<br>'
                         f'{y_name}=%{{customdata[1]:.4g}}<br>'
                         f'{y_label}=%{{z:.3f}} ± %{{customdata[3]:.3f}}<br>'
                         f'reps=%{{customdata[2]:d}}<extra></extra>')
    else:
        customdata = np.dstack([Xv, Yv, n_reps])
        hovertemplate = (f'{x_name}=%{{customdata[0]:.4g}}<br>'
                         f'{y_name}=%{{customdata[1]:.4g}}<br>'
                         f'{y_label}=%{{z:.3f}}<br>reps=%{{customdata[2]:d}}<extra></extra>')

    fig = go.Figure(go.Heatmap(
        z=z, x=x_idx, y=y_idx,
        colorscale='Viridis', zmin=zmin, zmax=zmax,
        colorbar=dict(title=y_label, len=0.9),
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))

    # Red box around every cell updated in the latest batch (the cells
    # currently being scanned). On the index grid every cell is unit-sized.
    cur = sc.get('current') or []
    if isinstance(cur, dict):       # backward-compat: old single-cell format
        cur = [cur]
    for cell in cur:
        xi, yi = cell.get('x_idx'), cell.get('y_idx')
        if (xi is None or yi is None
                or not (0 <= xi < nx) or not (0 <= yi < ny)):
            continue
        fig.add_shape(type='rect', xref='x', yref='y',
                      x0=xi-0.5, x1=xi+0.5, y0=yi-0.5, y1=yi+0.5,
                      line=dict(color='#ff0000', width=3), fillcolor='rgba(0,0,0,0)',
                      layer='above')

    xtv, xtt = _tickset(xv)
    ytv, ytt = _tickset(yv)
    title_text = _scan_title(f'{scan_name} ({avg_reps} reps/pt)', d.get('scan_filename'))
    fig.update_layout(**_L, title=title_text,
                      xaxis=dict(title=x_name, tickmode='array',
                                 tickvals=xtv, ticktext=xtt, **_A),
                      yaxis=dict(title=y_name, tickmode='array',
                                 tickvals=ytv, ticktext=ytt, **_A))
    return fig


def _fig_avghist(d):
    fig = go.Figure()
    has_live_f = d.get('live_gauss_fits') is not None
    # Show loaded fit only when no live fit
    if not has_live_f:
        _add_avg_fit_curve(fig, d.get('loaded_gauss_fits'), '#888', 'Loaded fit', faint=True)
    # Live fit curve (replaces loaded when available)
    _add_avg_fit_curve(fig, d.get('live_gauss_fits'), '#44aaff', 'Live fit', faint=False)
    # Live bars
    _add_avg_bars(fig, d.get('live_hist_data'), d.get('n_accum_shots', 0))
    fig.update_layout(**_L, title='Avg Histogram', barmode='overlay',
                      xaxis=dict(title='Intensity', **_A), yaxis=dict(title='Density', **_A),
                      legend=dict(x=0.5, y=0.99, bgcolor='rgba(0,0,0,0.3)', font=dict(size=8)))
    return fig


def _add_avg_fit_curve(fig, fits, color, name, faint=False):
    if not fits or not isinstance(fits, list):
        return
    valid = [g['params'] for g in fits if isinstance(g, dict) and g.get('params') is not None]
    if not valid:
        return
    # Vectorized: stack all params into (N,6) array, compute all curves at once
    P = np.array(valid)  # (N, 6): mu1, sig1, w1, mu2, sig2, w2
    xmin = float((P[:, 0] - 4*P[:, 1]).min())
    xmax = float((P[:, 3] + 4*P[:, 4]).max())
    xf = np.linspace(xmin, xmax, 200)
    # Broadcast: xf(200,) vs P(N,6) → (N,200) for each Gaussian
    dx1 = (xf[None, :] - P[:, 0:1]) / P[:, 1:2]  # (N, 200)
    dx2 = (xf[None, :] - P[:, 3:4]) / P[:, 4:5]
    g1 = P[:, 2:3] / (P[:, 1:2] * np.sqrt(2*np.pi)) * np.exp(-0.5 * dx1**2)
    g2 = P[:, 5:6] / (P[:, 4:5] * np.sqrt(2*np.pi)) * np.exp(-0.5 * dx2**2)
    avg = (g1 + g2).mean(axis=0)
    op = 0.3 if faint else 0.8
    fig.add_trace(go.Scatter(x=xf, y=avg, mode='lines',
                              line=dict(color=color, width=1.5, dash='dot' if faint else 'solid'),
                              fill='tozeroy', fillcolor=f'rgba(136,136,136,{0.05 if faint else 0.1})',
                              name=name, opacity=op))


def _add_avg_bars(fig, hist_data, n_shots):
    if not hist_data or not isinstance(hist_data, list) or len(hist_data) == 0:
        return
    # Common x-axis across all sites, then interpolate each site's density
    all_c = np.concatenate([h['bin_centers'] for h in hist_data])
    centers = np.linspace(all_c.min(), all_c.max(), 50)
    avg = np.zeros(50)
    for h in hist_data:
        avg += np.interp(centers, h['bin_centers'], h['counts'], left=0, right=0)
    avg /= len(hist_data)
    bw = (centers[-1] - centers[0]) / (len(centers) - 1) * 0.85
    fig.add_trace(go.Bar(x=centers, y=avg, marker_color='#4488cc', opacity=0.8,
                         width=bw, name=f'Live ({n_shots})'))


# ---- Rep site histograms ----

def _figs_reps(d):
    sites = d.get('hist_rep_sites')
    if not sites:
        return [_waiting('Site Hist')] * 4
    labels = ['Best', 'Worst', 'Random', 'Random']
    figs = []
    for k in range(4):
        if k < len(sites):
            figs.append(_build_hist(d, sites[k], f'{labels[k]}: Site {sites[k]+1}'))
        else:
            figs.append(_waiting('Site Hist'))
    return figs


# ---- Single-site histogram (shared builder) ----

def _fig_site(d, idx):
    fig = _build_hist(d, idx, f'Site {idx+1} Histogram')
    info = []
    t = d.get('thresholds')
    if t is not None and idx < len(t):
        info.append(html.Div(f'Threshold: {t[idx]:.2f}'))
    fits = d.get('live_gauss_fits') or d.get('loaded_gauss_fits')
    if fits and isinstance(fits, list) and idx < len(fits):
        p = fits[idx].get('params') if isinstance(fits[idx], dict) else None
        if p is not None:
            info.extend([html.Div(f'mu_empty: {p[0]:.2f}'), html.Div(f'mu_atom: {p[3]:.2f}'),
                         html.Div(f'sig_empty: {p[1]:.2f}'), html.Div(f'sig_atom: {p[4]:.2f}')])
    inf = d.get('infidelities')
    if inf is not None and idx < len(inf):
        v = float(inf[idx])
        c = '#4c4' if v < 0.01 else '#cc4' if v < 0.05 else '#c44'
        info.append(html.Div(html.Span(f'Infidelity: {v:.2e}', style={'color': c, 'fontWeight': 'bold'})))
    rates = d.get('loading_rates')
    if rates is not None and idx < len(rates):
        info.append(html.Div(f'Loading: {rates[idx]:.1%}'))
    info.append(html.Div(f'Shots: {d.get("n_accum_shots", 0)}', style={'color': '#888'}))
    return fig, info


def _build_hist(d, idx, title):
    """Build site histogram: loaded fit (background) + live bars + live fit (foreground)."""
    fig = go.Figure()
    loaded_fits = d.get('loaded_gauss_fits')
    live_hist = d.get('live_hist_data')
    live_fits = d.get('live_gauss_fits')
    thresholds = d.get('thresholds')
    inf = d.get('infidelities')

    has_live = live_hist is not None and isinstance(live_hist, list) and idx < len(live_hist)
    has_loaded_f = loaded_fits is not None and isinstance(loaded_fits, list) and idx < len(loaded_fits)
    has_live_f = live_fits is not None and isinstance(live_fits, list) and idx < len(live_fits)

    if not has_live and not has_loaded_f and not has_live_f:
        return _waiting(title)

    # Determine x range from histogram data (not fit tails)
    xmin, xmax = 195, 210  # fallback
    if has_live:
        bc = live_hist[idx]['bin_centers']
        xmin, xmax = float(bc.min()), float(bc.max())
        pad = (xmax - xmin) * 0.05
        xmin -= pad
        xmax += pad
    elif has_loaded_f:
        p = loaded_fits[idx].get('params') if isinstance(loaded_fits[idx], dict) else None
        if p is not None:
            xmin, xmax = p[0] - 5*p[1], p[3] + 5*p[4]

    # Layer 1: Loaded fit curves (faint background) — only when no live fit
    if has_loaded_f and not has_live_f:
        p = loaded_fits[idx].get('params') if isinstance(loaded_fits[idx], dict) else None
        if p is not None:
            xf = np.linspace(xmin, xmax, 200)
            y1 = p[2]*norm.pdf(xf, p[0], p[1])
            y2 = p[5]*norm.pdf(xf, p[3], p[4])
            fig.add_trace(go.Scatter(x=xf, y=y1, mode='lines', line=dict(color='#44cc44', width=1.5, dash='dot'),
                                     fill='tozeroy', fillcolor='rgba(68,204,68,0.08)', name='Empty (loaded)', opacity=0.5))
            fig.add_trace(go.Scatter(x=xf, y=y2, mode='lines', line=dict(color='#cc44cc', width=1.5, dash='dot'),
                                     fill='tozeroy', fillcolor='rgba(204,68,204,0.08)', name='Atom (loaded)', opacity=0.5))

    # Layer 2: Live histogram bars
    if has_live:
        h = live_hist[idx]
        bw = np.diff(h['bin_centers']).mean() * 0.85 if len(h['bin_centers']) > 1 else 1
        fig.add_trace(go.Bar(x=h['bin_centers'], y=h['counts'], marker_color='#5588bb',
                             opacity=0.7, width=bw, name='Live'))

    # Layer 3: Live fit curves (solid, on top)
    if has_live_f:
        p = live_fits[idx].get('params') if isinstance(live_fits[idx], dict) else None
        if p is not None:
            xf = np.linspace(xmin, xmax, 200)
            y1 = p[2]*norm.pdf(xf, p[0], p[1])
            y2 = p[5]*norm.pdf(xf, p[3], p[4])
            fig.add_trace(go.Scatter(x=xf, y=y1, mode='lines', line=dict(color='#44cc44', width=2),
                                     name='Empty (live)'))
            fig.add_trace(go.Scatter(x=xf, y=y2, mode='lines', line=dict(color='#cc44cc', width=2),
                                     name='Atom (live)'))
            fig.add_trace(go.Scatter(x=xf, y=y1+y2, mode='lines', line=dict(color='white', width=1.5, dash='dot'),
                                     name='Sum'))

    # Threshold line
    if thresholds is not None and idx < len(thresholds):
        fig.add_vline(x=float(thresholds[idx]), line=dict(color='#ff4444', width=2, dash='dash'))

    # Infidelity badge (above legend, top-right corner)
    if inf is not None and idx < len(inf):
        v = float(inf[idx])
        c = '#4c4' if v < 0.01 else '#cc4' if v < 0.05 else '#c44'
        fig.add_annotation(text=f'Infid: {v:.1e}', xref='paper', yref='paper',
                           x=0.99, y=1.0, xanchor='right', yanchor='top',
                           showarrow=False, font=dict(size=10, color=c, family='monospace'),
                           bgcolor='rgba(20,20,40,0.8)', bordercolor=c)

    fig.update_layout(**_L, title=title, xaxis=dict(title='Intensity', **_A),
                      yaxis=dict(title='Density', **_A),
                      legend=dict(x=0.99, y=0.88, xanchor='right', yanchor='top',
                                  bgcolor='rgba(0,0,0,0.3)', font=dict(size=7)),
                      barmode='overlay')
    return fig


# ---- Queue panel ----

def _q_fmt_dur(start_ts, end_ts=None):
    if not start_ts:
        return ''
    end = end_ts if end_ts else time.time()
    elapsed = max(0, int(end - start_ts))
    m, s = divmod(elapsed, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def _q_axes(axes):
    if not axes:
        return '--'
    parts = []
    for ax in axes:
        n = ax.get('name', '') or f"dim{ax.get('dim', '?')}"
        n = n.split('.')[-1] if '.' in n else n
        try:
            lo = float(ax.get('min', 0))
            hi = float(ax.get('max', 0))
            parts.append(f'{n}: {lo:g}..{hi:g} ({ax.get("npts", 0)}pt)')
        except (TypeError, ValueError):
            parts.append(f'{n}: ({ax.get("npts", 0)}pt)')
    return ' x '.join(parts)


_Q_COLS = [
    ('marker', '',         30),
    ('id',     'ID',       40),
    ('scan',   'Scan',     None),
    ('seq',    'Seq',      None),
    ('axes',   'Axes',     None),
    ('reps',   'Reps',     50),
    ('file',   'Data ID',  150),
    ('status', 'Status',   110),
]


def _q_row(entry, kind):
    summary = entry.get('summary') or {}
    scan = summary.get('scan_name') or summary.get('scan_filename') or '--'
    seq = entry.get('seqName') or '--'
    axes = _q_axes(summary.get('axes'))
    reps = summary.get('total_per_group') or summary.get('num_per_group') or '--'
    fid = entry.get('file_id') or ''
    eid = entry.get('id', '')
    if kind == 'running':
        marker, color = '>', '#0c6'
        status = f'run {_q_fmt_dur(entry.get("start_ts"))}'
    elif kind == 'history':
        st = entry.get('status') or entry.get('state') or ''
        dur = _q_fmt_dur(entry.get('start_ts'), entry.get('finish_ts'))
        if st == 'ok':
            marker, color = '+', '#888'
            status = f'ok {dur}' if dur else 'ok'
        else:
            marker, color = 'x', '#c44'
            status = st if st else 'error'
    else:
        marker, color = '.', TEXT
        status = 'queued'

    cells = {
        'marker': marker, 'id': str(eid), 'scan': scan, 'seq': seq,
        'axes': axes, 'reps': str(reps), 'file': fid, 'status': status,
    }
    tds = []
    for cid, _, w in _Q_COLS:
        style = {'padding': '3px 8px', 'whiteSpace': 'nowrap',
                 'overflow': 'hidden', 'textOverflow': 'ellipsis'}
        if w is not None:
            style['width'] = f'{w}px'
        if cid in ('axes', 'scan'):
            style['maxWidth'] = '320px'
        tds.append(html.Td(cells[cid], style=style))
    return html.Tr(tds, style={'color': color, 'borderBottom': '1px solid #1a1a30'})


def _render_queue_panel(q):
    title_style = {'fontSize': '14px', 'color': '#e94560', 'fontWeight': 'bold',
                   'marginBottom': '6px'}
    if q is None:
        return [html.Div('Scan Queue', style=title_style),
                html.Div('No queue data yet.',
                         style={'color': '#666', 'fontStyle': 'italic'})]

    head_cells = [html.Th(t, style={'padding': '4px 8px', 'textAlign': 'left',
                                     'color': '#bbb', 'fontWeight': '500',
                                     'borderBottom': '1px solid #333'})
                  for _, t, _ in _Q_COLS]

    rows = []
    running = q.get('running')
    if running and running.get('id') is not None:
        rows.append(_q_row(running, 'running'))
    for e in q.get('queued', []) or []:
        if e.get('id') is not None:
            rows.append(_q_row(e, 'queued'))
    hist = q.get('history') or []
    if hist:
        rows.append(html.Tr([html.Td(
            '-- history --', colSpan=len(_Q_COLS),
            style={'color': '#888', 'textAlign': 'center', 'padding': '4px',
                   'fontStyle': 'italic'})]))
        for e in hist[:30]:
            if e.get('id') is not None:
                rows.append(_q_row(e, 'history'))

    if not rows:
        body = html.Div('Queue empty', style={'color': '#666',
                                              'fontStyle': 'italic'})
    else:
        body = html.Table(
            [html.Thead(html.Tr(head_cells)), html.Tbody(rows)],
            style={'width': '100%', 'borderCollapse': 'collapse',
                   'fontFamily': 'Consolas, monospace', 'fontSize': '11px'})

    total_q = len(q.get('queued', []) or [])
    if running:
        total_q += 1
    title = html.Div([
        html.Span('Scan Queue'),
        html.Span(f'  ({total_q} active)', style={
            'color': '#888', 'fontSize': '11px', 'fontWeight': 'normal'}),
    ], style=title_style)
    return [title, body]


# ---- SLM panel ----

def _render_slm_panel(slm):
    """Render the SLM hardware row from the proxy's pickle snapshot.

    Layout (single row):
      [SLM phase PNG] [SLM camera PNG] [lock + health + rearrange-diag readout]

    When the proxy isn't running or the SLM PC is offline, the whole row
    greys out with a clear banner and a "last poll" timestamp.
    """
    title_style = {'fontSize': '14px', 'color': '#e94560', 'fontWeight': 'bold',
                   'marginBottom': '6px'}
    title = html.Span('SLM Hardware', style=title_style)

    if slm is None:
        return [
            html.Div([title, html.Span('  (proxy disabled)', style={
                'color': '#888', 'fontSize': '11px', 'fontWeight': 'normal'})]),
            html.Div('No data yet — proxy not running, or --no-slm was passed.',
                     style={'color': '#666', 'fontStyle': 'italic'}),
        ]

    offline = bool(slm.get('slm_offline'))
    slm_url = slm.get('slm_url', '')
    status_color = '#888' if offline else '#0c6'
    status_text = 'OFFLINE' if offline else 'online'
    header = html.Div([
        title,
        html.Span(f'  {status_text}', style={
            'color': status_color, 'fontSize': '11px', 'fontWeight': 'bold'}),
        html.Span(f'  {slm_url}', style={
            'color': '#888', 'fontSize': '11px', 'fontWeight': 'normal'}),
    ], style={'marginBottom': '8px'})

    if offline:
        last_err = slm.get('last_error_msg', {})
        err_lines = [html.Div(f'{k}: {v}', style={'fontSize': '10px', 'color': '#c44'})
                     for k, v in (last_err or {}).items()]
        return [header,
                html.Div('SLM PC unreachable. Last errors:' if err_lines
                         else 'SLM PC unreachable.',
                         style={'color': '#888', 'fontStyle': 'italic',
                                'marginBottom': '6px'}),
                *err_lines]

    # Online: build the three sub-panels.
    phase_uri = _png_bytes_to_data_uri(slm.get('phase_png'))
    cam_uri = _png_bytes_to_data_uri(slm.get('camera_png'))
    lock = slm.get('lock_status') or {}
    health = slm.get('health') or {}
    diag = slm.get('rearrange_diag') or {}

    def _png_block(uri, label):
        if uri is None:
            return html.Div(label + ' — waiting…',
                            style={'color': '#666', 'fontStyle': 'italic',
                                   'width': '300px', 'height': '300px',
                                   'border': '1px solid #1a1a30',
                                   'display': 'flex', 'alignItems': 'center',
                                   'justifyContent': 'center'})
        return html.Div([
            html.Div(label, style={'fontSize': '11px', 'color': '#bbb',
                                   'marginBottom': '3px'}),
            html.Img(src=uri, style={
                'width': '300px', 'maxHeight': '300px', 'objectFit': 'contain',
                'imageRendering': 'pixelated',
                'border': '1px solid #1a1a30'}),
        ], style={'flex': '0 0 auto'})

    # Lock + health text block (rightmost panel).
    info_rows = []
    if isinstance(lock, dict):
        for dev, st in (lock or {}).items():
            if isinstance(st, dict):
                holder = st.get('holder') or st.get('client_id') or '—'
                age_s = st.get('age_s')
                age_txt = f'{age_s:.0f}s' if isinstance(age_s, (int, float)) else ''
                info_rows.append(html.Div(
                    f'lock[{dev}]: held by {holder} {age_txt}',
                    style={'fontSize': '11px', 'color': '#bbb'}))
    uptime = health.get('uptime_s') if isinstance(health, dict) else None
    if isinstance(uptime, (int, float)):
        info_rows.append(html.Div(
            f'health: uptime={int(uptime)}s',
            style={'fontSize': '11px', 'color': '#bbb'}))
    # Most recent rearrange entry, if any.
    entries = diag.get('entries') if isinstance(diag, dict) else None
    if entries:
        latest = entries[-1]
        d = latest.get('diag') or {}
        total_ms = d.get('total_ms')
        nsteps = d.get('nsteps')
        n_loaded = d.get('n_loaded')
        info_rows.append(html.Div('latest rearrange:', style={
            'fontSize': '11px', 'color': '#ffdd44', 'marginTop': '6px',
            'fontWeight': 'bold'}))
        info_rows.append(html.Div(
            f'  total={total_ms}ms nsteps={nsteps} n_loaded={n_loaded}',
            style={'fontSize': '10px', 'color': '#bbb',
                   'fontFamily': 'Consolas, monospace'}))
    if not info_rows:
        info_rows.append(html.Div('(no lock / health / diag data yet)',
                                  style={'fontStyle': 'italic', 'color': '#666',
                                         'fontSize': '11px'}))

    body = html.Div(style={'display': 'flex', 'gap': '12px',
                           'alignItems': 'flex-start'}, children=[
        _png_block(phase_uri, 'phase (SLM)'),
        _png_block(cam_uri, 'camera (SLM)'),
        html.Div(info_rows, style={'flex': '1', 'minWidth': '0'}),
    ])
    return [header, body]


def _png_bytes_to_data_uri(png_bytes):
    """Wrap raw PNG bytes from the SLM proxy as a data URI for <img src=…>."""
    if not png_bytes or not isinstance(png_bytes, (bytes, bytearray)):
        return None
    return 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()
