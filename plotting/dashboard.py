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
import threading
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
            # Pass the parent (this process) PID so the child can watchdog
            # us. If we die unexpectedly (terminal close, taskkill, crash)
            # the child notices in ~3 s and self-terminates, releasing
            # port 8050 instead of becoming an orphan.
            self._proc = multiprocessing.Process(
                target=_dash_main,
                args=(self._host, self._port, _DATA_FILE, os.getpid()),
                daemon=True)
            self._proc.start()
            logger.info('Dashboard process started (pid=%d, parent=%d) at http://%s:%d',
                        self._proc.pid, os.getpid(), self._host, self._port)

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

# Heavy keys excluded from /api/snapshot output. Only the data URIs
# themselves are oversized -- the shape/percentile metadata is a
# few floats and is useful for the HTML dashboard's Plotly.js
# image-with-overlay panel, so we keep it on snapshot.
_SNAPSHOT_HEAVY_KEYS = ('_img_data_uri', '_img2_data_uri', '_img_mid_data_uri')


# In-flight xref builds (scan dir -> Popen), so a scan loaded repeatedly in the
# Sequence tab spawns at most one background provenance build at a time.
_XREF_BUILDS = {}
_XREF_BUILD_LOCK = threading.Lock()


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


def _parse_reconstruct_result(stdout):
    """Extract the offline reconstruct driver's ``RECONSTRUCT_RESULT:{json}`` line.

    The engine prints its own chatter; the driver emits exactly one result line. Scans from
    the end (the result is last). Returns the parsed dict, or None if absent/unparseable.
    """
    import json as _json
    prefix = 'RECONSTRUCT_RESULT:'
    for line in reversed((stdout or '').splitlines()):
        i = line.find(prefix)
        if i != -1:
            try:
                return _json.loads(line[i + len(prefix):])
            except ValueError:
                return None
    return None


def _shot_frame_png_bytes(h5_path, row):
    """Return PNG bytes for a single camera frame (``/imgs[row]``).

    Frames are min-max normalized to uint8 (same contrast treatment as
    the averaged-image card). ``/imgs`` is chunked one-frame-per-chunk,
    so this reads exactly one chunk. Returns None when the row is out of
    range or there's no usable ``/imgs`` dataset.
    """
    import io as _io
    import h5py
    from PIL import Image
    with h5py.File(h5_path, 'r') as f:
        d = f.get('imgs')
        if d is None or d.ndim != 3:
            return None
        if row < 0 or row >= d.shape[0]:
            return None
        frame = d[row].astype(np.float64)
    lo = float(np.nanmin(frame))
    hi = float(np.nanmax(frame))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.zeros(frame.shape, dtype=np.uint8)
    else:
        scaled = ((frame - lo) / (hi - lo) * 255.0).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(scaled).save(buf, format='PNG')
    return buf.getvalue()


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


def _runs_grid_response(scan_id):
    """Phase 4 helper: return per-run grid sidecar for a scan.

    Sidecar-first: returns `<scan_dir>/slm_grid.json` when Phase 4 sync
    has saved it. Otherwise falls through to a live SLM passthrough
    against `/slm/runs/<scan_id>/grid_sidecar`. 404 if neither has it
    (typical for runs whose SLM build predates the rearrange_grid_sidecar
    writer).
    """
    from flask import jsonify
    scan_dir = _resolve_scan_dir(scan_id)
    if scan_dir:
        sidecar = os.path.join(scan_dir, 'slm_grid.json')
        if os.path.exists(sidecar):
            try:
                with open(sidecar, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                logger.warning('slm_grid.json read failed (%s): %s',
                               sidecar, e)
    # Live passthrough.
    try:
        from yb_analysis.slm_sync import SlmSyncClient
        client = SlmSyncClient()
        data = client.get_grid_sidecar(scan_id)
        if data is None:
            return jsonify({'error': 'no grid sidecar for this scan_id'}), 404
        return jsonify(data)
    except Exception as e:
        logger.warning('runs_grid passthrough failed: %s', e)
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
        '/api/runs/<scan_id>/diag':     'per-scan SLM diag rows: synced sidecar first, then live SLM passthrough',
        '/api/runs/<scan_id>/code':     'per-scan code-snapshot manifest: synced sidecar first, then live SLM passthrough',
        '/api/runs/<scan_id>/grid':     'per-scan grid sidecar (init/target knm coords + grid_rotation): synced sidecar first, then live SLM passthrough',
        '/api/runs/<scan_id>/analysis': 'lab-side per-scan analysis: survival, loading, diag aggregate, sweep description, code/grid pointers',
        '/api/runs/<scan_id>/avg_image': 'PNG of mean image across all shots (single-image scans only; computed on first call, cached as avg_image.png next to data_*.h5)',
        '/api/sequence/list':   'flattened-sequence index for a scan: ?scan_id= or ?folder= -> {files, points, scanned_axes}',
        '/api/sequence/figure': 'Plotly figure for selected channels: ?scan_id=|folder= &file= &seq= &chns=a,b,c',
        '/api/sequence/params': 'parameters tree (+ scanned_paths) for a scan, from the .json sidecar (expConfig+base.params+base.vars), else .seq-embedded: ?scan_id=|folder=',
        '/api/sequence/xref':   'param<->channel PROVENANCE (build-time, from sequence/xref.json) for the loaded .seq: -> {available, param_to_channels, channel_to_params}',
        '/api/sequence/build_xref': 'POST ?scan_id=|folder=: build sequence/xref.json in the BACKGROUND (live-lib provenance_scan.py subprocess) for a scan with a .seq but no xref; -> {ok,started} / {ok,available} / {ok:False}',
        '/api/sequence/reconstruct': 'POST ?scan_id=: regenerate a scan\'s missing .seq via the engine-python driver (use_dummy_device subprocess); -> {ok,n_seq,approximate} or {deferred}',
        '/api/sequence/dump_toggle': 'GET/POST the "save sequence dumps" toggle (pyctrl runtime_state flag); POST ?on=1|0',
        '/api/sequence/scans':  'all scans (Analysis-picker list) flagged with has_seq + has_snapshot + has_descriptor (three-state picker): -> {scans:[{scan_id,name,swept,has_seq,n_seq,has_snapshot,has_descriptor}]}',
        '/api/sequence/pick_folder': 'POST: open a native folder picker on the lab PC desktop -> {path}',
        '/api/queue/submit':                              'POST: submit a scan descriptor (JSON body) to the SequenceRunner queue',
        '/api/queue/cancel/<int:entry_id>':               'POST: cancel a queued job or descriptor by id',
        '/api/queue/move/<int:entry_id>/<direction>':     'POST: move a queued entry up/down within its kind',
        '/api/queue/requeue/<int:entry_id>':              'POST: re-submit an existing entry\'s original descriptor (same params) as a new descriptor',
        '/api/affine/current':       'current global SLM->camera affine (2x3 + rotation/scale/rms/coverage/last_scan_id)',
        '/api/affine/history':       'bounded history of past affines + current',
        '/api/affine/rollback':      'POST: restore the most recent history affine',
        '/api/patterns':             'registered loading patterns (compact metadata; no big arrays)',
        '/api/patterns/<name>':      'full pattern record incl. knm positions + per-site phases',
        '/api/patterns/<name>/refresh': 'POST: re-derive a pattern from the SLM (force)',
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
        # Strip the megabyte-scale image data URIs (Live tab fetches PNG
        # bytes via /api/live/imageN), but keep _img_shape / vlo / vhi so
        # the dashboard's Plotly.js heatmap-with-overlay knows the
        # geometry + colour range.
        return jsonify(_to_jsonable(
            {k: v for k, v in d.items() if k not in _SNAPSHOT_HEAVY_KEYS}))

    # ---- Loading-pattern affine + registry (calibration) ----
    # The global SLM->camera affine and the per-pattern grid registry live as
    # JSON under <PATH_PREFIX>/yb_dashboard_state/; these routes expose them
    # read-only (plus guarded rollback/refresh) for the Calibration card.

    @server.route('/api/affine/current')
    def _api_affine_current():
        from yb_analysis.analysis import affine_transform as _aff
        return jsonify(_to_jsonable(_aff.load_affine() or {}))

    @server.route('/api/affine/history')
    def _api_affine_history():
        from yb_analysis.analysis import affine_transform as _aff
        try:
            data = _aff._read()
        except Exception:
            data = {'current': None, 'history': []}
        return jsonify(_to_jsonable({'current': data.get('current'),
                                     'history': data.get('history', [])}))

    @server.route('/api/affine/rollback', methods=['POST'])
    def _api_affine_rollback():
        from yb_analysis.analysis import affine_transform as _aff
        ok = _aff.rollback()
        return jsonify(_to_jsonable({'ok': ok, 'current': _aff.load_affine()})), \
            (200 if ok else 409)

    @server.route('/api/patterns')
    def _api_patterns():
        from yb_analysis.analysis import pattern_registry as _reg
        return jsonify(_to_jsonable({'patterns': _reg.list_patterns()}))

    @server.route('/api/patterns/<name>')
    def _api_pattern(name):
        from yb_analysis.analysis import pattern_registry as _reg
        rec = _reg.get_pattern(name)
        if rec is None:
            return jsonify({'error': 'pattern not found'}), 404
        return jsonify(_to_jsonable(rec))

    @server.route('/api/patterns/<name>/refresh', methods=['POST'])
    def _api_pattern_refresh(name):
        from yb_analysis.analysis import pattern_registry as _reg
        rec = _reg.get_pattern(name)
        if rec is None:
            return jsonify({'error': 'pattern not found (refresh needs an '
                                     'existing record for its base_phase_path)'}), 404
        try:
            out = _reg.fetch_or_refresh_pattern(
                name, base_phase_path=rec.get('base_phase_path'),
                default_loading_zernike=rec.get('default_loading_zernike'),
                order=rec.get('order', 'col'),
                legacy_zerniked=rec.get('legacy_zerniked', False),
                baked_zernike=rec.get('baked_zernike'), force=True)
            return jsonify(_to_jsonable(
                {'ok': out is not None,
                 'pattern': _reg._compact(out) if out else None}))
        except Exception as ex:
            logger.exception('pattern refresh failed')
            return jsonify({'error': str(ex)}), 500

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

    @server.route('/api/runs/<scan_id>/diag_live')
    def _api_runs_diag_live(scan_id):
        """Incremental live-diag pull-poll (Phase 5.5 Track D).

        While a scan is in flight its synced sidecar doesn't exist yet, so
        this always goes live to the SLM PC's incremental diag endpoint
        (``?since_seq_id=N``) and returns only rows newer than ``since``.
        The browser keeps its own ring buffer and stops polling when the
        scan finishes (post-scan sync then writes the final sidecar). 503
        when the SLM PC is unreachable — the caller just retries next tick.
        """
        from flask import jsonify, request
        try:
            since = request.args.get('since_seq_id')
            since = int(since) if since not in (None, '') else None
        except ValueError:
            return jsonify({'error': 'since_seq_id must be an int'}), 400
        try:
            from yb_analysis.slm_sync import SlmSyncClient
            data = SlmSyncClient().get_diag(scan_id, since_seq_id=since)
            if data is None:
                return jsonify({'error': 'slm offline'}), 503
            return jsonify(data)
        except Exception as e:
            logger.debug('runs_diag_live passthrough failed: %s', e)
            return jsonify({'error': str(e)}), 503

    @server.route('/api/runs/<scan_id>/code')
    def _api_runs_code(scan_id):
        """Return the code-snapshot manifest for `scan_id`.

        Prefers `<scan_dir>/slm_code.json` if Phase 2 sync persisted
        it; otherwise passes through to the SLM PC's
        `/slm/runs/{scan_id}/code`.
        """
        return _runs_code_response(scan_id)

    # ---- Per-scan analysis (Phase 4) ---------------------------------
    @server.route('/api/runs/<scan_id>/analysis')
    def _api_runs_analysis(scan_id):
        """Lab-side per-scan analysis: survival, loading, diag aggregate,
        sweep description, code/grid sidecar pointers. JSON-safe.

        Query params:
          ``filter`` — JSON-encoded ``{axis_name: [allowed values]}``.
            Limits the analysis to scan points whose swept-param value
            matches one of the allowed values for each constrained
            axis. Cards downstream of the filter card (sweep / per-site
            / per-iteration) re-render with the filtered subset.

        Errors:
          400 — scan_id not a 14-digit string, or filter not valid JSON
          404 — scan_dir not found under DATA_DIR
          500 — analysis failed (logicals unpacking, h5 corruption, ...)
        """
        from yb_analysis.analysis.run_analysis import (
            analyze_scan, RunAnalysisError)
        filters = None
        raw_filter = request.args.get('filter')
        if raw_filter:
            try:
                filters = json.loads(raw_filter)
                if not isinstance(filters, dict):
                    raise ValueError('filter must be a JSON object')
            except (ValueError, json.JSONDecodeError) as ex:
                return jsonify({'error': f'invalid filter: {ex}'}), 400
        # Optional: refit discrimination infidelity from this run's own
        # intensities instead of the stored scan-start calibration
        # (non-destructive). Driven by the dashboard's "recompute" button.
        recompute = request.args.get('recompute_infidelity', '0') not in (
            '0', '', 'false', 'False')
        # "Re-analyze" button: drop the cached expensive results and recompute.
        force_recache = request.args.get('force_recache', '0') not in (
            '0', '', 'false', 'False')
        try:
            result = analyze_scan(scan_id, filters=filters,
                                  recompute_infidelity=recompute,
                                  force_recache=force_recache)
        except RunAnalysisError as ex:
            msg = str(ex).lower()
            if 'must be 14 digits' in msg or 'must pass scan_id' in msg:
                return jsonify({'error': str(ex)}), 400
            if 'could not find' in msg or 'not a directory' in msg:
                return jsonify({'error': str(ex)}), 404
            return jsonify({'error': str(ex)}), 500
        except Exception as ex:
            logging.exception('analyze_scan(%s) failed', scan_id)
            return jsonify({'error': str(ex)}), 500
        return jsonify(result)

    @server.route('/api/runs/<scan_id>/grid')
    def _api_runs_grid(scan_id):
        """Per-run grid sidecar (Phase 4).

        Sidecar-first: returns `<scan_dir>/slm_grid.json` if Phase 2/4
        sync persisted it. Otherwise passes through to the SLM PC's
        `/slm/runs/<scan_id>/grid_sidecar`. Returns 404 if neither.
        """
        return _runs_grid_response(scan_id)

    # ---- Sequence plotter: flattened .seq from a scan's sequence/ dir ----
    def _open_seq_folder():
        """Resolve a SequenceFolder from ``?scan_id=`` or ``?folder=``.

        Returns ``(folder, None)`` on success, or ``(None, (response, code))``
        — a ready-to-return error tuple — on failure.
        """
        from flask import request, jsonify
        from yb_analysis.sequence import manifest as _seqman
        scan_id = request.args.get('scan_id')
        folder = request.args.get('folder')
        if scan_id:
            base = _resolve_scan_dir(scan_id)
            if not base:
                return None, (jsonify({'error': 'bad scan_id: %s' % scan_id}), 400)
        elif folder:
            base = folder
        else:
            return None, (jsonify({'error': 'pass scan_id or folder'}), 400)
        sf = _seqman.SequenceFolder.open(base)
        if sf is None:
            return None, (jsonify({'error': 'no .seq files found', 'folder': base}), 404)
        return sf, None

    def _pick_sequence(dump, sel):
        """Choose a Sequence in a dump by name or seq_idx; default the first."""
        if not dump.sequences:
            return None
        if sel:
            for s in dump.sequences:
                if s.name == sel or str(s.seq_idx) == str(sel):
                    return s
        return dump.sequences[0]

    def _load_selected_seq():
        """Shared (folder -> file -> sequence) resolution for figure/params."""
        from flask import request, jsonify
        sf, err = _open_seq_folder()
        if err:
            return None, None, err
        fname = request.args.get('file')
        if not fname:
            files = sf.seq_files()
            if not files:
                return None, None, (jsonify({'error': 'no .seq files'}), 404)
            fname = files[0]
        try:
            dump = sf.load(fname)
        except (OSError, ValueError, FileNotFoundError) as ex:
            return None, None, (jsonify({'error': str(ex)}), 404)
        seq = _pick_sequence(dump, request.args.get('seq'))
        if seq is None:
            return None, None, (jsonify({'error': 'no sequence in file'}), 404)
        return sf, seq, None

    @server.route('/api/sequence/list')
    def _api_sequence_list():
        sf, err = _open_seq_folder()
        if err:
            return err
        return jsonify(_to_jsonable(sf.index()))

    @server.route('/api/sequence/figure')
    def _api_sequence_figure():
        from flask import request, Response
        import plotly.utils as _putils
        from yb_analysis.sequence import figure as _seqfig
        sf, seq, err = _load_selected_seq()
        if err:
            return err
        raw = request.args.get('chns', '')
        chns = [c for c in raw.split(',') if c]
        fig = _seqfig.build_sequence_figure(seq, chns)
        body = json.dumps(fig.to_dict(), cls=_putils.PlotlyJSONEncoder,
                          allow_nan=False, default=str)
        return Response(body, mimetype='application/json')

    def _seq_scan_base():
        """Scan folder (where ``data_<stamp>.json`` lives) from ?scan_id= or ?folder=."""
        from flask import request
        scan_id = request.args.get('scan_id')
        folder = request.args.get('folder')
        if scan_id:
            return _resolve_scan_dir(scan_id)
        if folder:
            return folder
        return None

    @server.route('/api/sequence/params')
    def _api_sequence_params():
        """Parameters card for one scan.

        Prefers the scan's ``.json`` sidecar (§12.2: ``expConfig`` baseline +
        ``base.params`` overrides + ``base.vars`` scanned axes -> the viewer's
        status-code tree) -- the always-available, engine-free source that works
        with or without a ``.seq``. Falls back to the ``.seq``-embedded params for a
        raw folder of ``.seq`` files that has no sidecar.
        """
        from yb_analysis.sequence import params_from_config as _pfc
        base = _seq_scan_base()
        built = _pfc.build_params_tree_for_folder(base) if base else None
        if built and built.get('has_params'):
            return jsonify(_to_jsonable({
                'params': built['params'],
                'has_params': True,
                'scanned_paths': built['scanned_paths'],
                'stats': built.get('stats'),
                'source': 'config',
            }))
        # Fallback: the params block embedded in the selected .seq file.
        sf, seq, err = _load_selected_seq()
        if err:
            return err
        scanned_paths = [a.get('path') for a in sf.scanned_axes() if a.get('path')]
        return jsonify(_to_jsonable({
            'seq_name': seq.name,
            'seq_idx': seq.seq_idx,
            'params': seq.params,
            'has_params': seq.params is not None,
            'scanned_paths': scanned_paths,
            'source': 'seq',
        }))

    @server.route('/api/sequence/xref')
    def _api_sequence_xref():
        """Param<->channel PROVENANCE for the loaded .seq (gated behind the engine build).

        Reads the ``sequence/xref.json`` provenance artifact (which param's SeqVal flows
        into which channel) for the selected ``file``. Returns ``{available,
        param_to_channels, channel_to_params}`` -- ``available`` is False (feature dormant)
        until the engine-side builder (reconstruction / B3) has written that artifact.
        """
        from flask import request
        from yb_analysis.sequence import xref as _xref
        sf, err = _open_seq_folder()
        if err:
            return err
        fname = request.args.get('file')
        if not fname:
            files = sf.seq_files()
            fname = files[0] if files else None
        return jsonify(_to_jsonable(_xref.load_xref(sf.dir, fname)))

    @server.route('/api/sequence/build_xref', methods=['POST'])
    def _api_sequence_build_xref():
        """Build a loaded scan's param<->channel ``xref.json`` in the BACKGROUND.

        For a scan that already has a ``.seq`` (live auto-dump) but no provenance yet:
        spawns ``pyctrl/tools/provenance_scan.py`` (the LIVE-lib, engine-free producer) as a
        detached subprocess and returns immediately, so the Sequence tab can show the ``.seq``
        first and light up the param<->channel affordance once the build lands (the JS polls
        ``/api/sequence/xref``). Idempotent: at most one build per scan dir runs at a time.
        Returns ``{ok, started}`` / ``{ok, available}`` (already built) / ``{ok:False, ...}``.
        """
        import os
        import json as _json
        import subprocess
        from flask import request, jsonify
        from yb_analysis.sequence import xref as _xref
        from yb_analysis.sequence import manifest as _seqman
        from yb_analysis.sequence.params_from_config import find_config_sidecar
        from yb_analysis import config as _cfg

        base = _seq_scan_base()
        if not base or not os.path.isdir(base):
            return jsonify({'ok': False, 'error': 'pass scan_id or folder'}), 400
        sf = _seqman.SequenceFolder.open(base)
        seq_dir = sf.dir if sf is not None else os.path.join(base, 'sequence')
        # ?force=1 rebuilds even when an artifact exists (e.g. to UPGRADE a pre-region
        # xref.json that lacks per-pulse maps). Without force, an existing artifact wins.
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'on')
        if not force and _xref.load_xref(seq_dir).get('available'):
            return jsonify({'ok': True, 'available': True})        # already built
        # Rebuilding the ScanGroup needs the self-contained descriptor in the sidecar.
        sidecar = find_config_sidecar(base)
        has_desc = False
        if sidecar:
            try:
                with open(sidecar, encoding='utf-8') as fh:
                    has_desc = bool(_json.load(fh).get('descriptor'))
            except (OSError, ValueError):
                has_desc = False
        if not has_desc:
            return jsonify({'ok': False, 'error': 'scan has no descriptor; cannot build xref'}), 200
        py = getattr(_cfg, 'PYCTRL_PYTHON', None)
        tool = os.path.join(getattr(_cfg, 'PYCTRL_CWD', '') or '', 'tools', 'provenance_scan.py')
        if not py or not os.path.exists(py) or not os.path.exists(tool):
            return jsonify({'ok': False, 'error': 'pyctrl python or provenance tool not found'}), 500
        key = os.path.abspath(base)
        with _XREF_BUILD_LOCK:
            running = _XREF_BUILDS.get(key)
            if running is not None and running.poll() is None:
                return jsonify({'ok': True, 'started': True, 'already': True})
            try:
                _XREF_BUILDS[key] = subprocess.Popen(
                    [py, tool, '--scan-dir', base],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as ex:  # noqa: BLE001
                return jsonify({'ok': False, 'error': 'spawn failed: %s' % ex}), 500
        return jsonify({'ok': True, 'started': True})

    @server.route('/api/sequence/reconstruct', methods=['POST'])
    def _api_sequence_reconstruct():
        """Regenerate a scan's missing .seq waveforms offline (engine-python subprocess).

        For a Reconstructable scan (code snapshot present, no .seq): runs
        ``pyctrl/tools/reconstruct_scan.py`` in the py3.8+libnacs venv
        (``use_dummy_device``) as a SEPARATE process -- it binds no port, drives no hardware,
        and never writes the ``runtime_state`` mmap (it ``set_global``s in its own pyseq).
        Deferred while a scan is running (CPU contention). Returns the driver's parsed result:
        ``{ok, n_seq, n_points, approximate, ...}`` or ``{deferred, reason}``.
        """
        import os
        import subprocess
        from flask import request, jsonify
        from yb_analysis import config as _cfg
        scan_id = request.args.get('scan_id')
        if not scan_id:
            return jsonify({'ok': False, 'error': 'pass scan_id'}), 400
        scan_dir = _resolve_scan_dir(scan_id)
        if not scan_dir or not os.path.isdir(scan_dir):
            return jsonify({'ok': False, 'error': 'bad scan_id: %s' % scan_id}), 400
        # Defer while a scan runs: the dummy subprocess would contend for CPU with the live
        # compile/eval. Best-effort -- allow if the queue can't be read.
        try:
            q = _read_queue_data()
            running = bool(q and q.get('running') and q['running'].get('id') is not None)
        except Exception:  # noqa: BLE001
            running = False
        if running:
            return jsonify({'ok': False, 'deferred': True,
                            'reason': 'a scan is running -- retry when idle'}), 200
        py = getattr(_cfg, 'PYCTRL_PYTHON', None)
        driver = os.path.join(getattr(_cfg, 'PYCTRL_CWD', '') or '',
                              'tools', 'reconstruct_scan.py')
        if not py or not os.path.exists(py) or not os.path.exists(driver):
            return jsonify({'ok': False,
                            'error': 'engine python or reconstruct driver not found'}), 500
        try:
            proc = subprocess.run([py, driver, '--scan-dir', scan_dir],
                                  capture_output=True, text=True, timeout=900)
        except Exception as ex:  # noqa: BLE001
            return jsonify({'ok': False, 'error': 'driver spawn failed: %s' % ex}), 500
        result = _parse_reconstruct_result(proc.stdout or '')
        if result is None:
            return jsonify({'ok': False, 'error': 'reconstruct driver produced no result',
                            'stderr': (proc.stderr or '')[-1500:]}), 500
        return jsonify(_to_jsonable(result)), (200 if result.get('ok') else 500)

    @server.route('/api/sequence/dump_toggle', methods=['GET', 'POST'])
    def _api_sequence_dump_toggle():
        """Read (GET) or set (POST ?on=1|0) the auto-dump toggle.

        The flag rides the pyctrl backend's mmap runtime-state store (offset 8);
        the runner reads it at scan start. Always returns ``{"on": <bool>}``.
        """
        from flask import request, jsonify
        from yb_analysis.sequence import dump_toggle
        if request.method == 'POST':
            raw = request.args.get('on')
            if raw is None:
                body = request.get_json(silent=True) or {}
                raw = body.get('on')
            on = str(raw).strip().lower() in ('1', 'true', 'yes', 'on')
            try:
                dump_toggle.set_save_sequence_dumps(on)
            except Exception as ex:  # noqa: BLE001
                return jsonify({'error': str(ex)}), 500
        return jsonify({'on': dump_toggle.get_save_sequence_dumps(False)})

    @server.route('/api/sequence/scans')
    def _api_sequence_scans():
        """Scans for the Sequence-tab picker -- the SAME set the Analysis runs
        picker shows (``runs_list.list_runs`` with meta), each flagged with
        whether it carries a ``sequence/`` dump.

        Scans WITHOUT a dump are returned too (``has_seq=False``) so the UI can
        GRAY them out instead of hiding them (parity with the Analysis picker).
        Newest first.
        """
        import os
        from flask import request, jsonify
        from yb_analysis.analysis.runs_list import list_runs
        from yb_analysis.sequence.manifest import find_sequence_dir
        since_days = request.args.get('since_days', type=int)
        max_count = request.args.get('max', type=int, default=500)
        scans = []
        for r in list_runs(since_days=since_days, max_count=max_count, with_meta=True):
            seq_dir = find_sequence_dir(r['scan_dir'])
            n = 0
            if seq_dir:
                try:
                    n = sum(1 for f in os.listdir(seq_dir) if f.endswith('.seq'))
                except OSError:
                    n = 0
            scans.append({
                'scan_id': r['scan_id'],
                'scan_dir': r['scan_dir'],
                'name': r.get('name'),
                'swept': r.get('swept'),
                'has_seq': n > 0,
                'n_seq': n,
                # Three-state picker (§12.4): has_seq -> Ready; else
                # (has_snapshot AND has_descriptor) -> Reconstructable; else
                # Unrecoverable. A snapshot WITHOUT a descriptor (scans predating
                # self-contained reconstruction) is NOT reconstructable.
                'has_snapshot': bool(r.get('has_snapshot')),
                'has_descriptor': bool(r.get('has_descriptor')),
            })
        return jsonify({'scans': scans})

    @server.route('/api/sequence/pick_folder', methods=['POST'])
    def _api_sequence_pick_folder():
        """Open a NATIVE folder picker on the lab PC and return the chosen path.

        Runs Tk's ``askdirectory`` in a short-lived CHILD PROCESS, so it never
        blocks this server's threads and never touches the main run_monitor
        process (whose Tk loop carries the safety-critical abort path). The
        dialog appears on the lab PC's desktop -- this is for local use. Returns
        ``{'path': ''}`` on cancel, ``{'error': ...}`` if Tk/display is absent.
        """
        import sys
        import subprocess
        from flask import jsonify
        code = (
            "import sys, tkinter as tk\n"
            "from tkinter import filedialog\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            "p = filedialog.askdirectory(title='Pick a folder of .seq files')\n"
            "r.destroy()\n"
            "sys.stdout.write(p or '')\n"
        )
        try:
            res = subprocess.run([sys.executable, '-c', code],
                                 capture_output=True, text=True, timeout=300)
        except Exception as ex:  # noqa: BLE001 - no display / Tk missing / timeout
            return jsonify({'error': str(ex)}), 500
        return jsonify({'path': (res.stdout or '').strip()})

    @server.route('/api/runs/<scan_id>/avg_image')
    def _api_runs_avg_image(scan_id):
        """Averaged camera image for single-image scans (Phase 5a).

        Returns the cached `<scan_dir>/avg_image.png` (computed on
        first request from /imgs in data_*.h5; ~10 s for a 300-shot
        scan, written-through to disk so subsequent requests are
        instant). Returns 404 when the scan has no /imgs or NumImages
        is not 1.
        """
        from yb_analysis.analysis.run_analysis import (
            ensure_avg_image_png, _scan_data_h5)
        from flask import send_file
        from pathlib import Path as _P
        # _resolve_scan_dir returns a string path (or None); wrap it
        # for the Path API we use below.
        sd_str = _resolve_scan_dir(scan_id)
        if sd_str is None:
            return jsonify({'error': f'scan_dir not found for {scan_id}'}), 404
        scan_dir = _P(sd_str)
        if not scan_dir.is_dir():
            return jsonify({'error': f'scan_dir not found for {scan_id}'}), 404
        if _scan_data_h5(scan_dir) is None:
            return jsonify({'error': 'no data_*.h5 in scan_dir'}), 404
        png_path = ensure_avg_image_png(scan_dir)
        if png_path is None or not png_path.is_file():
            return jsonify({'error': 'no /imgs dataset or compute failed'}), 404
        # Long cache lifetime — the average is fully determined by the
        # raw imgs, which never change for a completed scan.
        resp = send_file(str(png_path), mimetype='image/png')
        resp.headers['Cache-Control'] = 'public, max-age=86400'
        return resp

    @server.route('/api/runs/<scan_id>/shot_image')
    def _api_runs_shot_image(scan_id):
        """PNG of one camera frame from a single shot (per-iteration popup).

        Query params:
          shot        — 1-indexed shot number (matches
                        per_iteration.shot_index; = storage order in /imgs)
          frame       — 0-indexed frame within the shot (0 .. num_images-1)
          num_images  — frames per shot (from per_iteration.num_images)

        /imgs stores frames interleaved, so the row is
        ``(shot - 1) * num_images + frame``. The frame is min-max
        normalized to uint8. 404 when the scan/imgs/row is missing.
        """
        from flask import request, jsonify, Response
        from pathlib import Path as _P
        from yb_analysis.analysis.run_analysis import _scan_data_h5
        sd = _resolve_scan_dir(scan_id)
        if sd is None or not _P(sd).is_dir():
            return jsonify({'error': f'scan_dir not found for {scan_id}'}), 404
        h5 = _scan_data_h5(_P(sd))
        if h5 is None:
            return jsonify({'error': 'no data_*.h5 in scan_dir'}), 404
        try:
            shot = int(request.args.get('shot', '1'))
            frame = int(request.args.get('frame', '0'))
            num_images = int(request.args.get('num_images', '1'))
        except ValueError:
            return jsonify({'error': 'shot/frame/num_images must be ints'}), 400
        if shot < 1 or num_images < 1 or frame < 0 or frame >= num_images:
            return jsonify({'error': 'bad shot/frame/num_images'}), 400
        row = (shot - 1) * num_images + frame
        try:
            png = _shot_frame_png_bytes(str(h5), row)
        except Exception as ex:
            logger.warning('shot_image(%s shot=%s frame=%s) failed: %s',
                           scan_id, shot, frame, ex)
            return jsonify({'error': str(ex)}), 500
        if png is None:
            return jsonify({'error': 'frame out of range or no /imgs'}), 404
        # The raw frame never changes for a completed scan -> cache it.
        resp = Response(png, mimetype='image/png')
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp

    # ---- Programmatic scan submission (Phase 3) ------------------------
    # These accept POST so a one-shot curl works; flask's url_map will
    # complain on GET, which is the right error for "this is an action,
    # not a fetch".
    from flask import request

    @server.route('/api/queue/submit', methods=['POST'])
    def _api_queue_submit():
        """Submit a scan descriptor. Body: JSON conforming to
        `yb_analysis/scans/descriptor.schema.json`. Returns
        `{descriptor_id: N, kind: 'descriptor'}` on success, or
        `{error: ...}` with 400/503."""
        try:
            payload = request.get_json(force=True, silent=False) or {}
        except Exception as ex:
            return jsonify({'error': f'invalid JSON body: {ex}'}), 400
        try:
            from yb_analysis.scans.client import submit_scan
            from yb_analysis.scans.descriptor import DescriptorError
        except Exception as ex:
            return jsonify({'error': f'scans package import failed: {ex}'}), 500
        try:
            did = submit_scan(
                seq=payload.get('seq'),
                params=payload.get('params'),
                runp=payload.get('runp'),
                opts=payload.get('opts'),
                label=payload.get('label', ''))
        except DescriptorError as ex:
            return jsonify({'error': f'descriptor invalid: {ex}'}), 400
        except TimeoutError as ex:
            return jsonify({'error': f'runner unreachable: {ex}'}), 503
        except Exception as ex:
            return jsonify({'error': f'submit failed: {ex}'}), 500
        return jsonify({'descriptor_id': int(did), 'kind': 'descriptor'})

    @server.route('/api/queue/cancel/<int:entry_id>', methods=['POST'])
    def _api_queue_cancel(entry_id):
        """Cancel a queued job or descriptor by id (auto-detects kind)."""
        try:
            from yb_analysis.scans.client import cancel
            ok = cancel(int(entry_id))
        except TimeoutError as ex:
            return jsonify({'error': f'runner unreachable: {ex}'}), 503
        except Exception as ex:
            return jsonify({'error': f'cancel failed: {ex}'}), 500
        if ok:
            return jsonify({'ok': True, 'id': int(entry_id)})
        return jsonify({'ok': False, 'id': int(entry_id),
                        'error': 'not found or already running'}), 404

    @server.route('/api/queue/move/<int:entry_id>/<direction>',
                  methods=['POST'])
    def _api_queue_move(entry_id, direction):
        """Move a queued entry up or down (within its own kind)."""
        if direction not in ('up', 'down'):
            return jsonify({'error': "direction must be 'up' or 'down'"}), 400
        try:
            from yb_analysis.scans.client import move
            ok = move(int(entry_id), direction)
        except TimeoutError as ex:
            return jsonify({'error': f'runner unreachable: {ex}'}), 503
        except Exception as ex:
            return jsonify({'error': f'move failed: {ex}'}), 500
        if ok:
            return jsonify({'ok': True, 'id': int(entry_id),
                            'direction': direction})
        return jsonify({'ok': False, 'id': int(entry_id),
                        'error': 'cannot move (at edge or unknown id)'}), 400

    @server.route('/api/queue/requeue/<int:entry_id>', methods=['POST'])
    def _api_queue_requeue(entry_id):
        """Re-queue an existing entry: re-submit the original descriptor it
        was queued with, producing a NEW descriptor with byte-identical
        parameters. Works for any running/queued/history entry that still
        carries its descriptor.

        `?code=1` ALSO pins the new descriptor to the source run's captured
        code snapshot (reproducibility), so the pyctrl run loop replays the
        exact experiment source that ran originally. Returns
        `{descriptor_id: N, source_id: entry_id, with_code: bool}` or an error."""
        with_code = request.args.get('code', '0') in ('1', 'true', 'yes')
        try:
            from yb_analysis.scans.client import requeue
            from yb_analysis.scans.descriptor import DescriptorError
        except Exception as ex:
            return jsonify({'error': f'scans package import failed: {ex}'}), 500
        try:
            did = requeue(int(entry_id), with_code=with_code)
        except LookupError as ex:
            return jsonify({'error': str(ex)}), 404
        except DescriptorError as ex:
            return jsonify({'error': f'descriptor invalid: {ex}'}), 400
        except TimeoutError as ex:
            return jsonify({'error': f'runner unreachable: {ex}'}), 503
        except Exception as ex:
            logger.exception('requeue callback failed')
            return jsonify({'error': f'requeue failed: {ex}'}), 500
        return jsonify({'descriptor_id': int(did), 'source_id': int(entry_id),
                        'with_code': bool(with_code), 'kind': 'descriptor'})

    # ---- Phase 4: runs list / groups / seqs ---------------------------
    @server.route('/api/runs/list')
    def _api_runs_list():
        from yb_analysis.analysis.runs_list import list_runs
        try:
            since_days = request.args.get('since_days', type=int)
            max_count  = request.args.get('max', type=int, default=500)
            with_meta  = request.args.get('with_meta', '1') != '0'
            rows = list_runs(since_days=since_days, max_count=max_count,
                             with_meta=with_meta)
            return jsonify({'runs': rows, 'count': len(rows)})
        except Exception as ex:
            logger.exception('runs_list failed')
            return jsonify({'error': str(ex), 'runs': []}), 500

    @server.route('/api/runs/groups', methods=['GET', 'POST'])
    def _api_runs_groups():
        from yb_analysis.analysis import run_groups
        if request.method == 'POST':
            try:
                body = request.get_json(force=True, silent=True) or {}
                name = body.get('name') or ''
                gid = run_groups.create_group(name)
                g = run_groups.get_group(gid)
                return jsonify({'group_id': gid, 'group': g})
            except ValueError as ex:
                return jsonify({'error': str(ex)}), 400
            except Exception as ex:
                logger.exception('create_group failed')
                return jsonify({'error': str(ex)}), 500
        # GET
        return jsonify({'groups': run_groups.list_groups()})

    @server.route('/api/runs/groups/<group_id>', methods=['GET', 'DELETE'])
    def _api_runs_group(group_id):
        from yb_analysis.analysis import run_groups
        if request.method == 'DELETE':
            ok = run_groups.delete_group(group_id)
            return jsonify({'ok': ok, 'group_id': group_id}), (200 if ok else 404)
        g = run_groups.get_group(group_id)
        if g is None:
            return jsonify({'error': 'group not found'}), 404
        return jsonify(g)

    @server.route('/api/runs/groups/<group_id>/add/<scan_id>',
                  methods=['POST'])
    def _api_runs_group_add(group_id, scan_id):
        from yb_analysis.analysis import run_groups
        ok = run_groups.add_member(group_id, scan_id)
        return jsonify({'ok': ok, 'group_id': group_id, 'scan_id': scan_id}), \
            (200 if ok else 404)

    @server.route('/api/runs/groups/<group_id>/remove/<scan_id>',
                  methods=['POST'])
    def _api_runs_group_remove(group_id, scan_id):
        from yb_analysis.analysis import run_groups
        ok = run_groups.remove_member(group_id, scan_id)
        return jsonify({'ok': ok, 'group_id': group_id, 'scan_id': scan_id}), \
            (200 if ok else 404)

    @server.route('/api/runs/groups/<group_id>/analysis')
    def _api_runs_group_analysis(group_id):
        """Aggregate analysis across every member scan of a group.
        Returns the same shape as /api/runs/<id>/analysis, with the
        per-param survival / loading curves averaged across the
        members (un-weighted mean; future Phase: weighted by n_shots)."""
        from yb_analysis.analysis import run_groups
        from yb_analysis.analysis.run_analysis import analyze_scan, RunAnalysisError
        g = run_groups.get_group(group_id)
        if g is None:
            return jsonify({'error': 'group not found'}), 404
        members = g.get('members') or []
        if not members:
            return jsonify({'error': 'group is empty', 'group_id': group_id}), 400
        results = []
        errors = []
        for m in members:
            try:
                results.append(analyze_scan(m['scan_id']))
            except RunAnalysisError as ex:
                errors.append({'scan_id': m['scan_id'], 'error': str(ex)})
            except Exception as ex:
                errors.append({'scan_id': m['scan_id'], 'error': str(ex)})
        if not results:
            return jsonify({
                'error': 'no member scan analyzed successfully',
                'errors': errors,
            }), 500
        return jsonify(_aggregate_group_analysis(results, group=g, errors=errors))

    # ---- Phase 4: seq catalog -----------------------------------------
    @server.route('/api/seqs/list')
    def _api_seqs_list():
        from yb_analysis.scans.seq_catalog import list_seqs, cache_version
        return jsonify({
            'seqs': list_seqs(summary=True),
            'cache_version': cache_version(),
        })

    @server.route('/api/seqs/<name>')
    def _api_seqs_get(name):
        from yb_analysis.scans.seq_catalog import get_seq
        entry = get_seq(name)
        if entry is None:
            return jsonify({'error': f'seq {name} not found'}), 404
        return jsonify(entry)

    @server.route('/api/seqs/refresh', methods=['POST'])
    def _api_seqs_refresh():
        from yb_analysis.scans.seq_catalog import invalidate_cache, cache_version
        invalidate_cache()
        return jsonify({'ok': True, 'cache_version': cache_version()})

    # ---- Phase 4: extra SLM proxies ----------------------------------
    @server.route('/api/slm/clients')
    def _api_slm_clients():
        return _slm_passthrough('/clients', cache_key='clients',
                                slow_poll=True)

    @server.route('/api/slm/logs')
    def _api_slm_logs():
        # SLM PC exposes /logs (not /logs/tail). Pass through `lines` arg.
        n = request.args.get('lines', '200')
        return _slm_passthrough(f'/logs?lines={n}', cache_key='logs',
                                slow_poll=True)

    @server.route('/api/slm/server_info')
    def _api_slm_server_info():
        return _slm_passthrough('/server_info', cache_key='server_info',
                                slow_poll=True)

    @server.route('/api/slm/gpu')
    def _api_slm_gpu():
        return _slm_passthrough('/gpu/info', cache_key='gpu', slow_poll=True)

    # ---- Phase 4b: live image as data URI for the new HTML dashboard ----
    # The /api/snapshot path strips _img_data_uri (it's heavy ~MB); the
    # HTML dashboard wants the PNG to render as an <img>. These endpoints
    # surface the cached data URI from yb_dash_data.pkl directly.

    @server.route('/api/live/image1')
    def _api_live_image1():
        return _live_image_response('_img_data_uri', '_img_shape',
                                    '_img_vlo', '_img_vhi')

    @server.route('/api/live/image2')
    def _api_live_image2():
        return _live_image_response('_img2_data_uri', '_img2_shape',
                                    '_img2_vlo', '_img2_vhi')

    @server.route('/api/live/image_mid')
    def _api_live_image_mid():
        return _live_image_response('_img_mid_data_uri', '_img_mid_shape',
                                    '_img_mid_vlo', '_img_mid_vhi')

    @server.route('/api/live/figures')
    def _api_live_figures():
        """Return ONE or ALL live-tab Plotly figures as JSON.

        Calls the same ``_fig_*`` functions Dash uses. plotly.py 6.x's
        ``to_plotly_json()`` emits typed-array blobs (``{bdata, dtype}``)
        instead of plain arrays — we decode them HERE to plain lists +
        nulls so the wire format works with any Plotly.js version. JSON
        encoded via ``PlotlyJSONEncoder`` so any leftover numpy / NaN
        / data URIs get the same treatment Dash gives them internally.
        """
        from flask import Response, jsonify
        import plotly.utils as _putils
        d = _read_data() or {}
        which = request.args.get('which', '').lower()
        try:
            marker_size = int(request.args.get('marker_size', 12))
        except ValueError:
            marker_size = 12
        cbar_scale = request.args.get('cbar_scale', '01')

        # Site-resolved histogram: ?which=site&site=N (1-indexed).
        try:
            site_idx = int(request.args.get('site', '1')) - 1
            if site_idx < 0:
                site_idx = 0
        except ValueError:
            site_idx = 0

        def _fig_for(name):
            try:
                if name == 'array':
                    return _fig_array(d)
                if name == 'array_mid':
                    return _fig_array(d, img_key='_img_mid_data_uri',
                                      shape_key='_img_mid_shape',
                                      vlo_key='_img_mid_vlo',
                                      vhi_key='_img_mid_vhi',
                                      logicals_key='logicals_mid',
                                      grid_key='grid_locations',
                                      title='Tweezer Array (middle)')
                if name == 'array2':
                    return _fig_array(d, img_key='_img2_data_uri',
                                      shape_key='_img2_shape',
                                      vlo_key='_img2_vlo', vhi_key='_img2_vhi',
                                      # img2 panel ALWAYS shows the final frame's
                                      # own logicals (_display_logicals2 is set for
                                      # every is_last frame, two-array or not). The
                                      # old `else 'logicals'` fell back to frame-0's
                                      # logicals in single-array NumImages=2 mode,
                                      # making img2 render identically to img1.
                                      logicals_key='logicals2',
                                      grid_key=('grid_locations_img2'
                                          if d.get('is_two_array') else 'grid_locations'),
                                      title='Tweezer Array (img 2)')
                if name == 'intens':   return _fig_intens(d)
                if name == 'loadlive': return _fig_loading_live(d)
                if name == 'load':     return _fig_loading(d, marker_size=marker_size)
                if name == 'infid':    return _fig_infid(d, marker_size=marker_size)
                if name == 'shift':    return _fig_shift(d)
                if name == 'scan':     return _fig_scan_curve(d, cbar_scale=cbar_scale)
                if name == 'avghist':  return _fig_avghist(d)
                if name in ('rep0', 'rep1', 'rep2', 'rep3'):
                    figs = _figs_reps(d)
                    idx = int(name[3])
                    return figs[idx] if idx < len(figs) else _waiting('Site Hist')
                if name == 'site':
                    fig, _info = _fig_site(d, site_idx)
                    return fig
            except Exception as ex:
                logger.exception('_fig_for %s failed', name)
                return _waiting(name, message=f'render error: {ex}')
            return None

        def _figdict(fig):
            if fig is None:
                return None
            try:
                raw = fig.to_plotly_json()
            except Exception:
                raw = {'data': list(getattr(fig, 'data', [])),
                       'layout': getattr(fig, 'layout', {})}
            return _decode_plotly_bdata(raw)

        if which:
            fig = _fig_for(which)
            if fig is None:
                return jsonify({'error': f'unknown figure name: {which}'}), 400
            body = json.dumps(_figdict(fig), cls=_putils.PlotlyJSONEncoder,
                              allow_nan=False, default=str)
            return Response(body, mimetype='application/json')

        names = ['array', 'array_mid', 'array2', 'intens', 'loadlive',
                 'load', 'infid', 'shift', 'scan', 'avghist',
                 'rep0', 'rep1', 'rep2', 'rep3']
        out = {n: _figdict(_fig_for(n)) for n in names}
        body = json.dumps({'figures': out}, cls=_putils.PlotlyJSONEncoder,
                          allow_nan=False, default=str)
        return Response(body, mimetype='application/json')

    # ====================================================================
    # CONTROL endpoints (Phase 5.5 Track A) — the dashboard mirrors the
    # Tkinter control panel so an operator on another machine has the same
    # authority. Pause/Start/Abort write the MATLAB MemoryMap directly (the
    # file is local to this PC). dummy_mode / init_dir / restart_* require
    # the main run_monitor process, so they're spooled to it via
    # yb_analysis.control.web_control and executed by ControlPanel.
    #
    # Safety: a remote-exposure gate (loopback always; tailscale in 'auto';
    # other LAN only when explicitly enabled) plus single-use confirmation
    # tokens for the two destructive ops (Abort, Restart All).
    # ====================================================================
    from yb_analysis.control import memmap_signal as _mm
    from yb_analysis.control import web_control as _wc

    def _current_backend():
        """Active backend for control routing, from YB_BACKEND — which
        run_monitor sets at dashboard spawn and is FIXED for this subprocess's
        lifetime (a backend switch restarts the whole monitor). Using the env,
        not the published status file, avoids a stale-status race where a fresh
        pyctrl session's dashboard could read the *previous* MATLAB session's
        status and wrongly poke a stale memmap. Unset defaults to 'matlab'
        (legacy / standalone dashboard); an unexpected value is logged and
        treated as non-matlab so routing fails CLOSED to ZMQ (never memmap)."""
        b = os.environ.get('YB_BACKEND', 'matlab')
        if b not in ('matlab', 'pyctrl'):
            logger.warning('YB_BACKEND=%r is not matlab|pyctrl; routing '
                           'control via ZMQ (no memmap) to be safe', b)
        return b

    def _backend_uses_memmap():
        """True iff the live backend is MATLAB (the only one with a local
        MemoryMap). Strict equality => any other value (incl. a misconfigured
        one) routes via ZMQ, never the local memmap."""
        return _current_backend() == 'matlab'

    _LOOPBACK = {'127.0.0.1', '::1', 'localhost', None, ''}

    def _is_tailscale(addr):
        # Tailscale CGNAT range 100.64.0.0/10.
        try:
            import ipaddress
            return ipaddress.ip_address(addr) in ipaddress.ip_network(
                '100.64.0.0/10')
        except Exception:
            return False

    def _controls_allowed():
        """Exposure policy. YB_DASH_REMOTE_CONTROLS = auto|on|off.

        Loopback is always allowed. 'auto' (default) also allows the
        tailnet; 'on' allows any remote; 'off' allows loopback only.
        """
        from flask import request
        addr = request.remote_addr
        if addr in _LOOPBACK:
            return True
        policy = os.environ.get('YB_DASH_REMOTE_CONTROLS', 'auto').lower()
        if policy == 'on':
            return True
        if policy == 'off':
            return False
        return _is_tailscale(addr)

    def _gate():
        """Return a 403 Response when controls aren't allowed, else None."""
        from flask import jsonify
        if _controls_allowed():
            return None
        resp = jsonify({'error': 'remote controls disabled on this interface'})
        resp.status_code = 403
        return resp

    # --- single-use confirmation tokens for destructive ops -------------
    # In-process (this Dash subprocess). The client requests a token, then
    # POSTs it back within the TTL; the server consumes it once. Pairs with
    # the click-and-hold UX but is independently enforced server-side.
    _CONFIRM_TTL_S = 30.0
    _confirm_tokens = {}     # token -> (issued_at, action)
    _confirm_counter = [0]

    def _issue_confirm_token(action):
        _confirm_counter[0] += 1
        # Unique, non-guessable-enough for a loopback safety interlock.
        tok = '%s-%d-%d' % (action, _confirm_counter[0], int(time.time() * 1000))
        _confirm_tokens[tok] = (time.monotonic(), action)
        # Opportunistic GC of stale tokens.
        now = time.monotonic()
        for k, (t0, _a) in list(_confirm_tokens.items()):
            if now - t0 > _CONFIRM_TTL_S:
                _confirm_tokens.pop(k, None)
        return tok

    def _consume_confirm_token(tok, action):
        rec = _confirm_tokens.get(tok)
        if rec is None:
            return False
        issued, act = rec
        _confirm_tokens.pop(tok, None)     # single use
        if act != action:
            return False
        return (time.monotonic() - issued) <= _CONFIRM_TTL_S

    @server.route('/api/control/status')
    def _api_control_status():
        # Full-fidelity control state published by the main process
        # (dummy mode string, last-seq meta, scan/seq/state) + whether
        # remote controls are allowed on this interface (drives the
        # sidebar's enabled/disabled rendering).
        from flask import jsonify
        st = _wc.read_status() or {}
        st['controls_allowed'] = _controls_allowed()
        return jsonify(st)

    @server.route('/api/control/confirm_token')
    def _api_control_confirm_token():
        from flask import jsonify, request
        g = _gate()
        if g is not None:
            return g
        action = (request.args.get('action') or '').lower()
        if action not in ('abort', 'restart_all', 'set_backend'):
            return jsonify(
                {'error': 'action must be abort|restart_all|set_backend'}), 400
        return jsonify({'token': _issue_confirm_token(action),
                        'ttl_s': _CONFIRM_TTL_S})

    @server.route('/api/control/pause', methods=['POST'])
    def _api_control_pause():
        from flask import jsonify
        g = _gate()
        if g is not None:
            return g
        if _backend_uses_memmap():
            # Pass the backend explicitly so safety doesn't rest on the guard
            # alone — signal_pause() with a non-matlab backend is a no-op even
            # if this branch were ever reached by mistake (defense in depth).
            wrote = _mm.signal_pause(_current_backend())
            return jsonify({'ok': True,
                            'via': 'memmap' if wrote else 'unavailable',
                            'memmap_present': wrote})
        # pyctrl: no local memmap — route to the main process's ZMQ client.
        _wc.enqueue('pause')
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/start', methods=['POST'])
    def _api_control_start():
        from flask import jsonify
        g = _gate()
        if g is not None:
            return g
        if _backend_uses_memmap():
            wrote = _mm.signal_start(_current_backend())
            return jsonify({'ok': True,
                            'via': 'memmap' if wrote else 'unavailable',
                            'memmap_present': wrote})
        _wc.enqueue('start')
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/abort', methods=['POST'])
    def _api_control_abort():
        from flask import jsonify, request
        g = _gate()
        if g is not None:
            return g
        # Token is consumed BEFORE the backend branch so BOTH paths require it.
        tok = request.args.get('confirm') or (request.get_json(silent=True)
                                              or {}).get('confirm')
        if not _consume_confirm_token(tok or '', 'abort'):
            return jsonify({'error': 'abort requires a valid confirm token'}), 400
        if _backend_uses_memmap():
            wrote = _mm.signal_abort(_current_backend())
            return jsonify({'ok': True,
                            'via': 'memmap' if wrote else 'unavailable',
                            'memmap_present': wrote})
        _wc.enqueue('abort')
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/dummy_mode', methods=['POST'])
    def _api_control_dummy_mode():
        from flask import jsonify, request
        g = _gate()
        if g is not None:
            return g
        body = request.get_json(silent=True) or {}
        mode = (body.get('mode') or '').lower()
        if mode not in ('off', 'default', 'last'):
            return jsonify({'error': 'mode must be off|default|last'}), 400
        _wc.enqueue('dummy_mode', mode=mode)
        return jsonify({'ok': True, 'mode': mode, 'via': 'run_monitor'})

    @server.route('/api/control/init_dir', methods=['POST'])
    def _api_control_init_dir():
        from flask import jsonify, request
        g = _gate()
        if g is not None:
            return g
        body = request.get_json(silent=True) or {}
        path = body.get('path') or ''
        if not path or not os.path.isdir(path):
            return jsonify({'error': 'path must be an existing directory'}), 400
        _wc.enqueue('init_dir', path=path)
        return jsonify({'ok': True, 'path': path, 'via': 'run_monitor'})

    @server.route('/api/control/restart_dash', methods=['POST'])
    def _api_control_restart_dash():
        from flask import jsonify
        g = _gate()
        if g is not None:
            return g
        _wc.enqueue('restart_dash')
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/restart_all', methods=['POST'])
    def _api_control_restart_all():
        from flask import jsonify, request
        g = _gate()
        if g is not None:
            return g
        tok = request.args.get('confirm') or (request.get_json(silent=True)
                                              or {}).get('confirm')
        if not _consume_confirm_token(tok or '', 'restart_all'):
            return jsonify(
                {'error': 'restart_all requires a valid confirm token'}), 400
        _wc.enqueue('restart_all')
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/set_backend', methods=['POST'])
    def _api_control_set_backend():
        from flask import jsonify, request
        g = _gate()
        if g is not None:
            return g
        body = request.get_json(silent=True) or {}
        target = (request.args.get('target') or body.get('target') or '').lower()
        if target not in ('matlab', 'pyctrl'):
            return jsonify({'error': 'target must be matlab|pyctrl'}), 400
        tok = request.args.get('confirm') or body.get('confirm')
        if not _consume_confirm_token(tok or '', 'set_backend'):
            return jsonify(
                {'error': 'set_backend requires a valid confirm token'}), 400
        _wc.enqueue('set_backend', target=target)
        return jsonify({'ok': True, 'target': target, 'via': 'run_monitor'})

    @server.route('/api/control/downsample', methods=['POST'])
    def _api_control_downsample():
        # Live-image downsample toggle: render preference, not a runner
        # action. Writes the browser->main reverse-channel control file the
        # main process reads when encoding the next frame. No exposure gate
        # (purely cosmetic, no authority over the experiment).
        from flask import jsonify, request
        body = request.get_json(silent=True) or {}
        on = bool(body.get('on', True))
        _write_control({'downsample': on})
        return jsonify({'ok': True, 'downsample': on})

    # ---- Camera control mirror (Phase 5.5) ----------------------------
    # Mirror of the Tkinter CameraPane. Status is published by the main
    # process (CameraPane._publish_web_status); connect/disconnect/apply
    # are spooled to it via web_control because the dashboard subprocess
    # has no ZMQ socket to the runner. Same exposure gate as the other
    # control ops — camera reconfig drops frames, so it's privileged.

    @server.route('/api/control/camera/status')
    def _api_control_camera_status():
        from flask import jsonify
        st = _wc.read_camera_status() or {}
        st['controls_allowed'] = _controls_allowed()
        return jsonify(st)

    def _parse_camera_body():
        """Pull (roi, exposure) from the request JSON, validating shape.

        Returns (roi, exposure, error_response). roi is a 4-int list or
        None; exposure is a float or None. error_response is a (json, code)
        tuple when validation fails, else None."""
        from flask import jsonify, request
        body = request.get_json(silent=True) or {}
        roi = body.get('roi')
        exposure = body.get('exposure')
        if roi is not None:
            try:
                roi = [int(v) for v in roi]
                if len(roi) != 4:
                    raise ValueError
            except (TypeError, ValueError):
                return None, None, (jsonify(
                    {'error': 'roi must be [x, y, w, h] integers'}), 400)
        if exposure is not None:
            try:
                exposure = float(exposure)
                if not (exposure > 0):
                    raise ValueError
            except (TypeError, ValueError):
                return None, None, (jsonify(
                    {'error': 'exposure must be a positive number (s)'}), 400)
        return roi, exposure, None

    @server.route('/api/control/camera/connect', methods=['POST'])
    def _api_control_camera_connect():
        from flask import jsonify
        g = _gate()
        if g is not None:
            return g
        roi, exposure, err = _parse_camera_body()
        if err is not None:
            return err
        if roi is None or exposure is None:
            return jsonify({'error': 'connect requires roi + exposure'}), 400
        _wc.enqueue('camera_connect', roi=roi, exposure=exposure)
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/camera/apply', methods=['POST'])
    def _api_control_camera_apply():
        from flask import jsonify
        g = _gate()
        if g is not None:
            return g
        roi, exposure, err = _parse_camera_body()
        if err is not None:
            return err
        if roi is None or exposure is None:
            return jsonify({'error': 'apply requires roi + exposure'}), 400
        _wc.enqueue('camera_apply', roi=roi, exposure=exposure)
        return jsonify({'ok': True, 'via': 'run_monitor'})

    @server.route('/api/control/camera/disconnect', methods=['POST'])
    def _api_control_camera_disconnect():
        from flask import jsonify
        g = _gate()
        if g is not None:
            return g
        _wc.enqueue('camera_disconnect')
        return jsonify({'ok': True, 'via': 'run_monitor'})


def _aggregate_focus_metrics(results):
    """Combine seq-specific focus metrics across group members.

    Aligns members by swept-VALUE (not index), so runs with different /
    overlapping defocus sweeps still merge where their x values match. Each
    metric is averaged across members weighted by that point's detected spot
    count (more spots -> more weight); n_spots are summed. Returns None when
    no member has focus metrics. This is why "combine runs" now actually
    pools the defocus optimisation curve across repeats."""
    members = [r.get('seq_specific') for r in results
               if isinstance(r.get('seq_specific'), dict)
               and r['seq_specific'].get('type') == 'focus_metrics']
    if not members:
        return None

    def _key(v):
        return round(float(v), 6)

    xmap = {}
    for ss in members:
        for v in ss.get('x') or []:
            xmap.setdefault(_key(v), float(v))
    xkeys = sorted(xmap)
    xs = [xmap[k] for k in xkeys]

    meta = {}
    for ss in members:
        for mk, m in (ss.get('metrics') or {}).items():
            meta.setdefault(mk, {'label': m.get('label'), 'unit': m.get('unit'),
                                 'higher_better': m.get('higher_better', True)})

    metrics_out = {}
    for mk, mm in meta.items():
        wsum = {k: 0.0 for k in xkeys}
        wtot = {k: 0.0 for k in xkeys}
        for ss in members:
            mx = ss.get('x') or []
            vals = (ss.get('metrics') or {}).get(mk, {}).get('values') or []
            nsp = ss.get('n_spots') or []
            for i, xv in enumerate(mx):
                k = _key(xv)
                if k not in wsum or i >= len(vals) or vals[i] is None:
                    continue
                w = float(nsp[i]) if (i < len(nsp) and nsp[i]) else 1.0
                wsum[k] += w * float(vals[i])
                wtot[k] += w
        metrics_out[mk] = {
            'values': [(wsum[k] / wtot[k]) if wtot[k] > 0 else None
                       for k in xkeys],
            'label': mm['label'], 'unit': mm['unit'],
            'higher_better': mm['higher_better'],
        }

    nsp_out = []
    for k in xkeys:
        tot = 0
        for ss in members:
            mx = ss.get('x') or []
            nsp = ss.get('n_spots') or []
            for i, xv in enumerate(mx):
                if _key(xv) == k and i < len(nsp) and nsp[i]:
                    tot += int(nsp[i])
        nsp_out.append(tot)

    return {
        'type': 'focus_metrics', 'source': 'images (combined)',
        'calibration_free': True, 'x': xs,
        'x_label': members[0].get('x_label') or 'param',
        'n_spots': nsp_out, 'metrics': metrics_out, 'n_members': len(members),
    }


def _aggregate_group_analysis(results, *, group, errors):
    """Combine a list of per-scan analysis dicts into a single dict.

    Strategy: take the param-axis-aligned average of survival_mean and
    loading_rate, propagating Bessel-corrected SEM. Falls back to the
    longest individual scan's sweep description when member sweeps
    don't match (more sophisticated joining could be added in
    Phase 5).
    """
    import math
    # Find the largest n_params across members and use that scan as the
    # sweep reference.
    primary = max(results, key=lambda r: r.get('n_params', 0) or 0)
    n_params = primary.get('n_params', 0) or 0

    def _stack(field):
        rows = []
        for r in results:
            vs = (r.get('summary') or {}).get(field) or []
            if len(vs) == n_params and any(v is not None for v in vs):
                rows.append(vs)
        return rows

    def _mean(rows):
        if not rows:
            return [None] * n_params
        out = []
        for j in range(n_params):
            xs = [row[j] for row in rows if row[j] is not None]
            out.append(sum(xs) / len(xs) if xs else None)
        return out

    def _sem(rows):
        if not rows:
            return [None] * n_params
        out = []
        for j in range(n_params):
            xs = [row[j] for row in rows if row[j] is not None]
            if len(xs) < 2:
                out.append(None)
                continue
            m = sum(xs) / len(xs)
            v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
            out.append(math.sqrt(v / len(xs)))
        return out

    surv_rows = _stack('survival_mean')
    load_rows = _stack('loading_rate')
    aggregated = {
        'scan_id': f'group:{group.get("id", "")}',
        'scan_dir': None,
        'scan_name': group.get('name'),
        'scan_filename': None,
        'n_params': n_params,
        'n_shots': sum((r.get('n_shots') or 0) for r in results),
        'sweep': primary.get('sweep'),
        'summary': {
            'survival_mean':     _mean(surv_rows),
            'survival_sem':      _sem(surv_rows),
            'loading_rate':      _mean(load_rows),
            'loading_rate_sem':  _sem(load_rows),
            'loss_mean':         [None] * n_params,
            'loss_sem':          [None] * n_params,
        },
        'per_site': None,
        'diag_aggregate': None,
        'code': {'present': False, 'n_files': 0},
        'grid': {'present': False, 'n_sites': 0},
        'round1': None,
        'survival_vs_distance': None,
        'survival_vs_distance_per_step': None,
        'per_shot_extra': None,
        # Seq-specific focus metrics pooled across members (the defocus
        # optimisation curve combined over repeats); None when no member
        # has them.
        'seq_specific': _aggregate_focus_metrics(results),
        # Group metadata: useful to the dashboard.
        'group': {
            'id': group.get('id'),
            'name': group.get('name'),
            'n_members': len(group.get('members') or []),
            'analyzed_members': [r.get('scan_id') for r in results],
            'failed_members': errors,
        },
    }
    return aggregated


def _decode_plotly_bdata(obj):
    """Recursively replace Plotly's binary-encoded array blobs with plain lists.

    Plotly.py 6.x emits typed arrays as ``{"bdata": "<b64>", "dtype": "<np>"}``
    inside the figure JSON for compactness. Plotly.js < 2.35 doesn't know
    how to decode those (the result renders as a blank plot). To stay
    compatible with older Plotly.js, walk the dict/list tree and inflate
    each bdata blob back to a Python list before jsonifying.
    """
    import base64
    if isinstance(obj, dict):
        # bdata-encoded array: {"bdata": "...", "dtype": "..."} (sometimes
        # with an optional "shape" key for multi-dim arrays).
        if 'bdata' in obj and 'dtype' in obj:
            try:
                buf = base64.b64decode(obj['bdata'])
                arr = np.frombuffer(buf, dtype=obj['dtype'])
                shape = obj.get('shape')
                if shape:
                    # Plotly emits shape as a string like "3,4" sometimes.
                    if isinstance(shape, str):
                        shape = [int(x) for x in shape.split(',') if x]
                    arr = arr.reshape(shape)
                # NaN/Inf are not JSON-valid; coerce to None.
                if arr.dtype.kind == 'f':
                    arr = np.where(np.isfinite(arr), arr, None).tolist()
                else:
                    arr = arr.tolist()
                return arr
            except Exception:
                return obj
        return {k: _decode_plotly_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_plotly_bdata(v) for v in obj]
    # NaN/Inf are not JSON-valid and json.dumps(allow_nan=False) raises on
    # them. The bdata branch above already coerces non-finite values inside
    # typed-array blobs, but figures often carry plain Python-float lists
    # (e.g. marker.color built via ndarray.tolist()), which Plotly emits
    # verbatim. Coerce those to None here so the whole tree is JSON-safe.
    # np.float64 is a subclass of float; np.float32 is not, hence the extra
    # np.floating check.
    if isinstance(obj, (float, np.floating)):
        return obj if math.isfinite(obj) else None
    return obj


def _live_image_response(uri_key, shape_key, vlo_key, vhi_key):
    """Serve a cached PNG image from yb_dash_data.pkl.

    Decodes the data URI back to raw PNG bytes and returns them with
    ``Content-Type: image/png`` so the dashboard can just
    ``<img src=...>`` against this endpoint -- no JSON parsing, no
    multi-megabyte base64 on every poll.

    Set ``?json=1`` for the legacy ``{data_uri, shape, vlo, vhi}`` JSON
    shape (still used by any caller that wants the percentile clip).
    """
    from flask import jsonify, Response, request
    import base64
    d = _read_data() or {}
    uri = d.get(uri_key)
    if uri is None:
        return jsonify({'error': 'no image yet'}), 404
    if request.args.get('json'):
        shape = d.get(shape_key)
        if shape is not None:
            try:
                shape = list(shape)
            except Exception:
                shape = None
        return jsonify({
            'data_uri': uri,
            'shape':    shape,
            'vlo':      d.get(vlo_key),
            'vhi':      d.get(vhi_key),
        })
    # Strip "data:image/png;base64," prefix and decode to raw PNG.
    if uri.startswith('data:image/png;base64,'):
        png = base64.b64decode(uri[len('data:image/png;base64,'):])
    else:
        return jsonify({'error': 'unexpected data URI format'}), 500
    return Response(png, mimetype='image/png', headers={
        # No browser caching -- the URL doesn't change but the bytes do.
        'Cache-Control': 'no-store, no-cache, must-revalidate',
    })


def _slm_passthrough(slm_path, *, cache_key=None, slow_poll=False):
    """Generic SLM passthrough used by Phase 4 extra proxies.

    Hits the SLM PC via the existing SlmSyncClient (Tailscale, retry on
    503, etc.). Returns JSON verbatim, or `{error:...}` with 503/404.
    slow_poll: hint for the dashboard JS not to spam this endpoint.
    """
    from flask import jsonify
    from yb_analysis.slm_sync import SlmSyncClient
    try:
        client = SlmSyncClient()
        # Use the private session/get pattern -- the typed accessors
        # don't cover every SLM path. The slow_poll flag is a doc hint;
        # we don't cache server-side (the SLM caches its own state).
        r = client._get(slm_path)   # pylint: disable=protected-access
        if r.status_code == 404:
            return jsonify({'error': f'SLM endpoint {slm_path} not found'}), 404
        if r.status_code == 503:
            return jsonify({'error': 'SLM server busy', 'slm_path': slm_path}), 503
        r.raise_for_status()
        try:
            return jsonify(r.json())
        except ValueError:
            # Non-JSON response (logs/tail may stream plain text). Wrap.
            return jsonify({'text': r.text})
    except Exception as ex:
        logger.warning('slm_passthrough %s failed: %s', slm_path, ex)
        return jsonify({'error': str(ex), 'slm_path': slm_path}), 503


# ===========================================================================
# Phase 4 main HTML dashboard (served at /)
# ===========================================================================

def _register_main_html_routes(server):
    """Mount the new SLM-styled HTML dashboard at / (and serve its
    static assets). The old Dash app lives at /old/ via
    `url_base_pathname='/old/'`."""
    from flask import send_from_directory, send_file
    import os as _os
    import time as _time

    here = _os.path.dirname(_os.path.abspath(__file__))
    static_dir = _os.path.join(here, 'static')
    templates_dir = _os.path.join(here, 'templates')

    # Locate the bundled plotly.min.js (ships with plotly-python). Serving
    # locally avoids the silent-bail when a CDN is blocked by lab network
    # policy or unreachable -- in that case pollLiveFigures() returns at
    # `if (!window.Plotly) return;` and every live-tab div stays empty
    # (renders as the card's dark background -> "completely black").
    try:
        import plotly as _plotly
        _plotly_js_path = _os.path.join(
            _os.path.dirname(_plotly.__file__),
            'package_data', 'plotly.min.js')
        if not _os.path.exists(_plotly_js_path):
            _plotly_js_path = None
    except Exception:
        _plotly_js_path = None

    @server.route('/static/dashboard/<path:fname>')
    def _serve_main_static(fname):
        return send_from_directory(static_dir, fname)

    @server.route('/vendor/plotly.min.js')
    def _serve_plotly_js():
        if _plotly_js_path is None:
            return ('plotly.js bundle not found', 500)
        return send_file(_plotly_js_path, mimetype='application/javascript')

    @server.route('/')
    def _serve_main_html():
        # Use Flask's render_template from the templates/ dir we ship.
        # Lazy import jinja so the test fixture doesn't pull it before
        # the route is hit.
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from yb_analysis import config as yb_cfg
        env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(['html']),
        )
        tmpl = env.get_template('main.html')
        # Cache-bust static URLs with a per-process timestamp so a
        # browser that's been holding the dashboard tab open while we
        # iterate doesn't end up running stale dashboard.js / dashboard.css.
        return tmpl.render(
            static_url='/static/dashboard/',
            slm_url=yb_cfg.SLM_URL,
            cache_bust=str(int(_time.time())),
            plotly_local_url='/vendor/plotly.min.js',
        )


def _dash_main(host, port, data_file, parent_pid=None):
    """Entry point for the Dash subprocess.

    parent_pid: if provided, the child spawns a watchdog thread that
    polls the parent's liveness every 3 s. When the parent disappears
    (Ctrl+C with no atexit, terminal close, taskkill, segfault, ...),
    the watchdog calls ``os._exit(0)`` to release port 8050 immediately
    instead of becoming a zombie that the operator has to hand-kill.
    """
    # Reconfigure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [dash] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    if parent_pid is not None:
        _start_parent_watchdog(parent_pid)
    app = _build_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Parent watchdog (subprocess-side)
# ---------------------------------------------------------------------------

def _is_pid_alive(pid):
    """Cross-platform: return True iff a process with `pid` exists.

    Tries psutil first (if installed); falls back to OpenProcess /
    GetExitCodeProcess on Windows and signal 0 elsewhere. Conservatively
    returns True on errors so a transient permission glitch doesn't
    cause us to wrongly self-terminate.
    """
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    try:
        if os.name == 'nt':
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                     False, int(pid))
            if not h:
                # OpenProcess returns NULL when the PID is gone OR when
                # we lack permission. ERROR_INVALID_PARAMETER (87) is
                # the "no such process" code; everything else is treated
                # as alive-but-not-queryable.
                last = ctypes.get_last_error()
                return last != 87
            try:
                exit_code = ctypes.c_ulong()
                ok = kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
                return bool(ok) and exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(h)
        else:
            os.kill(int(pid), 0)
            return True
    except (OSError, PermissionError):
        # PermissionError on POSIX = pid exists but we can't signal it;
        # ProcessLookupError (OSError ESRCH) = gone. Conservatively
        # treat both as "alive" so we don't false-positive-exit.
        return True
    except Exception:
        return True


def _start_parent_watchdog(parent_pid, poll_interval_s=3.0):
    """Daemon thread that calls os._exit(0) when the parent dies."""
    import threading

    def _loop():
        while True:
            time.sleep(poll_interval_s)
            if not _is_pid_alive(parent_pid):
                # Parent gone -- die immediately. os._exit (not sys.exit)
                # bypasses Python finalisation so we release port 8050
                # without the slow Flask/Werkzeug teardown that can
                # itself hang during shutdown.
                try:
                    logger.warning(
                        'parent pid %d gone — dashboard exiting', parent_pid)
                except Exception:
                    pass
                os._exit(0)

    t = threading.Thread(target=_loop, name='dash-parent-watchdog',
                         daemon=True)
    t.start()
    return t


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
    # Phase 4 routing: mount Dash at /old/ so the new SLM-styled HTML
    # dashboard can take /. The url_base_pathname must end in /.
    app = Dash(__name__, title='Yb Tweezer Dashboard (old live page)',
               url_base_pathname='/old/')

    # ---- Read-only JSON endpoints (piggyback on Dash's Flask server) ----
    # Lets external clients (e.g. the SLM server) poll experiment state over
    # the LAN. All GET, no writes. Bound to the same port as the dashboard.
    _register_api_routes(app.server)

    # ---- New SLM-styled HTML dashboard at / (Phase 4) -----------------
    _register_main_html_routes(app.server)

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

    # Phase 4 layout: 4 named tabs sharing the existing callback machinery.
    # Every callback-bound ID stays in the DOM (Dash dcc.Tabs renders all
    # children and uses display:none to hide inactive ones), so refresh()
    # and the SLM / queue callbacks fire on every tick regardless of which
    # tab is visible. Anchor URLs come from dcc.Location: /#live, /#slm,
    # /#analysis, /#queue.
    _tab_style = {'backgroundColor': PANEL, 'color': TEXT,
                  'border': 'none', 'padding': '8px 18px',
                  'fontFamily': '"Segoe UI", sans-serif'}
    _tab_selected = {'backgroundColor': '#1c1c2e', 'color': '#e94560',
                     'border': 'none', 'padding': '8px 18px',
                     'borderBottom': '2px solid #e94560',
                     'fontWeight': '600'}

    _live_children = [
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
    ]   # end of _live_children

    _slm_children = [
        html.Div(id='slm-panel', style={
            'backgroundColor': PANEL, 'padding': '10px 14px',
            'marginTop': '6px', 'borderRadius': '4px',
            'fontFamily': '"Segoe UI", sans-serif', 'fontSize': '12px',
            'color': TEXT}),
    ]

    # Analysis tab (Phase 4): scan picker + lab-side run_analysis output +
    # protocol source viewer. All driven by callbacks below.
    _analysis_children = [
        html.Div(style={'display': 'flex', 'gap': '12px', 'marginTop': '6px',
                        'alignItems': 'flex-end', 'flexWrap': 'wrap'},
                 children=[
            html.Div(style={'flex': '1', 'minWidth': '320px'}, children=[
                html.Label('Scan ID (YYYYMMDDHHMMSS):',
                           style={'fontSize': '11px', 'color': '#aaa'}),
                dcc.Input(id='analysis-scan-id', type='text',
                          placeholder='e.g. 20260529025015',
                          debounce=True,
                          style={'width': '100%',
                                 'backgroundColor': '#1a1a2e',
                                 'color': TEXT,
                                 'border': '1px solid #2b2b4a',
                                 'borderRadius': '3px',
                                 'padding': '6px 8px',
                                 'fontFamily': 'monospace'}),
            ]),
            html.Button('Load', id='analysis-load-btn', n_clicks=0,
                        style={'backgroundColor': '#2a7fff', 'color': '#fff',
                               'border': 'none', 'borderRadius': '3px',
                               'padding': '7px 18px', 'cursor': 'pointer',
                               'fontWeight': '600'}),
            html.Div(id='analysis-status', style={
                'fontSize': '11px', 'color': '#888', 'flex': '1',
                'minWidth': '200px'}),
        ]),
        # Three side-by-side panels: summary text, survival curve,
        # loading-rate curve. Heights kept generous so survival fits read.
        html.Div(style={'display': 'flex', 'gap': '10px',
                        'marginTop': '10px'}, children=[
            html.Div(style={'flex': '1', 'minWidth': '0',
                            'backgroundColor': PANEL,
                            'padding': '10px 14px',
                            'borderRadius': '4px'}, children=[
                html.Div('Summary', style={'fontWeight': '600',
                                            'marginBottom': '6px',
                                            'color': '#e94560'}),
                html.Pre(id='analysis-summary', style={
                    'fontSize': '12px', 'color': TEXT,
                    'whiteSpace': 'pre-wrap', 'lineHeight': '1.5',
                    'margin': '0'}),
            ]),
            html.Div(style={'flex': '2', 'minWidth': '0'}, children=[
                dcc.Graph(id='analysis-survival',
                          figure=_waiting(''),
                          style={'height': '380px'},
                          config={'displayModeBar': False}),
            ]),
            html.Div(style={'flex': '2', 'minWidth': '0'}, children=[
                dcc.Graph(id='analysis-loading',
                          figure=_waiting(''),
                          style={'height': '380px'},
                          config={'displayModeBar': False}),
            ]),
        ]),
        # Protocol source viewer (lazy-fetched via slm_sync.ondemand).
        html.Details([
            html.Summary('Protocol source (rearrange_protocols.py at scan time)',
                         style={'cursor': 'pointer', 'color': '#aaa',
                                'fontSize': '12px', 'padding': '8px 0'}),
            html.Pre(id='analysis-protocol-src', style={
                'fontSize': '11px', 'color': '#ccc',
                'backgroundColor': '#101020', 'padding': '10px 14px',
                'borderRadius': '4px', 'maxHeight': '500px',
                'overflow': 'auto', 'whiteSpace': 'pre',
                'fontFamily': 'Consolas, "Courier New", monospace'}),
        ], style={'marginTop': '14px'}),
    ]

    # Queue tab (Phase 4): existing queue-panel + Submit-Scan form.
    _queue_children = [
        html.Div(id='queue-panel', style={
            'backgroundColor': PANEL, 'padding': '10px 14px',
            'marginTop': '6px', 'borderRadius': '4px',
            'fontFamily': '"Segoe UI", sans-serif', 'fontSize': '12px',
            'color': TEXT}),
        html.Div(style={'backgroundColor': PANEL,
                        'padding': '12px 14px', 'marginTop': '12px',
                        'borderRadius': '4px'}, children=[
            html.Div('Submit Scan (Phase 3 descriptor)', style={
                'fontWeight': '600', 'color': '#e94560',
                'marginBottom': '8px'}),
            html.Div(style={'fontSize': '11px', 'color': '#888',
                            'marginBottom': '8px'},
                children=['Paste a JSON descriptor conforming to ',
                          html.Code('yb_analysis/scans/descriptor.schema.json',
                                    style={'fontSize': '11px',
                                           'backgroundColor': '#1a1a2e',
                                           'padding': '1px 5px'}),
                          ' — see /api/endpoints for the route.']),
            dcc.Textarea(id='submit-scan-json',
                         placeholder='{\n  "seq": "CoolingSeq",\n  '
                                     '"params": {"Cooling.Detuning": '
                                     '{"scan": 1, "linspace": [20e6, 30e6, 11]}},\n  '
                                     '"runp": {"NumPerGroup": 4000, "Scramble": true}\n}',
                         style={'width': '100%', 'minHeight': '180px',
                                'backgroundColor': '#101020', 'color': TEXT,
                                'border': '1px solid #2b2b4a',
                                'borderRadius': '3px', 'padding': '10px',
                                'fontFamily': 'Consolas, "Courier New", monospace',
                                'fontSize': '12px'}),
            html.Div(style={'marginTop': '8px', 'display': 'flex',
                            'gap': '10px', 'alignItems': 'center'}, children=[
                html.Button('Submit', id='submit-scan-btn', n_clicks=0,
                            style={'backgroundColor': '#2a7fff',
                                   'color': '#fff', 'border': 'none',
                                   'borderRadius': '3px',
                                   'padding': '7px 18px', 'cursor': 'pointer',
                                   'fontWeight': '600'}),
                html.Div(id='submit-scan-result', style={
                    'fontSize': '11px', 'color': '#888', 'flex': '1'}),
            ]),
        ]),
    ]

    app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh',
        'fontFamily': '"Segoe UI", sans-serif', 'color': TEXT,
        'padding': '10px'}, children=[
        # Drive tab selection from #fragment so /#live, /#slm, /#analysis,
        # /#queue are bookmarkable.
        dcc.Location(id='tab-url', refresh=False),
        html.H1('Yb Tweezer Dashboard', style={'textAlign': 'center',
            'color': '#e94560', 'margin': '5px 0 10px 0', 'fontSize': '24px'}),
        dcc.Tabs(id='main-tabs', value='live', persistence=True,
                 persistence_type='session', children=[
            dcc.Tab(label='Live', value='live', style=_tab_style,
                    selected_style=_tab_selected, children=_live_children),
            dcc.Tab(label='SLM hardware', value='slm', style=_tab_style,
                    selected_style=_tab_selected, children=_slm_children),
            dcc.Tab(label='Analysis', value='analysis', style=_tab_style,
                    selected_style=_tab_selected, children=_analysis_children),
            dcc.Tab(label='Queue', value='queue', style=_tab_style,
                    selected_style=_tab_selected, children=_queue_children),
        ]),
        # Debug
        html.Details([
            html.Summary('Debug Info', style={'cursor': 'pointer',
                'color': '#888', 'fontSize': '11px'}),
            html.Pre(id='debug-pre', style={'fontSize': '10px', 'color': '#aaa',
                'maxHeight': '300px', 'overflow': 'auto', 'whiteSpace': 'pre-wrap'}),
        ], style={'marginTop': '10px'}),
        dcc.Interval(id='tick', interval=3000, n_intervals=0),
        # Holds the downsample-toggle state; the callback's real job is the
        # side effect of writing _CONTROL_FILE for the main process to read.
        dcc.Store(id='downsample-state'),
        # Cache the most recent analysis-result so the protocol-source
        # callback can reference it without re-running run_analysis.
        dcc.Store(id='analysis-result-store'),
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

    # ===== Phase 4 callbacks =====================================

    # Tab <-> URL sync: bookmark /#live, /#slm, /#analysis, /#queue.
    @app.callback(Output('main-tabs', 'value'),
                  Input('tab-url', 'hash'),
                  prevent_initial_call=True)
    def _sync_tab_from_url(h):
        if not h:
            return no_update
        h = h.lstrip('#').lower()
        return h if h in ('live', 'slm', 'analysis', 'queue') else no_update

    @app.callback(Output('tab-url', 'hash'),
                  Input('main-tabs', 'value'),
                  prevent_initial_call=True)
    def _sync_url_from_tab(tab_val):
        return f'#{tab_val}' if tab_val else no_update

    # Analysis tab: load button -> run analysis -> render results.
    @app.callback(
        [Output('analysis-summary', 'children'),
         Output('analysis-survival', 'figure'),
         Output('analysis-loading', 'figure'),
         Output('analysis-status', 'children'),
         Output('analysis-result-store', 'data')],
        Input('analysis-load-btn', 'n_clicks'),
        State('analysis-scan-id', 'value'),
        prevent_initial_call=True)
    def _run_analysis_cb(n_clicks, scan_id):
        if not scan_id:
            return ('Enter a scan_id and click Load.',
                    _waiting('survival'), _waiting('loading'),
                    'no scan_id supplied', None)
        scan_id = str(scan_id).strip()
        try:
            from yb_analysis.analysis.run_analysis import (
                analyze_scan, RunAnalysisError)
        except Exception as ex:
            return (f'analysis import failed: {ex}',
                    _waiting('survival'), _waiting('loading'),
                    'import error', None)
        try:
            result = analyze_scan(scan_id)
        except RunAnalysisError as ex:
            return (f'Could not analyze: {ex}',
                    _waiting('survival'), _waiting('loading'),
                    f'error: {ex}', None)
        except Exception as ex:
            logger.exception('analysis callback failed')
            return (f'Unexpected error: {ex}',
                    _waiting('survival'), _waiting('loading'),
                    'unexpected error', None)
        summary_txt = _format_analysis_summary(result)
        surv_fig = _build_analysis_curve(result, 'survival_mean',
                                         'survival_sem', 'Survival (P11)')
        load_fig = _build_analysis_curve(result, 'loading_rate',
                                         'loading_rate_sem',
                                         'Loading rate')
        status = (f'loaded scan {result.get("scan_id","")}, '
                  f'{result.get("n_params",0)} params, '
                  f'{result.get("n_shots",0)} shots')
        return summary_txt, surv_fig, load_fig, status, result

    # Protocol source viewer: when analysis-result-store gets a scan, fetch
    # rearrange_protocols.py source via slm_sync.ondemand. The Details
    # element collapses by default; the body is rendered eagerly so opening
    # the disclosure is instant. Failures (no scan, no code snapshot, SLM
    # offline) surface as a one-line message rather than blank.
    @app.callback(Output('analysis-protocol-src', 'children'),
                  Input('analysis-result-store', 'data'),
                  prevent_initial_call=True)
    def _fetch_protocol_source(result):
        if not isinstance(result, dict):
            return '(no analysis loaded)'
        scan_id = result.get('scan_id')
        if not scan_id:
            return '(no scan_id)'
        scan_dir = result.get('scan_dir')
        try:
            from yb_analysis.slm_sync.ondemand import get_protocol_source
        except Exception as ex:
            return f'(import failed: {ex})'
        try:
            src = get_protocol_source(scan_id, scan_dir=scan_dir)
        except Exception as ex:
            return f'(fetch failed: {ex})'
        if src is None:
            return ('(no protocol source available — either no code '
                    'snapshot for this scan_id or the SLM PC is offline)')
        return src

    # Submit Scan form: POST descriptor JSON to /api/queue/submit.
    @app.callback(Output('submit-scan-result', 'children'),
                  Input('submit-scan-btn', 'n_clicks'),
                  State('submit-scan-json', 'value'),
                  prevent_initial_call=True)
    def _submit_scan_cb(n_clicks, body):
        if not body or not body.strip():
            return html.Span('Empty descriptor.',
                             style={'color': '#e94560'})
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as ex:
            return html.Span(f'Invalid JSON: {ex}',
                             style={'color': '#e94560'})
        try:
            from yb_analysis.scans.client import submit_scan
            from yb_analysis.scans.descriptor import DescriptorError
        except Exception as ex:
            return html.Span(f'scans import failed: {ex}',
                             style={'color': '#e94560'})
        try:
            did = submit_scan(
                seq=payload.get('seq'),
                params=payload.get('params'),
                runp=payload.get('runp'),
                opts=payload.get('opts'),
                label=payload.get('label', ''))
        except DescriptorError as ex:
            return html.Span(f'Descriptor invalid: {ex}',
                             style={'color': '#e94560'})
        except TimeoutError as ex:
            return html.Span(f'Runner unreachable: {ex}',
                             style={'color': '#e94560'})
        except Exception as ex:
            logger.exception('submit_scan callback failed')
            return html.Span(f'Submit failed: {ex}',
                             style={'color': '#e94560'})
        return html.Span(f'submitted descriptor_id={did}',
                         style={'color': '#52d273', 'fontWeight': '600'})

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


# ---- Phase 4: Analysis tab helpers ----

def _format_analysis_summary(result):
    """Human-readable text block for the Summary panel in the Analysis tab.

    Pulls the high-signal fields from an `analyze_scan` result so the
    operator can see at a glance: scan id, sweep shape, mean survival,
    mean loading, SLM diag rollups, and code/grid sidecar presence.
    """
    if not isinstance(result, dict):
        return 'No analysis result.'
    sweep = result.get('sweep') or {}
    summary = result.get('summary') or {}
    diag = result.get('diag_aggregate')
    code = result.get('code') or {}
    grid = result.get('grid') or {}

    def _avg(values):
        vs = [v for v in (values or []) if v is not None]
        if not vs:
            return None
        return sum(vs) / len(vs)

    lines = [
        f"scan_id    : {result.get('scan_id', '')}",
        f"scan_name  : {result.get('scan_name') or '(none)'}",
        f"sweep cols : {', '.join(sweep.get('cols') or []) or '(none)'}",
        f"sweep dims : {sweep.get('dims') or []}",
        f"n_params   : {result.get('n_params', 0)}",
        f"n_shots    : {result.get('n_shots', 0)}",
        '',
        f"mean survival : {_fmt_pct(_avg(summary.get('survival_mean')))}",
        f"mean loading  : {_fmt_pct(_avg(summary.get('loading_rate')))}",
        f"mean loss     : {_fmt_pct(_avg(summary.get('loss_mean')))}",
    ]
    if diag is not None:
        lines += [
            '',
            f"diag rows     : {diag.get('n_rows', 0)}",
            f"mean total_ms : {_fmt_num(diag.get('mean_total_ms'))}",
            f"p99 total_ms  : {_fmt_num(diag.get('p99_total_ms'))}",
            f"mean n_loaded : {_fmt_num(diag.get('mean_n_loaded'))}",
            f"aborted shots : {diag.get('aborted_count', 0)}",
        ]
    lines += [
        '',
        f"code snapshot : {'present' if code.get('present') else '(none)'}"
        + (f" ({code.get('n_files', 0)} files)" if code.get('present') else ''),
        f"grid sidecar  : {'present' if grid.get('present') else '(none)'}"
        + (f" ({grid.get('n_sites', 0)} sites)" if grid.get('present') else ''),
    ]
    return '\n'.join(lines)


def _fmt_pct(v):
    if v is None:
        return '(no data)'
    try:
        return f'{100.0 * float(v):.2f}%'
    except (TypeError, ValueError):
        return '(no data)'


def _fmt_num(v):
    if v is None:
        return '(no data)'
    try:
        return f'{float(v):.2f}'
    except (TypeError, ValueError):
        return '(no data)'


def _build_analysis_curve(result, mean_key, sem_key, title):
    """Build a 1-D Plotly line+errorbar figure from the summary block.

    For 2-D scans this still renders (x = scan_point_index instead of a
    sweep value), but the picker would ideally pivot to a heatmap; that's
    a Phase 4b polish.
    """
    summary = (result or {}).get('summary') or {}
    sweep = (result or {}).get('sweep') or {}
    y = summary.get(mean_key) or []
    yerr = summary.get(sem_key) or []
    if not y:
        return _waiting(title, 'No data')

    # x axis: 1-D scan -> sweep values, else scan-point index.
    values = sweep.get('values') or []
    if len(values) == 1 and len(values[0]) == len(y):
        x = values[0]
        xlabel = (sweep.get('cols') or ['param'])[0]
    else:
        x = list(range(1, len(y) + 1))
        xlabel = 'scan point'

    # Drop None entries from y / yerr in parallel.
    pts = [(xi, yi, ei if isinstance(ei, (int, float)) else None)
           for xi, yi, ei in zip(x, y, yerr + [None] * len(y))
           if yi is not None]
    if not pts:
        return _waiting(title, 'No data')
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    es = [p[2] for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        error_y=dict(type='data', array=es, visible=True, color='#2a7fff'),
        mode='markers+lines',
        marker=dict(size=8, color='#2a7fff'),
        line=dict(color='#2a7fff', width=2),
        name=mean_key,
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=40, b=45),
        title=title,
        xaxis=dict(title=xlabel, gridcolor='#2b2b4a'),
        yaxis=dict(gridcolor='#2b2b4a'),
        showlegend=False,
        uirevision=mean_key,
    )
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
        # 0d (no swept axis): fall back to a per-shot time series so the
        # panel isn't dead during single-point scans. Uses
        # `loading_history`, which the data manager already maintains
        # as a rolling window of fraction-loaded per shot.
        return _fig_scan_timeseries(d)

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


def _fig_scan_timeseries(d):
    """0d-scan fallback for the Scan Curve panel.

    When no scan axis is defined, show fraction-loaded per shot over
    the recent history window. This is the same data the small
    "Loading rate (live)" panel uses, but rendered at full Scan-Curve
    size so the operator isn't staring at "Waiting for data..." for an
    entire 0d run.
    """
    hist = d.get('loading_history')
    if hist is None or len(hist) == 0:
        return _waiting('Scan Curve',
                        'No swept axis; waiting for shots to fill time series...')
    hist = np.asarray(hist, dtype=float)
    n = hist.size
    x = np.arange(1, n + 1)
    avg = float(hist.mean()) if n else 0.0
    fig = go.Figure(go.Scatter(
        x=x, y=hist, mode='lines+markers',
        line=dict(color='#58a6ff', width=1.5),
        marker=dict(size=4, color='#58a6ff'),
        name='loaded fraction',
    ))
    fig.add_shape(type='line', xref='paper', x0=0, x1=1, y0=avg, y1=avg,
                  line=dict(color='#ffdd44', width=1.5, dash='dash'))
    fig.add_annotation(
        text=f'Avg: {100*avg:.1f}%  ({n} shots, 0d / no swept axis)',
        xref='paper', yref='paper', x=0.99, y=0.99,
        showarrow=False, xanchor='right', yanchor='top',
        font=dict(color='#ffdd44', size=11),
        bgcolor='rgba(20,20,40,0.7)')
    title = _scan_title(d.get('scan_name') or 'Scan (0d)',
                        d.get('scan_filename'))
    fig.update_layout(**_L, title=title,
                      xaxis=dict(title='Shot # (oldest → latest)', **_A),
                      yaxis=dict(title='Fraction loaded',
                                 tickformat='.0%', autorange=True, **_A))
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
