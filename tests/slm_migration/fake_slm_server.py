"""Lightweight fake SLM server for Phase 0–2 tests.

Implements the subset of SLMnet's HTTP surface that yb_analysis.slm_proxy
polls: /health, /devices, /lock/status, /camera/capture_png, /slm/phase_png,
/slm/rearrange_diag. Phase 2 adds three more endpoints
(/slm/runs/{scan_id}/diag, /slm/runs/{scan_id}/code, /slm/code/by_hash/{sha256})
when those tests are written.

Used as a context manager so test teardown is automatic:

    with FakeSlmServer() as fake:
        proxy = SlmProxy(slm_url=fake.url, intervals_ms={'health': 50, ...})
        proxy.start()
        ...
        proxy.stop()

The server runs in a daemon thread on an OS-assigned free port (`127.0.0.1:0`).
Payloads are configurable via setter methods so individual tests can craft
specific scenarios without restarting the server.
"""

import threading
from io import BytesIO

# A minimal 1x1 PNG (base64-decoded once at import time). Real PNG bytes so
# the proxy's response-content check sees image/png-shaped data; the dashboard
# doesn't actually decode it in Phase 0 (just wraps in data URI).
_MIN_PNG = (
    b'\x89PNG\r\n\x1a\n'
    b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00'
    b'\x1f\x15\xc4\x89'
    b'\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfe\x02\xfe\xdc\xccY\xe7'
    b'\x00\x00\x00\x00IEND\xaeB`\x82'
)


class FakeSlmServer:
    """A minimal Flask-based fake of SLMnet's HTTP endpoints.

    Defaults to returning sensible-looking JSON / PNG payloads. Tests can
    customize behavior via the setter methods below before / during a test.
    """

    def __init__(self, port=0):
        from flask import Flask, jsonify, Response, request

        self._app = Flask('fake_slm_server')
        self._port = port
        self._server = None
        self._thread = None
        self._url = None
        self._lock = threading.Lock()

        # Mutable response payloads (test customizable).
        self._payloads = {
            'health':         {'uptime_s': 12.5, 'version': 'fake-1'},
            'devices':        {'slm': 'ok', 'camera': 'ok'},
            'lock_status':    {'slm': {'holder': 'matlab', 'age_s': 3}},
            'rearrange_diag': {'count': 0, 'entries': []},
        }
        self._png = {
            'camera': _MIN_PNG,
            'phase':  _MIN_PNG,
        }
        # Per-endpoint failure injection: dict[name] -> int HTTP status.
        # When set, the matching route returns that status instead of the
        # normal payload. Tests use this to exercise the proxy's error path.
        self._fail_status = {}
        # Request counters per endpoint — let tests assert polling happened.
        self._hits = {
            'health': 0, 'devices': 0, 'lock_status': 0,
            'camera_png': 0, 'phase_png': 0, 'rearrange_diag': 0,
            'rearrange': 0, 'results': 0, 'setup_rearrangement': 0,
            'initialize_loading_pattern': 0, 'write_loading_phase': 0,
        }
        # Phase 1: captured request bodies for the POST endpoints. Tests
        # use these to assert MATLAB sent scan_id / seq_id correctly.
        self._captured_bodies = {
            'rearrange': [],
            'results': [],
            'setup_rearrangement': [],
            'initialize_loading_pattern': [],
            'write_loading_phase': [],
        }
        # Per-endpoint gate-busy injection: name -> remaining number of
        # 503 'server busy' responses to emit before succeeding. Lets
        # tests exercise the client's gate-busy retry loop.
        self._gate_busy = {}
        # Default response payloads for the POST endpoints (override-able).
        self._post_payloads = {
            'rearrange': {'ok': True, 'diag': {'total_ms': 42, 'n_loaded': 100}},
            'results':   {'ok': True, 'saved_path': '/fake/path.json'},
            'setup_rearrangement': {'ok': True, 'n_init_grid': 100,
                                    'n_targets': 50},
            'initialize_loading_pattern': {
                'ok': True, 'name': 'fake', 'order': 'col', 'n_sites': 4,
                'positions_knm': [[0, 0], [1, 0], [0, 1], [1, 1]],
                'phases': [0.0, 0.1, 0.2, 0.3], 'amps': [1, 1, 1, 1],
                'lattice': {'rows': [0, 1, 0, 1], 'cols': [0, 0, 1, 1],
                            'n_rows': 2, 'n_cols': 2, 'pitch_x': 1.0,
                            'pitch_y': 1.0, 'row_basis': [1, 0],
                            'col_basis': [0, 1], 'tilt_deg': 0.0,
                            'n_missing': 0, 'x0': 0.0, 'y0': 0.0},
                'base_sha256': 'ab' * 32, 'loading_zernike': None,
                'wrote_to_slm': False, 'derive_ms': 1.0,
            },
            'write_loading_phase': {'ok': True, 'base_sha256': 'ab' * 32,
                                    'loading_zernike': None},
        }
        # Phase 2: per-scan diag ledger + code-snapshot stores. Tests
        # populate via add_diag_row / set_code_manifest / add_blob.
        self._scan_diag = {}           # scan_id -> list of diag rows
        self._scan_diag_overflow = {}  # scan_id -> bool
        self._scan_code = {}           # scan_id -> manifest dict
        self._blobs = {}               # sha256 -> source bytes (str or bytes)

        def _resp_json(name):
            with self._lock:
                self._hits[name] += 1
                fail = self._fail_status.get(name)
                if fail:
                    return jsonify({'error': f'forced {fail}'}), fail
                return jsonify(self._payloads[name])

        def _resp_png(name):
            with self._lock:
                self._hits[name] += 1
                fail = self._fail_status.get(name)
                if fail:
                    return jsonify({'error': f'forced {fail}'}), fail
                key = 'camera' if name == 'camera_png' else 'phase'
                return Response(self._png[key], mimetype='image/png')

        @self._app.route('/health')
        def _h():
            return _resp_json('health')

        @self._app.route('/devices')
        def _d():
            return _resp_json('devices')

        @self._app.route('/lock/status')
        def _ls():
            return _resp_json('lock_status')

        @self._app.route('/camera/capture_png')
        def _cpng():
            return _resp_png('camera_png')

        @self._app.route('/slm/phase_png')
        def _ppng():
            return _resp_png('phase_png')

        @self._app.route('/slm/rearrange_diag')
        def _rd():
            return _resp_json('rearrange_diag')

        # ---- Phase 1 POST endpoints (body capture for MATLAB tests) ----

        def _resp_post(name, augment=None):
            with self._lock:
                self._hits[name] += 1
                try:
                    body = request.get_json(silent=True) or {}
                except Exception:
                    body = {}
                self._captured_bodies[name].append(body)
                gb = self._gate_busy.get(name, 0)
                if gb > 0:
                    self._gate_busy[name] = gb - 1
                    return jsonify({'detail': 'server busy: rearrange'}), 503
                fail = self._fail_status.get(name)
                if fail:
                    return jsonify({'error': f'forced {fail}'}), fail
                payload = self._post_payloads[name]
                if augment is not None:
                    payload = augment(dict(payload), body)
                return jsonify(payload)

        def _augment_loading(payload, body):
            """Mimic the real server's 3-D vs 2-D response: when the request
            carries a non-empty ``planes_z_rad`` the server splits the sites
            across the declared planes (layer-major) and echoes the 3-D
            fields; otherwise every 3-D field is null (the legacy 2-D shape)."""
            planes = body.get('planes_z_rad')
            pos2d = payload.get('positions_knm', []) or []
            n = payload.get('n_sites', len(pos2d))
            if planes:
                k = len(planes)
                n_per_plane = [n // k] * k
                for i in range(n - (n // k) * k):
                    n_per_plane[i] += 1
                plane_of_site, z_rad = [], []
                for li, (npl, z) in enumerate(zip(n_per_plane, planes)):
                    plane_of_site += [li] * npl
                    z_rad += [float(z)] * npl
                pos3d = [[p[0], p[1], z_rad[i]] for i, p in enumerate(pos2d)]
                payload.update({
                    'is_3d': True, 'z_rad': z_rad, 'positions_knm3d': pos3d,
                    'planes_z_rad': [float(z) for z in planes],
                    'n_per_plane': n_per_plane, 'plane_of_site': plane_of_site,
                })
            else:
                payload.update({
                    'is_3d': False, 'z_rad': None, 'positions_knm3d': None,
                    'planes_z_rad': None, 'n_per_plane': None,
                    'plane_of_site': None,
                })
            return payload

        @self._app.route('/slm/rearrange', methods=['POST'])
        def _rearrange():
            return _resp_post('rearrange')

        @self._app.route('/slm/results', methods=['POST'])
        def _results():
            return _resp_post('results')

        @self._app.route('/slm/setup_rearrangement', methods=['POST'])
        def _setup_rearr():
            return _resp_post('setup_rearrangement')

        @self._app.route('/slm/initialize_loading_pattern', methods=['POST'])
        def _init_loading():
            return _resp_post('initialize_loading_pattern',
                              augment=_augment_loading)

        @self._app.route('/slm/write_loading_phase', methods=['POST'])
        def _write_loading():
            return _resp_post('write_loading_phase')

        # ---- Phase 2 retrieval endpoints ----
        # Per-scan diag ledger (Phase 1 endpoint, fake-ified for Phase 2 sync tests).
        # The fake's internal store is `self._scan_diag`, keyed by scan_id.
        @self._app.route('/slm/runs/<scan_id>/diag', methods=['GET'])
        def _runs_diag(scan_id):
            with self._lock:
                entries = list(self._scan_diag.get(str(scan_id), []))
                overflow = bool(self._scan_diag_overflow.get(str(scan_id),
                                                              False))
            since = request.args.get('since_seq_id')
            if since is not None:
                try:
                    cutoff = int(since)
                    entries = [e for e in entries
                               if isinstance(e.get('seq_id'), int)
                               and e['seq_id'] > cutoff]
                except (TypeError, ValueError):
                    pass
            return jsonify({'scan_id': str(scan_id), 'count': len(entries),
                            'overflow': overflow, 'entries': entries})

        # Per-scan code-snapshot manifest.
        @self._app.route('/slm/runs/<scan_id>/code', methods=['GET'])
        def _runs_code(scan_id):
            with self._lock:
                manifest = self._scan_code.get(str(scan_id))
            if manifest is None:
                return jsonify(
                    {'detail': f'no code snapshot for scan_id={scan_id}'}), 404
            return jsonify({
                'scan_id': str(scan_id),
                'safe_run_id': f'fake_sid_{scan_id}',
                'manifest': manifest,
                'manifest_path': f'fake/path/{scan_id}/manifest.json',
            })

        # Source-by-hash retrieval. The fake stores them in `self._blobs`.
        @self._app.route('/slm/code/by_hash/<sha256>', methods=['GET'])
        def _code_by_hash(sha256):
            import re as _re
            if not _re.fullmatch(r'[0-9a-f]{64}', sha256):
                return jsonify({'detail':
                                'sha256 must match ^[0-9a-f]{64}$'}), 400
            with self._lock:
                blob = self._blobs.get(sha256)
            if blob is None:
                return jsonify({'detail': f'no blob with sha256={sha256}'}), 404
            return Response(blob, mimetype='text/plain; charset=utf-8')

    # ---- Lifecycle ----

    def start(self):
        from werkzeug.serving import make_server
        self._server = make_server('127.0.0.1', self._port, self._app,
                                   threaded=True)
        # Capture the OS-assigned port if we passed port=0.
        self._port = self._server.server_port
        self._url = f'http://127.0.0.1:{self._port}'
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name='fake-slm-server', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- Test-facing API ----

    @property
    def url(self):
        return self._url

    def set_payload(self, name, payload):
        """Override the JSON payload returned by an endpoint."""
        with self._lock:
            self._payloads[name] = payload

    def set_png(self, which, png_bytes):
        """Override the camera/phase PNG bytes ('camera' or 'phase')."""
        with self._lock:
            self._png[which] = png_bytes

    def fail(self, name, status=500):
        """Force `name` endpoint to return the given HTTP status."""
        with self._lock:
            self._fail_status[name] = status

    def unfail(self, name):
        """Clear any forced failure on `name`."""
        with self._lock:
            self._fail_status.pop(name, None)

    def gate_busy(self, name, times):
        """Make `name` return 503 'server busy' for the next `times`
        requests before succeeding — exercises the client retry loop."""
        with self._lock:
            self._gate_busy[name] = int(times)

    def hits(self, name):
        """How many requests this endpoint has served so far."""
        with self._lock:
            return self._hits[name]

    def reset_hits(self):
        with self._lock:
            for k in self._hits:
                self._hits[k] = 0

    def captured_bodies(self, name):
        """Return a list of all JSON bodies POSTed to `name` so far.

        Used by Phase 1 tests to assert MATLAB sent the expected
        scan_id / seq_id fields.
        """
        with self._lock:
            return list(self._captured_bodies[name])

    def set_post_payload(self, name, payload):
        """Override the JSON response returned by a POST endpoint."""
        with self._lock:
            self._post_payloads[name] = payload

    def clear_captured(self):
        """Drop everything captured so far. Useful between test scenarios."""
        with self._lock:
            for k in self._captured_bodies:
                self._captured_bodies[k] = []

    # ---- Phase 2 test API ----

    def add_diag_row(self, scan_id, seq_id, diag=None, **extra):
        """Push one row into the per-scan diag ledger."""
        import time as _time
        with self._lock:
            self._scan_diag.setdefault(str(scan_id), []).append({
                'ts_iso': extra.get('ts_iso', '2026-05-29T00:00:00.000000'),
                'ts_epoch': extra.get('ts_epoch', _time.time()),
                'scan_id': str(scan_id),
                'seq_id': int(seq_id),
                'retry_count': extra.get('retry_count', 0),
                'diag': dict(diag) if diag else {},
                'run_id': extra.get('run_id', 'fake-run'),
                'client_id': extra.get('client_id', 'fake-client'),
            })

    def set_diag_overflow(self, scan_id, overflow=True):
        with self._lock:
            self._scan_diag_overflow[str(scan_id)] = bool(overflow)

    def set_code_manifest(self, scan_id, manifest):
        """Install a code-snapshot manifest dict for this scan_id."""
        with self._lock:
            self._scan_code[str(scan_id)] = dict(manifest)

    def add_blob(self, sha256, source_text):
        """Install a source-by-hash blob (as bytes or string)."""
        if isinstance(source_text, str):
            source_text = source_text.encode('utf-8')
        with self._lock:
            self._blobs[sha256] = source_text
