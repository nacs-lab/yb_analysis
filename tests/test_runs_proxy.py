"""Analysis-API offload proxy (dashboard._install_runs_proxy / _should_proxy_runs).

The heavy DISK-reading /api/runs/* endpoints run in a sibling process (own GIL)
so a live scan's figure pre-builder pegging the Dash process can't starve them.
The Dash app forwards those paths to the sibling via a before_request hook.

Run: yb_analysis-env python -m pytest yb_analysis/tests/test_runs_proxy.py -v
"""
import os

import pytest
from flask import Flask

from yb_analysis.plotting import dashboard as D


@pytest.mark.parametrize('path, expected', [
    ('/api/runs/list', True),
    ('/api/runs/dates', True),
    ('/api/runs/20260713154236/analysis', True),
    ('/api/runs/20260713154236/avg_image', True),
    ('/api/runs/20260713154236/grid', True),
    ('/api/runs/20260713154236/code', True),
    ('/api/runs/20260713154236/shot_image', True),
    ('/api/runs/groups', True),
    ('/api/runs/groups/g1/analysis', True),
    # NOT offloaded -- needs live in-process state, or isn't a runs endpoint:
    ('/api/runs/20260713154236/diag_live', False),
    ('/api/runs/20260713154236/diag', False),
    ('/api/status', False),
    ('/api/live/figures', False),
    ('/old/', False),
])
def test_should_proxy_runs(path, expected):
    assert D._should_proxy_runs(path) is expected


def test_install_is_noop_without_env(monkeypatch):
    """No sibling configured -> the hook is NOT installed (old in-process behavior)."""
    monkeypatch.delenv('YB_ANALYSIS_API_PORT', raising=False)
    app = Flask(__name__)
    before = len(app.before_request_funcs.get(None, []))
    D._install_runs_proxy(app)
    after = len(app.before_request_funcs.get(None, []))
    assert after == before      # nothing added


def test_install_adds_hook_with_env(monkeypatch):
    monkeypatch.setenv('YB_ANALYSIS_API_PORT', '8071')
    app = Flask(__name__)
    D._install_runs_proxy(app)
    assert len(app.before_request_funcs.get(None, [])) == 1


def test_proxy_forwards_offloaded_and_passes_through_others(monkeypatch):
    """With the env set, an offloaded path is forwarded via the requests
    session (stubbed); a non-offloaded path falls through to the local view."""
    monkeypatch.setenv('YB_ANALYSIS_API_PORT', '8071')

    forwarded = {}

    _BODY = b'{"runs": [], "count": 0, "_via": "sibling"}'

    class _FakeRaw:
        headers = {'Content-Type': 'application/json'}
        def read(self, decode_content=False):
            return _BODY

    class _FakeResp:
        status_code = 200
        raw = _FakeRaw()

    class _FakeSession:
        def request(self, method, url, data, headers, timeout, stream=False):
            forwarded['url'] = url
            forwarded['method'] = method
            return _FakeResp()

    import requests as _rq
    monkeypatch.setattr(_rq, 'Session', lambda: _FakeSession())

    app = Flask(__name__)

    @app.route('/api/runs/list')          # LOCAL handler (should be bypassed)
    def _local_list():
        return ('LOCAL', 200)

    @app.route('/api/status')             # non-offloaded local handler
    def _local_status():
        return ('STATUS-LOCAL', 200)

    D._install_runs_proxy(app)
    c = app.test_client()

    # Offloaded -> forwarded to the sibling (local handler bypassed).
    r = c.get('/api/runs/list?max=50')
    assert r.status_code == 200
    assert b'sibling' in r.data
    assert forwarded['url'].startswith('http://127.0.0.1:8071/api/runs/list')

    # Non-offloaded -> served by the local handler (never forwarded).
    forwarded.clear()
    r2 = c.get('/api/status')
    assert r2.data == b'STATUS-LOCAL'
    assert forwarded == {}


def test_proxy_falls_back_to_local_when_sibling_down(monkeypatch):
    """Sibling unreachable -> the hook returns None so Flask serves the still-
    registered local handler (slower under load, but never a hard failure)."""
    monkeypatch.setenv('YB_ANALYSIS_API_PORT', '8071')

    class _DeadSession:
        def request(self, *a, **k):     # accepts stream= and all kwargs
            raise ConnectionError('sibling down')

    import requests as _rq
    monkeypatch.setattr(_rq, 'Session', lambda: _DeadSession())

    app = Flask(__name__)

    @app.route('/api/runs/list')
    def _local_list():
        return ('LOCAL-FALLBACK', 200)

    D._install_runs_proxy(app)
    c = app.test_client()
    r = c.get('/api/runs/list')
    assert r.data == b'LOCAL-FALLBACK'    # fell through to local


def test_gzip_compresses_large_json_when_accepted():
    """_install_gzip compresses a big body only when the client accepts gzip;
    small bodies + non-accepting clients pass through untouched."""
    import gzip
    app = Flask(__name__)
    big = b'x' * 50000

    @app.route('/big')
    def _big():
        return (big, 200, {'Content-Type': 'application/json'})

    @app.route('/small')
    def _small():
        return (b'ok', 200, {'Content-Type': 'application/json'})

    D._install_gzip(app)
    c = app.test_client()

    # accepts gzip + big -> compressed, and decompresses back to the original
    r = c.get('/big', headers={'Accept-Encoding': 'gzip'})
    assert r.headers.get('Content-Encoding') == 'gzip'
    assert len(r.data) < len(big)
    assert gzip.decompress(r.data) == big

    # no Accept-Encoding -> untouched
    r2 = c.get('/big')
    assert 'Content-Encoding' not in r2.headers
    assert r2.data == big

    # small body -> not worth compressing, untouched
    r3 = c.get('/small', headers={'Accept-Encoding': 'gzip'})
    assert 'Content-Encoding' not in r3.headers
    assert r3.data == b'ok'
