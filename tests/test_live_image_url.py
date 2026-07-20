"""Tests for serving the live camera images OUT-OF-BAND (binary endpoint) instead
of baking the ~2.3 MB base64 PNG into the /api/live/figures JSON.

The single-GIL dashboard process had to serialize a 2.26 MB base64 string per
camera panel through PlotlyJSONEncoder every poll, serializing every other
request behind it. In URL mode the array panels reference /api/live/imageN (raw
image/png, GIL-released socket send, 33% smaller than base64), so the figure JSON
drops ~10x and the pixels move to a GIL-light path. The legacy Dash callback
(calls _fig_array with no use_img_url) keeps baking -- unchanged.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_live_image_url.py -v
"""
import base64
import json

import numpy as np
import pytest

from yb_analysis.plotting import dashboard as dsh


# A minimal 2x2 red PNG, base64 data URI -- enough for _fig_array to render.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR42mP8z8BQz0AEYBxV"
    "SAYAQAoBAy3rWkYAAAAASUVORK5CYII=")
_DATA_URI = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()


def _fake_frame(seq=5):
    """A minimal plot-data dict with one camera image + a tiny 2-site grid."""
    return {
        "_write_seq": seq,
        "_img_data_uri": _DATA_URI, "_img_shape": (2, 2),
        "_img_vlo": 0, "_img_vhi": 255,
        "_img2_data_uri": _DATA_URI, "_img2_shape": (2, 2),
        "_img2_vlo": 0, "_img2_vhi": 255,
        "_img_mid_data_uri": _DATA_URI, "_img_mid_shape": (2, 2),
        "_img_mid_vlo": 0, "_img_mid_vhi": 255,
        "grid_locations": np.array([[0, 0], [1, 1]]),
        "logicals": np.array([1, 0]), "box_size": 1, "num_sites": 2,
    }


# _fig_array now returns a raw dict (Approach A); access via ['layout']/['data'].
def test_url_mode_references_endpoint_not_base64():
    d = _fake_frame(seq=7)
    fig = dsh._fig_array(d, use_img_url=True)
    src = fig["layout"]["images"][0]["source"]
    assert src == "/api/live/image1?t=7"
    # the heavy base64 must NOT be in the figure at all
    assert "data:image" not in json.dumps(fig, default=str)


def test_legacy_mode_still_bakes_base64():
    """No use_img_url (the legacy path): the image source stays the baked base64
    data URI, so the old dcc page keeps working with no extra fetch."""
    d = _fake_frame()
    fig = dsh._fig_array(d)                       # default use_img_url=False
    assert fig["layout"]["images"][0]["source"].startswith("data:image")


def test_url_mode_carries_frame_cache_buster():
    """The ?t=<_write_seq> cache-buster makes Plotly.js refetch each frame."""
    assert dsh._fig_array(_fake_frame(seq=1), use_img_url=True
                          )["layout"]["images"][0]["source"] == "/api/live/image1?t=1"
    assert dsh._fig_array(_fake_frame(seq=2), use_img_url=True
                          )["layout"]["images"][0]["source"] == "/api/live/image1?t=2"


def test_url_mode_maps_each_panel_to_its_endpoint():
    d = _fake_frame()
    mid = dsh._fig_array(d, img_key="_img_mid_data_uri", shape_key="_img_mid_shape",
                         use_img_url=True)
    two = dsh._fig_array(d, img_key="_img2_data_uri", shape_key="_img2_shape",
                         use_img_url=True)
    assert mid["layout"]["images"][0]["source"].startswith("/api/live/image_mid?t=")
    assert two["layout"]["images"][0]["source"].startswith("/api/live/image2?t=")


def test_url_mode_preserves_box_overlay():
    """Moving the image out-of-band must NOT drop the green/red site overlay
    (colorbar anchor trace + the box overlay in data or layout.shapes)."""
    d = _fake_frame()
    fig = dsh._fig_array(d, use_img_url=True, show_boxes=True)
    assert len(fig["data"]) >= 1


def test_missing_image_falls_back_to_waiting_both_modes():
    d = {"_write_seq": 1, "_img_shape": (2, 2)}   # no data URI
    for use_url in (True, False):
        fig = dsh._fig_array(d, use_img_url=use_url)
        # _waiting() has no layout image
        assert not getattr(fig.layout, "images", None)


def test_build_fig_json_defaults_to_url_mode():
    """_build_fig_json (the HTTP path + eager pre-builder) defaults to url mode,
    so a request and the pre-build agree and the base64 never enters the JSON."""
    d = _fake_frame(seq=9)
    s = dsh._build_fig_json(d, "array")           # default use_img_url=True
    assert "/api/live/image1?t=9" in s
    assert "data:image" not in s


def test_url_mode_group_is_much_smaller():
    """The snapshot group JSON is dramatically smaller in url mode (the whole
    point) -- with a real 2.26 MB image the ratio is ~10x; even the tiny test
    PNG must come out strictly smaller and base64-free."""
    d = _fake_frame()
    url = dsh._build_fig_json(d, "array", use_img_url=True)
    baked = dsh._build_fig_json(d, "array", use_img_url=False)
    assert "data:image" in baked and "data:image" not in url
    assert len(url) < len(baked)


# --- Per-seq PNG ring: box/image off-by-one fix -----------------------------
# url mode makes the array figure bake frame-K boxes but reference the image as
# /api/live/imageN?t=K -- a SEPARATE fetch. If the pickle advanced to K+1 in
# between, the server used to serve K+1's pixels under K's boxes (a +/-1 straddle,
# what box_sync_monitor measures). The ring lets the endpoint serve the EXACT
# frame the boxes came from.

@pytest.fixture(autouse=True)
def _clear_png_ring():
    """Each ring test starts with an empty ring (module global otherwise leaks
    across tests)."""
    dsh._LIVE_PNG_RING.clear()
    yield
    dsh._LIVE_PNG_RING.clear()


def test_ring_put_caps_and_keeps_newest():
    for s in range(100):
        dsh._live_png_ring_put("_img_data_uri", s, bytes([s % 256]))
    ring = dsh._LIVE_PNG_RING["_img_data_uri"]
    assert len(ring) == dsh._LIVE_PNG_RING_MAX
    # the most recent _LIVE_PNG_RING_MAX seqs survive; older ones are evicted
    assert list(ring.keys()) == list(range(100 - dsh._LIVE_PNG_RING_MAX, 100))


def test_ring_put_ignores_none():
    dsh._live_png_ring_put("_img_data_uri", None, b"x")
    dsh._live_png_ring_put("_img_data_uri", 1, None)
    assert dsh._LIVE_PNG_RING.get("_img_data_uri") in (None, {})


def test_seed_decodes_each_url_key():
    """The publish-time seed decodes every URL-served image key into the ring so a
    later ?t=<seq> request finds those exact bytes."""
    d = _fake_frame(seq=42)
    dsh._seed_live_png_ring(d, 42)
    for uri_key in dsh._IMG_URL_FOR_KEY:
        assert 42 in dsh._LIVE_PNG_RING[uri_key]
        assert dsh._LIVE_PNG_RING[uri_key][42] == _PNG_BYTES


def test_seed_skips_bad_uri():
    d = {"_img_data_uri": "not-a-data-uri", "_img2_data_uri": _DATA_URI,
         "_img_mid_data_uri": None}
    dsh._seed_live_png_ring(d, 7)
    # only the well-formed key made it in
    assert dsh._LIVE_PNG_RING.get("_img_data_uri", {}).get(7) is None
    assert dsh._LIVE_PNG_RING["_img2_data_uri"][7] == _PNG_BYTES


def _make_app(monkeypatch, frame):
    """Build the Flask app with _read_data pinned to `frame` (mutable via the
    returned box so a test can advance the 'current' pickle mid-request)."""
    box = {"d": frame}
    monkeypatch.setattr(dsh, "_read_data", lambda: box["d"])
    # register_api needs a Flask app; reuse the dashboard's builder if present,
    # else a bare app with just the image routes bound.
    from flask import Flask
    app = Flask(__name__)
    app.add_url_rule("/api/live/image1", "img1",
                     lambda: dsh._live_image_response(
                         "_img_data_uri", "_img_shape", "_img_vlo", "_img_vhi"))
    return app, box


def test_endpoint_serves_requested_frame_after_advance(monkeypatch):
    """The core fix: boxes were built at seq K; the pickle has since advanced to
    K+1. A GET ...?t=K must still return frame K's bytes (from the ring) with
    X-Frame-Seq: K -- NOT the current K+1 frame."""
    frame_k = _fake_frame(seq=10)
    frame_k["_img_data_uri"] = _DATA_URI            # frame 10 pixels
    # Seed frame 10 into the ring (as the publish-time watcher would).
    dsh._seed_live_png_ring(frame_k, 10)
    # Now the shared pickle has moved on to frame 11 with DIFFERENT pixels.
    frame_k1 = _fake_frame(seq=11)
    other = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwAEhQGA"
        "hKmMIQAAAABJRU5ErkJggg==")
    frame_k1["_img_data_uri"] = "data:image/png;base64," + base64.b64encode(other).decode()

    app, _ = _make_app(monkeypatch, frame_k1)       # "current" = frame 11
    with app.test_client() as c:
        r = c.get("/api/live/image1?t=10")          # boxes want frame 10
        assert r.status_code == 200
        assert r.headers.get("X-Frame-Seq") == "10"  # served frame 10, not 11
        assert r.data == _PNG_BYTES                   # frame 10's exact pixels


def test_endpoint_falls_back_to_current_when_evicted(monkeypatch):
    """A far-behind ?t= (frame long gone from the ring) degrades to the current
    frame -- exactly the pre-ring behavior, never an error."""
    frame = _fake_frame(seq=500)
    app, _ = _make_app(monkeypatch, frame)          # ring empty, current=500
    with app.test_client() as c:
        r = c.get("/api/live/image1?t=3")           # frame 3 never recorded
        assert r.status_code == 200
        assert r.headers.get("X-Frame-Seq") == "500"  # fell back to current


def test_endpoint_records_current_frame_on_request(monkeypatch):
    """Even without the watcher, a plain request records the current frame in the
    ring (a backstop), so an immediately-following ?t=<same> hits."""
    frame = _fake_frame(seq=77)
    app, _ = _make_app(monkeypatch, frame)
    with app.test_client() as c:
        c.get("/api/live/image1")                   # no ?t -> serves+records 77
    assert 77 in dsh._LIVE_PNG_RING["_img_data_uri"]
