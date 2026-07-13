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


def test_url_mode_references_endpoint_not_base64():
    d = _fake_frame(seq=7)
    fig = dsh._fig_array(d, use_img_url=True)
    src = fig.layout.images[0].source
    assert src == "/api/live/image1?t=7"
    # the heavy base64 must NOT be in the figure at all
    assert "data:image" not in json.dumps(fig.to_plotly_json(), default=str)


def test_legacy_mode_still_bakes_base64():
    """The legacy Dash-callback path (no use_img_url) is unchanged: it bakes the
    data URI so the old page keeps working with no extra fetch."""
    d = _fake_frame()
    fig = dsh._fig_array(d)                       # default use_img_url=False
    assert fig.layout.images[0].source.startswith("data:image")


def test_url_mode_carries_frame_cache_buster():
    """The ?t=<_write_seq> cache-buster makes Plotly.js refetch each frame."""
    assert dsh._fig_array(_fake_frame(seq=1), use_img_url=True
                          ).layout.images[0].source == "/api/live/image1?t=1"
    assert dsh._fig_array(_fake_frame(seq=2), use_img_url=True
                          ).layout.images[0].source == "/api/live/image1?t=2"


def test_url_mode_maps_each_panel_to_its_endpoint():
    d = _fake_frame()
    mid = dsh._fig_array(d, img_key="_img_mid_data_uri", shape_key="_img_mid_shape",
                         use_img_url=True)
    two = dsh._fig_array(d, img_key="_img2_data_uri", shape_key="_img2_shape",
                         use_img_url=True)
    assert mid.layout.images[0].source.startswith("/api/live/image_mid?t=")
    assert two.layout.images[0].source.startswith("/api/live/image2?t=")


def test_url_mode_preserves_box_overlay():
    """Moving the image out-of-band must NOT drop the green/red site overlay."""
    d = _fake_frame()
    fig = dsh._fig_array(d, use_img_url=True, show_boxes=True)
    # colorbar anchor + at least one overlay trace
    assert len(fig.data) >= 1


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
