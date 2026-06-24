"""/api/runs/<id>/avg_image: non-blocking background compute + status polling.

The averaged-image PNG takes ~10 s to compute for a big single-image scan (one
gzip-decompress per sampled frame). The route must NOT block the request (and the
browser <img>) for that long -- it kicks the compute onto a background thread and
returns 202 ``{status:'computing'}`` immediately; the client polls ``?check=1`` and
shows a clear message until the PNG is ready. These tests use a tiny in-memory /imgs
so the "compute" is fast, and drive the route through a Flask test client.

    python -m pytest yb_analysis/tests/test_avg_image_async_route.py -v
"""

import time

import numpy as np
import pytest

from yb_analysis.plotting import dashboard as dash_mod


def _make_scan(tmp_path, n_frames=4, h=8, w=8):
    """Create a fake scan dir with a data_*.h5 carrying a small /imgs."""
    import h5py
    scan_dir = tmp_path / "data_20260623_141027"
    scan_dir.mkdir()
    h5 = scan_dir / "data_20260623_141027.h5"
    with h5py.File(h5, "w") as f:
        imgs = (np.arange(n_frames * h * w, dtype=np.uint16)
                .reshape(n_frames, h, w))
        f.create_dataset("imgs", data=imgs, chunks=(1, h, w),
                         compression="gzip")
    return scan_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    from flask import Flask
    app = Flask("avg_image_test")
    dash_mod._register_api_routes(app)
    app.testing = True
    # Fresh module-level build state per test.
    monkeypatch.setattr(dash_mod, "_AVG_IMAGE_BUILDS", {})
    monkeypatch.setattr(dash_mod, "_AVG_IMAGE_ERRORS", {})
    scan_dir = _make_scan(tmp_path)
    monkeypatch.setattr(dash_mod, "_resolve_scan_dir",
                        lambda scan_id: str(scan_dir))
    return app.test_client(), scan_dir


def _await_png(scan_dir, timeout_s=10.0):
    from yb_analysis.analysis.run_analysis import AVG_IMAGE_PNG
    png = scan_dir / AVG_IMAGE_PNG
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if png.is_file():
            return png
        time.sleep(0.05)
    return png


def test_unknown_scan_404(client):
    cl, _sd = client
    import yb_analysis.plotting.dashboard as d
    # Point resolver at nothing.
    d._resolve_scan_dir = lambda scan_id: None
    r = cl.get("/api/runs/20260101000000/avg_image")
    assert r.status_code == 404


def test_first_request_is_nonblocking_202(client):
    cl, scan_dir = client
    # Not cached yet -> must return 202 'computing' immediately, NOT block.
    t0 = time.time()
    r = cl.get("/api/runs/20260623141027/avg_image")
    elapsed = time.time() - t0
    assert r.status_code == 202
    assert r.get_json()["status"] == "computing"
    # The point of the change: the request returns fast (the compute is on a
    # background thread). Even our tiny image shouldn't make this slow, and a
    # real one would be ~10 s -- so assert a generous-but-real bound.
    assert elapsed < 2.0


def test_check_reports_status_then_ready(client):
    cl, scan_dir = client
    # ?check=1 also kicks the build and reports a non-image status.
    r = cl.get("/api/runs/20260623141027/avg_image?check=1")
    assert r.status_code in (200, 202)
    assert r.get_json()["status"] in ("computing", "ready")
    # Wait for the background compute to finish, then check=1 -> ready.
    _await_png(scan_dir)
    r2 = cl.get("/api/runs/20260623141027/avg_image?check=1")
    assert r2.status_code == 200
    assert r2.get_json()["status"] == "ready"


def test_serves_png_once_ready(client):
    cl, scan_dir = client
    cl.get("/api/runs/20260623141027/avg_image")     # kick the build
    _await_png(scan_dir)
    r = cl.get("/api/runs/20260623141027/avg_image")  # now cached
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"          # PNG magic


def test_error_is_reported_not_looped(client, monkeypatch):
    cl, scan_dir = client
    # Force the compute to fail; the route should surface 'error', and a repeat
    # request must NOT respawn a thread (deterministic failure short-circuits).
    import yb_analysis.analysis.run_analysis as ra

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(ra, "ensure_avg_image_png", boom)
    cl.get("/api/runs/20260623141027/avg_image")      # spawns thread that fails
    # Let the thread record the error.
    deadline = time.time() + 5.0
    while time.time() < deadline and not dash_mod._AVG_IMAGE_ERRORS:
        time.sleep(0.05)
    assert dash_mod._AVG_IMAGE_ERRORS, "error should be recorded"
    r = cl.get("/api/runs/20260623141027/avg_image?check=1")
    assert r.get_json()["status"] == "error"
    assert "synthetic failure" in r.get_json()["error"]
