"""Calibration-free seq-specific focus metrics (LoadingDefocusScan).

Covers the raw-image spot detector, the per-defocus focus curve computed
straight from images (no grid / no thresholds), and the group aggregator.

    python -m pytest yb_analysis/tests/test_phase5_5_focus_metrics.py -v
"""

import numpy as np
import pytest

from yb_analysis.analysis.run_analysis import (
    _detect_spots_focus, _focus_metrics_from_images)
from yb_analysis.plotting.dashboard import _aggregate_focus_metrics


def _spot_image(sigma, H=90, W=90, spacing=20, amp=200.0, bg=10.0, seed=0):
    """A small array of Gaussian spots of width `sigma` on a flat background."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), bg, dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    for cy in range(spacing, H - spacing, spacing):
        for cx in range(spacing, W - spacing, spacing):
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                / (2 * sigma ** 2))
    return img + rng.normal(0, 1.0, img.shape)


def test_detect_spots_radius_tracks_width():
    r_tight, _, _ = _detect_spots_focus(_spot_image(1.5))
    r_wide, _, _ = _detect_spots_focus(_spot_image(3.0))
    assert r_tight.size > 0 and r_wide.size > 0
    # A wider Gaussian -> larger measured RMS radius.
    assert np.median(r_wide) > np.median(r_tight)


def test_detect_spots_empty_on_flat():
    flat = np.full((50, 50), 5.0)
    r, pk, ctr = _detect_spots_focus(flat)
    assert r.size == 0 and pk.size == 0 and ctr.size == 0


def test_focus_metrics_from_images_finds_best_focus(tmp_path):
    import h5py
    # 3 defocus points; sigma minimal (best focus) at the MIDDLE point.
    sigmas = [3.0, 1.4, 3.0]
    n_per = 4
    imgs, seq_ids = [], []
    for p, s in enumerate(sigmas):
        for k in range(n_per):
            imgs.append(_spot_image(s, seed=p * 10 + k))
            seq_ids.append(p + 1)
    h5 = tmp_path / 'data_20260101_000000.h5'
    with h5py.File(h5, 'w') as f:
        f.create_dataset('imgs', data=np.asarray(imgs, dtype=np.float64))
    scan = {'NumImages': 1, 'Params': [1, 2, 3]}
    out = _focus_metrics_from_images(
        str(tmp_path), scan, np.array([-1.0, 0.0, 1.0]), np.array(seq_ids))
    assert out is not None and out['type'] == 'focus_metrics'
    assert out['calibration_free'] is True
    sw = out['metrics']['spot_width']
    assert sw['higher_better'] is False and sw['unit'] == 'px'
    vals = np.array(sw['values'], dtype=float)
    # Best focus (smallest spot) is the middle point.
    assert int(np.nanargmin(vals)) == 1
    # Cached on disk after first compute.
    assert (tmp_path / 'focus_metrics.json').is_file()


def test_focus_metrics_none_without_sweep(tmp_path):
    import h5py
    h5 = tmp_path / 'data_20260101_000000.h5'
    with h5py.File(h5, 'w') as f:
        f.create_dataset('imgs', data=np.zeros((3, 40, 40)))
    scan = {'NumImages': 1, 'Params': [1, 1, 1]}   # single point
    assert _focus_metrics_from_images(
        str(tmp_path), scan, np.array([0.0]), np.array([1, 1, 1])) is None


def test_aggregate_focus_metrics_pools_by_value():
    # Two members with overlapping x; spot_width weighted by n_spots.
    m1 = {'type': 'focus_metrics', 'x': [-1.0, 0.0, 1.0], 'x_label': 'z4',
          'n_spots': [10, 10, 10], 'metrics': {
              'spot_width': {'values': [4.0, 2.0, 4.0], 'label': 'w',
                             'unit': 'px', 'higher_better': False}}}
    m2 = {'type': 'focus_metrics', 'x': [0.0, 1.0, 2.0], 'x_label': 'z4',
          'n_spots': [30, 10, 10], 'metrics': {
              'spot_width': {'values': [3.0, 4.0, 5.0], 'label': 'w',
                             'unit': 'px', 'higher_better': False}}}
    agg = _aggregate_focus_metrics([{'seq_specific': m1}, {'seq_specific': m2}])
    assert agg is not None
    assert agg['x'] == [-1.0, 0.0, 1.0, 2.0]          # union of values
    assert agg['n_spots'] == [10, 40, 20, 10]         # summed per value
    vals = agg['metrics']['spot_width']['values']
    # x=-1: only m1 -> 4.0 ; x=0: (10*2 + 30*3)/40 = 2.75 ; x=2: only m2 -> 5.0
    assert vals[0] == pytest.approx(4.0)
    assert vals[1] == pytest.approx(2.75)
    assert vals[3] == pytest.approx(5.0)


def test_aggregate_focus_metrics_none_when_absent():
    assert _aggregate_focus_metrics([{'seq_specific': None}, {}]) is None
