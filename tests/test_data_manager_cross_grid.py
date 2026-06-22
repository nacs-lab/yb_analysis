"""Cross-grid (init pattern != target pattern) live-pipeline safety.

A rearrangement scan whose initial loading pattern and final target pattern
differ detects img1 on the loading grid and img2 on the target grid, so the
two per-shot logicals have DIFFERENT site counts. ``a1 & a2`` (matched-index
per-site survival) is then undefined and raises a numpy broadcast ValueError.

Before the guard, ``DataManager._per_shot_survival_series`` raised inside
``get_plot_data()`` every cycle, so the live dashboard ``_process_once`` aborted
repeatedly -- the live display (and the per-batch save that runs after it)
fell behind the backend by however many shots elapsed between successful
cycles. This locks in: no crash on mismatched grids, target-aware TP when diag
targets are known, and the unchanged same-grid per-site survival.
"""

import numpy as np

from yb_analysis.acquisition.data_manager import DataManager


def _dm(scan_logicals, seq_targets):
    # Bypass the heavy __init__ (disk calibration preload) -- the method only
    # reads num_images_per_seq, _scan_logicals, _seq_targets.
    dm = object.__new__(DataManager)
    dm.num_images_per_seq = 2
    dm._scan_logicals = scan_logicals
    dm._seq_targets = seq_targets
    return dm


def _b(a):
    return np.asarray(a, dtype=bool)


def test_cross_grid_no_targets_is_none_not_crash():
    """img1 (6 sites) vs img2 (4 sites): the per-site fallback would
    broadcast-crash. A shot with no diag targets must yield None (not raise);
    a later shot whose targets ARE known fills in via the target-aware path."""
    i1 = _b([1, 1, 1, 1, 1, 0])
    dm = _dm(
        [(1, i1, _b([1, 1, 1, 0])),     # no targets -> None (was a ValueError)
         (2, i1, _b([1, 1, 1, 0]))],    # targets known -> target-aware TP
        seq_targets={2: [0, 1, 2, 3]})
    out = dm._per_shot_survival_series()      # must not raise
    assert out is not None
    assert out['values'][0] is None
    assert abs(out['values'][1] - 0.75) < 1e-9   # 3 of 4 target sites occupied
    assert out['target_aware'] is True


def test_cross_grid_with_diag_targets_is_target_aware():
    """When the shot's diag targets are known, survival = TP at those img2
    sites (works regardless of the img1/img2 grid-size mismatch)."""
    i1 = _b([1, 1, 1, 1, 1, 0])
    dm = _dm([(1, i1, _b([1, 1, 0, 0]))], seq_targets={1: [0, 1, 2, 3]})
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 0.5) < 1e-9   # 2 of 4 target sites occupied
    assert out['target_aware'] is True


def test_same_grid_per_site_survival_unchanged():
    """Same-grid run (img1 and img2 same site count, no targets) keeps the
    classic matched-index per-site survival = survived / loaded."""
    dm = _dm([(1, _b([1, 1, 1, 0]), _b([1, 1, 0, 0]))], seq_targets={})
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - (2 / 3)) < 1e-9
    assert out['target_aware'] is False
