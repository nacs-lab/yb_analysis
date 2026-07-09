"""Live survival conditioning-frame toggle (survival_ref: img1 | mid).

For NumImages >= 3 (rearrange-verify) scans the DataManager keeps each shot's
MIDDLE (verify) frame bits in ``_scan_mid_logicals``; with
``set_survival_ref('mid')`` the live survival consumers (scan curve + per-shot
series) receive (seq_id, mid, final) triples via ``_effective_scan_logicals``
instead of (seq_id, img1, final). Loading readouts are unaffected (they never
go through the substitution). Locks in:

  1. default = img1 (raw accumulator returned untouched, identity)
  2. 'mid' substitutes per shot, with per-shot fallback to img1 when the mid
     bits are missing
  3. _per_shot_survival_series conditions on the substituted frame
  4. bare/partial DMs (no new attrs) don't crash (getattr defaults)
"""

import numpy as np

from yb_analysis.acquisition.data_manager import DataManager


def _b(a):
    return np.asarray(a, dtype=bool)


def _dm(scan_logicals, mid=None, ref='img1', n_img=3):
    dm = object.__new__(DataManager)
    dm.num_images_per_seq = n_img
    dm._scan_logicals = scan_logicals
    dm._seq_targets = {}
    dm._scan_mid_logicals = dict(mid or {})
    dm.survival_ref = ref
    return dm


def test_default_img1_is_identity():
    sl = [(1, _b([1, 1, 0]), _b([1, 0, 0]))]
    dm = _dm(sl, mid={1: _b([0, 1, 0])}, ref='img1')
    assert dm._effective_scan_logicals() is sl   # untouched accumulator


def test_mid_substitutes_conditioning_frame():
    l1 = _b([1, 1, 1, 0])          # loading: sites 0,1,2
    lm = _b([0, 1, 1, 0])          # verify:  sites 1,2 (site0 discarded by rearrange)
    l2 = _b([1, 1, 0, 0])          # final:   sites 0,1
    dm = _dm([(1, l1, l2)], mid={1: lm}, ref='mid')
    eff = dm._effective_scan_logicals()
    assert (eff[0][1] == lm).all()
    assert (eff[0][2] == l2).all()
    # Per-shot survival now conditions on the verify frame: loaded={1,2},
    # survive={1} -> 0.5 (img1 conditioning would give 2/3).
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 0.5) < 1e-9
    dm.survival_ref = 'img1'
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 2.0 / 3.0) < 1e-9


def test_mid_missing_falls_back_per_shot():
    l1 = _b([1, 1]); l2 = _b([1, 0])
    dm = _dm([(1, l1, l2), (2, l1, l2)], mid={2: _b([0, 1])}, ref='mid')
    eff = dm._effective_scan_logicals()
    assert (eff[0][1] == l1).all()               # shot 1: no mid bits -> img1
    assert (eff[1][1] == _b([0, 1])).all()       # shot 2: mid bits used


def test_set_survival_ref_normalizes():
    dm = _dm([], ref='img1')
    assert dm.set_survival_ref('MID') == 'mid'
    assert dm.set_survival_ref('bogus') == 'img1'


def test_bare_dm_without_new_attrs_is_safe():
    """Partially-built DMs (older tests, dummy display path) lack survival_ref /
    _scan_mid_logicals -- the helper must default to the raw accumulator."""
    dm = object.__new__(DataManager)
    dm.num_images_per_seq = 2
    sl = [(1, _b([1, 1]), _b([1, 0]))]
    dm._scan_logicals = sl
    dm._seq_targets = {}
    assert dm._effective_scan_logicals() is sl
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 0.5) < 1e-9
