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


# --- target_restrict toggle (orthogonal to survival_ref) --------------------
#
# 2x2 on the 0d per-shot survival series (a2 = final frame; targets from diag):
#   targets + img1 -> raw TP: a2[tgt].sum() / n_tgt
#   targets + mid  -> verify-conditioned survival at targets:
#                     (mid[tgt] & a2[tgt]).sum() / mid[tgt].sum()   (STIRAP)
#   all sites      -> matched-index survival: (a1&a2).sum() / a1.sum()

def _dm_t(scan_logicals, targets, mid=None, ref='img1', restrict=True, n_img=3):
    dm = _dm(scan_logicals, mid=mid, ref=ref, n_img=n_img)
    dm._seq_targets = dict(targets)
    dm.target_restrict = restrict
    return dm


def test_targets_img1_is_raw_tp():
    # targets = {0,1,2} of 4 sites; final fills 0 and 2 -> TP = 2/3.
    l1 = _b([1, 1, 1, 0]); l2 = _b([1, 0, 1, 0])
    dm = _dm_t([(1, l1, l2)], targets={1: [0, 1, 2]}, ref='img1')
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 2.0 / 3.0) < 1e-9
    assert out['target_aware'] is True
    assert out['cond_mid'] is False


def test_targets_mid_is_verify_conditioned_survival():
    # Same run, condition on the verify frame: of the target sites occupied in
    # verify ({0,1}), fraction still occupied in final ({0}) -> 1/2. Distinct
    # from the raw TP (2/3) above, so the toggle is doing real work.
    l1 = _b([1, 1, 1, 0]); lm = _b([1, 1, 0, 0]); l2 = _b([1, 0, 1, 0])
    dm = _dm_t([(1, l1, l2)], targets={1: [0, 1, 2]}, mid={1: lm}, ref='mid')
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 0.5) < 1e-9
    assert out['target_aware'] is True
    assert out['cond_mid'] is True


def test_restrict_off_uses_all_sites_even_with_targets():
    # target_restrict OFF -> ignore diag targets, matched-index survival over the
    # whole array: loaded={0,1,2}, survive={0,2} -> 2/3, and NOT target-aware.
    l1 = _b([1, 1, 1, 0]); l2 = _b([1, 0, 1, 0])
    dm = _dm_t([(1, l1, l2)], targets={1: [0, 1]}, ref='img1', restrict=False)
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 2.0 / 3.0) < 1e-9
    assert out['target_aware'] is False
    assert out['cond_mid'] is False


def test_restrict_on_but_no_targets_falls_back_to_all_sites():
    # Normal survival scan (no diag targets): target_restrict ON is a no-op ->
    # whole-array survival, never a blank plot.
    l1 = _b([1, 1, 0]); l2 = _b([1, 0, 0])
    dm = _dm_t([(1, l1, l2)], targets={}, ref='img1', restrict=True)
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 0.5) < 1e-9
    assert out['target_aware'] is False


def test_restrict_on_gaps_shots_missing_targets():
    # Run HAS diag targets, but one shot's targets haven't arrived -> that shot
    # gaps (None) rather than silently swapping in an all-sites number.
    l1 = _b([1, 1, 1]); l2 = _b([1, 0, 1])
    dm = _dm_t([(1, l1, l2), (2, l1, l2)], targets={1: [0, 2]}, ref='img1')
    out = dm._per_shot_survival_series()
    assert abs(out['values'][0] - 1.0) < 1e-9   # shot 1: TP over {0,2} = 2/2
    assert out['values'][1] is None             # shot 2: no targets yet


def test_set_target_restrict_normalizes():
    dm = _dm([], ref='img1')
    assert dm.set_target_restrict(False) is False
    assert dm.set_target_restrict(1) is True


# --- toggle DELIVERY: panel -> DataManager ----------------------------------
#
# The reduction tests above all poke `dm.survival_ref` / `dm.target_restrict`
# directly, so they passed while the toggles were BROKEN end-to-end: the
# ControlPanel handlers gated on `_last_real_dm`, which is assigned ONLY on a
# real shot save, so a toggle flipped with no scan running was silently dropped
# (REST returned 200, the spool file drained, the value never reached the DM).
# These tests cover the delivery path itself. 2026-08-06.

class _FakeDM:
    """Minimal stand-in exposing just the setters _apply_view_toggles calls."""
    def __init__(self):
        self.survival_ref = 'img1'
        self.target_restrict = True
        self.site_mask = None
        self.plot_data_calls = 0

    def set_survival_ref(self, ref):
        self.survival_ref = 'mid' if str(ref).lower() == 'mid' else 'img1'
        return self.survival_ref

    def set_target_restrict(self, enabled):
        self.target_restrict = bool(enabled)
        return self.target_restrict

    def set_site_mask_enabled(self, enabled, spec=None):
        self.site_mask = (bool(enabled), spec)

    def get_plot_data(self):
        self.plot_data_calls += 1
        return {'ok': True}


class _FakePanel:
    """ControlPanel's toggle machinery, unbound from Tk."""
    from yb_analysis.gui.control_panel import ControlPanel as _CP
    _apply_view_toggles = _CP._apply_view_toggles
    _repaint_dashboard = _CP._repaint_dashboard

    def __init__(self):
        self._site_mask_enabled = False
        self._site_mask_spec = None
        self._survival_ref = 'img1'
        self._target_restrict = True
        self._dashboard = None


def test_apply_view_toggles_pushes_all_settings():
    p = _FakePanel()
    dm = _FakeDM()
    p._survival_ref = 'mid'
    p._target_restrict = False
    p._apply_view_toggles(dm)
    assert dm.survival_ref == 'mid'
    assert dm.target_restrict is False


def test_apply_view_toggles_pushes_defaults_too():
    # REGRESSION: the old new-scan path skipped values equal to the DM default,
    # so a DM left at 'mid' by a previous scan kept the stale value when the
    # panel had since been set back to 'img1'.
    p = _FakePanel()
    dm = _FakeDM()
    dm.survival_ref = 'mid'          # stale from a previous scan
    dm.target_restrict = False
    p._survival_ref = 'img1'         # panel says img1 / restrict ON
    p._target_restrict = True
    p._apply_view_toggles(dm)
    assert dm.survival_ref == 'img1'
    assert dm.target_restrict is True


def test_apply_view_toggles_none_dm_is_noop():
    _FakePanel()._apply_view_toggles(None)   # must not raise


def test_repaint_dashboard_refreshes_from_dm():
    # The toggle only changes how ALREADY-ACCUMULATED shots are reduced, so the
    # dashboard must repaint immediately rather than waiting for the next shot
    # (which never comes once a scan has finished).
    p = _FakePanel()
    dm = _FakeDM()
    updated = []

    class _FakeDash:
        def update(self, data):
            updated.append(data)

    p._dashboard = _FakeDash()
    p._repaint_dashboard(dm)
    assert dm.plot_data_calls == 1
    assert updated == [{'ok': True}]


def test_repaint_dashboard_survives_no_dashboard_or_dm():
    p = _FakePanel()
    p._repaint_dashboard(None)          # no dm
    p._dashboard = None
    p._repaint_dashboard(_FakeDM())     # no dashboard


def test_peek_and_latest_data_manager_never_construct():
    from yb_analysis.acquisition import data_manager as dmod
    assert dmod.peek_data_manager(987654321) is None
    assert dmod.peek_data_manager(None) is None
    sentinel = object()
    dmod._cache[4242] = sentinel
    try:
        assert dmod.peek_data_manager(4242) is sentinel
        assert dmod.peek_data_manager('4242') is sentinel   # coerces
        assert dmod.latest_data_manager() is sentinel
    finally:
        dmod._cache.pop(4242, None)


# --- scan CURVE honours Cond. verify (the plotted line) ---------------------
#
# Separate code path from the 0d per-shot series above: compute_scan_curve's
# target-aware branch. Until 2026-08-06 it computed raw TP UNCONDITIONALLY and
# never read logic1, so the plotted curve silently ignored the toggle -- on a
# dark-control run it read 0.961 where verify-conditioned was 0.992.

def test_scan_curve_target_branch_honours_cond_mid():
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    import numpy as np
    # 4 sites, targets {0,1,2}. verify(mid) fills {0,1}; final fills {0,2}.
    #   raw TP            = |final & tgt| / |tgt|      = 2/3
    #   verify-conditioned= |mid&final&tgt| / |mid&tgt| = 1/2
    mid = _b([1, 1, 0, 0]); fin = _b([1, 0, 1, 0])
    sl = [(1, mid, fin)]
    pidx = np.array([1]); sp = np.array([0.0])
    tgt = {1: [0, 1, 2]}
    tp = compute_scan_curve(sl, pidx, sp, 2, seq_targets=tgt, cond_mid=False)
    cv = compute_scan_curve(sl, pidx, sp, 2, seq_targets=tgt, cond_mid=True)
    assert tp['target_aware'] is True and cv['target_aware'] is True
    assert abs(float(tp['y_mean'][0]) - 2.0 / 3.0) < 1e-9
    assert abs(float(cv['y_mean'][0]) - 0.5) < 1e-9


def test_scan_curve_cond_mid_defaults_to_raw_tp():
    # Default (no kwarg) must stay raw TP -- the historical behaviour every
    # other caller and stored analysis relies on.
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    import numpy as np
    mid = _b([1, 1, 0, 0]); fin = _b([1, 0, 1, 0])
    out = compute_scan_curve([(1, mid, fin)], np.array([1]), np.array([0.0]), 2,
                             seq_targets={1: [0, 1, 2]})
    assert abs(float(out['y_mean'][0]) - 2.0 / 3.0) < 1e-9


def test_scan_curve_cond_mid_skips_empty_conditioning_shot():
    # A shot whose conditioning frame has NO atoms on the target sites cannot
    # yield a ratio -- it must be dropped, not counted as 0 (which would drag
    # the curve down exactly like the raw-TP fill floor does).
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    import numpy as np
    good_mid = _b([1, 1, 0, 0]); good_fin = _b([1, 1, 0, 0])   # ratio 1.0
    dead_mid = _b([0, 0, 0, 0]); dead_fin = _b([1, 1, 0, 0])   # no denominator
    out = compute_scan_curve([(1, good_mid, good_fin), (2, dead_mid, dead_fin)],
                             np.array([1, 1]), np.array([0.0]), 2,
                             seq_targets={1: [0, 1], 2: [0, 1]}, cond_mid=True)
    assert abs(float(out['y_mean'][0]) - 1.0) < 1e-9
    assert int(out['n_reps'][0]) == 1        # the dead shot was skipped
