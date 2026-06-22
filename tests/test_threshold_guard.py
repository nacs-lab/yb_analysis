"""Unit tests for the structural detection-threshold guard (data_manager).

Covers the failure that corrupted the 33x33_uniform per-pattern store on
2026-06-09: a low-loading / near-unimodal run produced a degenerate full
Gaussian fit (all per-site thresholds collapsed onto the empty peak), and the
unbounded cheap refit then exploded the per-site spread and saved it.

The methods are exercised on lightweight ``DataManager.__new__`` instances with
just the attributes each method touches, so we test the guard logic directly
without the heavy __init__ / file I/O.
"""
import numpy as np
import pytest

import yb_analysis.acquisition.data_manager as dm
from yb_analysis.acquisition.data_manager import DataManager
from yb_analysis.config import (
    THRES_LIVE_VALLEY_MARGIN, THRES_FIT_MIN_SEP_FRAC, THRES_LOADED_MAX_SPREAD,
    THRES_ACCUM_CROSS_RUN_WINDOW_S, THRES_ACCUM_CROSS_RUN_MAX, UPDATE_THRES_INTERVAL,
)


def _bimodal(n_sites, n_shots, loading, mu_e=200.0, sig_e=0.8,
             mu_a=207.0, sig_a=1.0, seed=0):
    """Synthetic per-site frame-0 intensities: (n_shots, n_sites)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_shots, n_sites), dtype=np.float64)
    for s in range(n_sites):
        loaded = rng.random(n_shots) < loading
        out[:, s] = np.where(loaded,
                             rng.normal(mu_a, sig_a, n_shots),
                             rng.normal(mu_e, sig_e, n_shots))
    return out


def _stub(num_sites):
    """A bare DataManager carrying only what the guard methods read."""
    d = DataManager.__new__(DataManager)
    d.num_sites = num_sites
    d.live_hist_data = None
    d.live_thresholds = None
    d.loaded_thresholds = np.full(num_sites, 203.0)
    d.loaded_gauss_fits = None
    d._pattern_names = {0: 'testpat'}
    d._thr_place_ratio = None
    d._thr_clamp_lo = d._thr_clamp_hi = None
    d._thr_fit_anchor = None
    d._thr_fit_med_below = None
    d._thr_fit_med_above = None
    d._thr_has_accepted_fit = False
    d._threshold_health = {'state': 'init'}
    d.config = {}
    d._save_threshold = lambda: None
    d._log_threshold_update = lambda *a, **k: None
    return d


# ---- A1: full-fit acceptance / rejection -------------------------------------

def test_full_fit_accepts_clean_bimodal():
    d = _stub(40)
    data = _bimodal(40, 250, loading=0.6)
    fits, thres, inf = d._fit_gaussians(data)
    ok, reason, stats = d._validate_full_fit(fits, thres, data)
    assert ok, reason
    assert stats['frac_separated'] >= THRES_FIT_MIN_SEP_FRAC
    # thresholds land in the valley between the peaks
    assert 200.5 < np.median(thres) < 206.5


def test_full_fit_rejects_low_loading_unimodal():
    d = _stub(40)
    data = _bimodal(40, 250, loading=0.02)   # near-unimodal: ~5 atoms / 250
    fits, thres, inf = d._fit_gaussians(data)
    ok, reason, stats = d._validate_full_fit(fits, thres, data)
    assert not ok
    assert 'separated' in reason or 'loaded fraction' in reason


# ---- A1b: blank / dropped-frame rejection (2026-06-19 corruption) ------------

def test_is_blank_intensities():
    assert dm._is_blank_intensities(np.zeros(50))            # whole-frame ~0 (dropped)
    assert dm._is_blank_intensities(np.array([]))            # empty
    assert dm._is_blank_intensities(np.full(50, np.nan))     # all-NaN frame
    assert not dm._is_blank_intensities(np.full(50, 200.0))  # real pedestal frame
    # A legitimately-EMPTY atom shot still sits on the camera pedestal -> not blank.
    assert not dm._is_blank_intensities(np.full(50, 200.0) + np.zeros(50))


def test_full_fit_rejects_blank_contaminated():
    """The 2026-06-19 failure: clean bimodal shots polluted by whole-frame-blank
    shots (a 0-vs-pedestal toggle the fit reads as a flawless empty/atom split).
    Such a fit clears the separation + loading checks, so the blank backstop must
    catch it."""
    d = _stub(40)
    bright = _bimodal(40, 138, loading=0.6)        # real empty~200 / loaded~207
    blank = np.zeros((62, 40))                      # 62 dropped frames (== the live case)
    data = np.vstack([bright, blank])
    fits, thres, inf = d._fit_gaussians(data)
    ok, reason, stats = d._validate_full_fit(fits, thres, data)
    assert not ok
    assert 'blank' in reason
    assert stats['blank_frac'] == pytest.approx(62 / 200, abs=0.01)


def test_compute_hist_data_robust_to_hot_pixel():
    """A single hot-pixel / outlier reading must not stretch the 50-bin range so
    wide the empty/atom doublet collapses into one bar."""
    data = _bimodal(1, 300, loading=0.6)
    data[0, 0] = 5000.0                             # one hot reading at site 0
    hd = DataManager._compute_hist_data(data, 1)
    bc = hd[0]['bin_centers']
    bin_width = bc[1] - bc[0]
    assert bin_width < 1.0                           # robust clip keeps bins tight
    # without clipping this would be ~5000/50 = 100 ADU/bin


# ---- A2: cheap-refit anchor gate + valley clamp ------------------------------

def test_cheap_refit_holds_without_anchor():
    d = _stub(40)
    d._intensity_accum = list(_bimodal(40, 120, loading=0.6))
    d._thr_has_accepted_fit = False        # no accepted fit yet
    d._update_thresholds_live_cheap()
    assert d.live_thresholds is None       # held, did not fly blind


def test_cheap_refit_clamps_inside_valley():
    d = _stub(40)
    # Accepted-fit anchor: peaks 200 / 207. STALE-LOW stored references (197 / 204
    # vs the real ~200 / ~207) make BOTH peak-drift terms read "+3" => candidate
    # 206, above the valley clamp's hi -> the clamp must catch it.
    d._thr_has_accepted_fit = True
    d._thr_place_ratio = np.full(40, 0.5)
    lo = np.full(40, 200.0 + THRES_LIVE_VALLEY_MARGIN * 7.0)
    hi = np.full(40, 207.0 - THRES_LIVE_VALLEY_MARGIN * 7.0)
    d._thr_clamp_lo, d._thr_clamp_hi = lo, hi
    d._thr_fit_anchor = np.full(40, 203.0)             # last full-fit cut
    d._thr_fit_med_below = np.full(40, 197.0)          # stale-low empty reference
    d._thr_fit_med_above = np.full(40, 204.0)          # stale-low atom reference
    d.loaded_thresholds = np.full(40, 203.0)           # start in the valley
    data = _bimodal(40, 120, loading=0.6)              # real peaks ~200 / ~207
    for _ in range(100):
        d._intensity_accum = list(data)
        d._update_thresholds_live_cheap()
    thr = np.asarray(d.live_thresholds)
    # The +3 ADU overshoot is caught by the valley clamp: every site stays at/below
    # hi (never on or past the loaded peak).
    assert np.all(thr <= hi + 1e-9)
    assert np.all(thr >= lo - 1e-9)
    assert thr.max() > 205.0     # it DID rise toward the clamp (clamp is doing work)
    assert thr.max() < 207.0     # but never reached the loaded peak


def test_cheap_refit_no_ratchet_on_stable_bimodal():
    """Fixed-anchor split (vs the old split-by-drifting-cut): a STABLE, densely
    loaded bimodal must not ratchet the threshold away from the valley over many
    cheap updates. Regression for the +1-2 ADU between-fit drift seen on
    33x33_uniform (corrected each full fit)."""
    d = _stub(40)
    d._thr_has_accepted_fit = True
    d._thr_place_ratio = np.full(40, 0.5)
    sep = 7.0
    d._thr_clamp_lo = np.full(40, 200.0 + THRES_LIVE_VALLEY_MARGIN * sep)
    d._thr_clamp_hi = np.full(40, 207.0 - THRES_LIVE_VALLEY_MARGIN * sep)
    d._thr_fit_anchor = np.full(40, 203.5)        # last full-fit cut (the valley)
    d._thr_fit_med_below = np.full(40, 200.0)     # empty/atom references (match data)
    d._thr_fit_med_above = np.full(40, 207.0)
    d.loaded_thresholds = np.full(40, 203.5)
    data = _bimodal(40, 120, loading=0.75)        # dense loaded peak (ratchet-prone)
    means = []
    for _ in range(300):
        d._intensity_accum = list(data)
        d._update_thresholds_live_cheap()
        means.append(float(np.nanmean(d.live_thresholds)))
    m = np.asarray(means)
    assert abs(m[-1] - 203.5) < 1.0               # stayed near the valley anchor
    assert abs(m[-1] - m[100]) < 0.25             # no sustained creep after settling


# ---- A3: on-load validation --------------------------------------------------

def test_loaded_validation_flags_degraded_spread():
    d = _stub(40)
    d.loaded_thresholds = np.linspace(199.0, 211.0, 40)   # spread ~3.5 ADU
    d._validate_loaded_thresholds()
    assert d._threshold_health['state'] == 'degraded'
    assert not d._thr_has_accepted_fit
    assert d._threshold_health['spread'] > THRES_LOADED_MAX_SPREAD


def test_loaded_validation_primes_anchor_when_healthy():
    d = _stub(40)
    d.loaded_thresholds = np.full(40, 202.5) + np.random.default_rng(1).normal(0, 0.2, 40)
    d.loaded_gauss_fits = [{'params': np.array([200., 0.8, 0.5, 207., 1.0, 0.5])}
                           for _ in range(40)]
    d._validate_loaded_thresholds()
    assert d._threshold_health['state'] == 'ok'
    assert d._thr_has_accepted_fit
    assert d._thr_clamp_lo is not None and d._thr_clamp_hi is not None


def test_loaded_validation_unknown_pattern_warns_but_ok():
    d = _stub(40)
    d._pattern_names = {}                                  # no loading pattern
    d.loaded_thresholds = np.full(40, 202.5)
    d._validate_loaded_thresholds()
    assert d._threshold_health['state'] == 'unknown_pattern'


# ---- A4: cross-run accumulation store ----------------------------------------

def test_pattern_accum_prunes_by_age_and_cap():
    name = 'accum_test_pat'
    with dm._pattern_accum_lock:
        dm._pattern_accum.pop(name, None)
    pa = dm._get_pattern_accum(name, 10)
    now = 1_000_000.0
    vec = np.zeros(10)
    # one stale entry (older than the window) + one fresh
    pa['entries'].append((now - THRES_ACCUM_CROSS_RUN_WINDOW_S - 10, vec))
    pa['entries'].append((now, vec))
    dm._prune_pattern_accum(pa, now)
    assert len(pa['entries']) == 1            # stale dropped

    # length cap
    for _ in range(THRES_ACCUM_CROSS_RUN_MAX + 50):
        pa['entries'].append((now, vec))
    dm._prune_pattern_accum(pa, now)
    assert len(pa['entries']) <= THRES_ACCUM_CROSS_RUN_MAX


def test_pattern_accum_resets_on_site_count_change():
    name = 'accum_resize_pat'
    with dm._pattern_accum_lock:
        dm._pattern_accum.pop(name, None)
    pa = dm._get_pattern_accum(name, 10)
    pa['entries'].append((1.0, np.zeros(10)))
    pa2 = dm._get_pattern_accum(name, 20)     # different grid -> fresh holder
    assert pa2['num_sites'] == 20
    assert len(pa2['entries']) == 0


# ---- A6: img2 independent refit (distinct loading pattern) -------------------

def _stub_img2(num_sites=8, num_sites_img2=12, same_pattern=False):
    """A bare DataManager carrying the img1 + img2 threshold state the img2
    refit / effective-threshold logic reads."""
    d = _stub(num_sites)
    d.num_images_per_seq = 2
    d.is_two_array = True
    d.scan_id = 20260611000000
    d._seq_total = 0
    d.num_sites_img2 = num_sites if same_pattern else num_sites_img2
    d._pattern_names = {0: 'pat_img1',
                        1: ('pat_img1' if same_pattern else 'pat_img2')}
    d.live_thresholds_img2 = None
    d.live_infidelities_img2 = None
    d.live_gauss_fits_img2 = None
    d.live_hist_data_img2 = None
    d.loaded_thresholds_img2 = np.full(d.num_sites_img2, 203.0)
    d.loaded_infidelities_img2 = np.full(d.num_sites_img2, np.nan)
    d.loaded_gauss_fits_img2 = None
    d._intensity_accum_img2 = []
    d._img_cnt_refit_img2 = 0
    d._thr_has_accepted_fit_img2 = False
    d._threshold_health_img2 = {'state': 'init'}
    d._save_threshold_img2 = lambda: None
    d._log_threshold_rejected = lambda *a, **k: None
    return d


def test_img2_refit_active_only_when_distinct():
    assert _stub_img2(same_pattern=False)._img2_refit_active()      # distinct -> own refit
    assert not _stub_img2(same_pattern=True)._img2_refit_active()   # same -> shares img1
    d = _stub_img2(same_pattern=False); d.is_two_array = False
    assert not d._img2_refit_active()                               # no img2 frame


def test_img2_refit_accepts_and_sets_live_thresholds():
    d = _stub_img2(num_sites=8, num_sites_img2=12)
    d._intensity_accum_img2 = list(_bimodal(12, 250, loading=0.6, seed=3))
    d._img_cnt_refit_img2 = UPDATE_THRES_INTERVAL
    d._maybe_refit_img2()
    assert d.live_thresholds_img2 is not None and len(d.live_thresholds_img2) == 12
    assert 200.5 < np.median(d.live_thresholds_img2) < 206.5   # lands in the valley
    assert d._threshold_health_img2['state'] == 'ok'
    # img2 now detects with its OWN live thresholds; img1 untouched
    assert d._effective_thresholds('pat_img2', d.loaded_thresholds_img2) is d.live_thresholds_img2
    assert d.live_thresholds is None


def test_img2_refit_rejects_degenerate_and_holds():
    d = _stub_img2(num_sites=8, num_sites_img2=12)
    d._intensity_accum_img2 = list(_bimodal(12, 250, loading=0.02, seed=4))  # near-unimodal
    d._img_cnt_refit_img2 = UPDATE_THRES_INTERVAL
    d._maybe_refit_img2()
    assert d.live_thresholds_img2 is None                  # held, not applied
    assert d._threshold_health_img2['state'] == 'degraded'
    # detection falls back to the loaded img2 store
    assert d._effective_thresholds('pat_img2', d.loaded_thresholds_img2) is d.loaded_thresholds_img2


def test_img2_shares_img1_refit_when_same_pattern():
    d = _stub_img2(same_pattern=True)
    d.live_thresholds = np.full(d.num_sites, 204.0)        # img1's live refit
    # same pattern as frame-0 -> uses img1's live thresholds, no separate img2 fit
    assert d._effective_thresholds('pat_img1', d.loaded_thresholds_img2) is d.live_thresholds


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
