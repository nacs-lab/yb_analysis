"""Unit tests for the per-shot COMMON-MODE brightness normalization.

The 399 imaging fluorescence breathes shot-to-shot as a GLOBAL multiplicative
gain (CV ~22-24%, correlation time ~2-3 shots) and dominates per-site imaging
fidelity (problem-memory ``open-imaging-common-mode-shot-wobble``). These tests
cover the estimator, the causal EWMA reference, the safety rails, and the
DataManager wiring -- in particular the invariant that ONLY the logicals are
affected: the stored/accumulated intensities stay RAW so the threshold refit
and its degeneracy guards never see a normalized value.

Like test_threshold_guard.py these exercise lightweight
``DataManager.__new__`` instances carrying just the attributes the methods
touch, so no heavy __init__ / file I/O is involved.
"""
import numpy as np
import pytest

from yb_analysis.detection import common_mode as cm
from yb_analysis.acquisition.data_manager import DataManager
from yb_analysis.config import CM_NORM_GAIN_MIN, CM_NORM_GAIN_MAX


MU_E = 200.0
SIG_E = 0.5
SEP = 4.0
SIG_A = 1.0


def _fits(n_sites, mu_e=MU_E, sep=SEP, sig_e=SIG_E, sig_a=SIG_A):
    """Per-site double-Gaussian fits in the package structure."""
    return [{'params': np.array([mu_e, sig_e, 0.4, mu_e + sep, sig_a, 0.6])}
            for _ in range(n_sites)]


def _shot(n_sites, loading, gain, rng, mu_e=MU_E, sep=SEP,
          sig_e=SIG_E, sig_a=SIG_A):
    """One frame of raw intensities with a COMMON-MODE ``gain`` on the atom
    signal only (the background is additive and unaffected -- measured: empty
    std 0.04 ADU, only 0.42-correlated). Returns (intensities, truth)."""
    truth = rng.random(n_sites) < loading
    base = rng.normal(mu_e, sig_e, n_sites)
    atom = gain * (sep + rng.normal(0.0, sig_a, n_sites))
    return base + np.where(truth, atom, 0.0), truth


# ---- A: the estimator --------------------------------------------------------

def test_reference_from_fits_extracts_mu_empty_and_atom_ref():
    ref = cm.reference_from_fits(_fits(8), 8)
    assert ref is not None
    mu_e, atom_ref, good = ref
    assert np.allclose(mu_e, MU_E)
    assert np.allclose(atom_ref, SEP)
    assert good.all()


def test_reference_from_fits_marks_unusable_sites_not_good():
    fits = _fits(6)
    fits[1] = {'params': None}                       # failed fit
    fits[2] = {'params': np.array([MU_E, 0.5, 0.4])}  # truncated
    fits[3] = {'params': np.array([MU_E, 0.5, 0.4, MU_E + 0.2, 0.5, 0.6])}  # no separation
    mu_e, atom_ref, good = cm.reference_from_fits(fits, 6)
    assert list(good) == [True, False, False, False, True, True]
    # A short fit list still yields a full-width reference (missing -> not good)
    mu_e2, _, good2 = cm.reference_from_fits(_fits(3), 6)
    assert good2.size == 6 and list(good2) == [True] * 3 + [False] * 3


def test_reference_from_fits_none_when_no_usable_fit():
    assert cm.reference_from_fits(None, 10) is None
    assert cm.reference_from_fits([], 10) is None
    assert cm.reference_from_fits([{'params': None}] * 5, 5) is None


@pytest.mark.parametrize('gain', [0.7, 1.0, 1.35])
def test_shot_gain_recovers_the_injected_common_mode_gain(gain):
    n = 400
    rng = np.random.default_rng(3)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    I, _ = _shot(n, 0.7, gain, rng)
    g = cm.shot_gain(I, thr, mu_e, atom_ref, good)
    assert g == pytest.approx(gain, abs=0.05)


def test_shot_gain_nan_when_too_few_bright_sites():
    n = 60
    rng = np.random.default_rng(0)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    # A blank / fully pushed-out frame: nothing above threshold -> leave alone.
    blank = np.zeros(n)
    assert not np.isfinite(cm.shot_gain(blank, thr, mu_e, atom_ref, good))
    # Only a handful of atoms (< CM_NORM_MIN_SITES) -> also NaN.
    I, _ = _shot(n, 0.05, 1.0, rng)
    assert not np.isfinite(cm.shot_gain(I, thr, mu_e, atom_ref, good))


def test_shot_gain_is_invariant_to_which_sites_are_filled():
    """The per-site division by that site's OWN calibrated atom signal is what
    makes the estimator immune to the filled SUBSET changing shot to shot (the
    rearrangement case). A per-site brightness gradient must not read as a
    common-mode gain."""
    n = 600
    rng = np.random.default_rng(11)
    sep_site = np.linspace(2.0, 8.0, n)          # strong per-site brightness spread
    fits = [{'params': np.array([MU_E, SIG_E, 0.4, MU_E + sep_site[s], SIG_A, 0.6])}
            for s in range(n)]
    mu_e, atom_ref, good = cm.reference_from_fits(fits, n)
    thr = MU_E + 0.5 * sep_site
    gains = []
    for half in (slice(0, n // 2), slice(n // 2, n)):   # dim half, then bright half
        I = np.full(n, MU_E) + rng.normal(0, SIG_E, n)
        I[half] += 1.2 * sep_site[half]
        gains.append(cm.shot_gain(I, thr, mu_e, atom_ref, good))
    assert gains[0] == pytest.approx(1.2, abs=0.05)
    assert gains[1] == pytest.approx(1.2, abs=0.05)


def test_normalize_restores_the_nominal_scale_and_spares_unusable_sites():
    n = 10
    fits = _fits(n)
    fits[4] = {'params': None}
    mu_e, atom_ref, good = cm.reference_from_fits(fits, n)
    mu_e_safe = np.where(good, mu_e, MU_E)
    raw = mu_e_safe + 1.5 * SEP                  # every site 1.5x too bright
    out = cm.normalize(raw, mu_e, good, 1.5)
    assert np.allclose(out[good], MU_E + SEP)
    assert out[4] == raw[4]                      # untouched, and NOT NaN
    assert np.isfinite(out).all()
    # gain 1 / non-finite / non-positive are exact no-ops (a copy, not a view)
    for g in (1.0, np.nan, 0.0, -2.0):
        same = cm.normalize(raw, mu_e, good, g)
        assert np.array_equal(same, raw) and same is not raw


# ---- B: the causal EWMA reference + rails ------------------------------------

def _tracker(**kw):
    kw.setdefault('warmup', 3)
    return cm.CommonModeTracker(**kw)


def test_tracker_holds_off_during_warmup_then_corrects():
    n = 400
    rng = np.random.default_rng(5)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    t = _tracker(warmup=5)
    for k in range(5):
        I, _ = _shot(n, 0.7, 1.0, rng)
        assert t.observe(0, I, thr, mu_e, atom_ref, good) == 1.0
    I, _ = _shot(n, 0.7, 1.4, rng)
    assert t.observe(0, I, thr, mu_e, atom_ref, good) == pytest.approx(1.4, abs=0.07)


def test_tracker_reference_cancels_a_stale_calibration_offset():
    """The 2198/2078 per-pattern stores are days older than the loading one, so
    the calibrated atom signal can be off by a large CONSTANT factor. The EWMA
    reference must absorb it -> the applied gain settles at 1, not at the
    offset (which would rescale every shot and move every cut)."""
    n = 400
    rng = np.random.default_rng(7)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    t = cm.CommonModeTracker(ema=0.3, warmup=3)
    applied = []
    for _ in range(40):
        I, _ = _shot(n, 0.7, 2.5, rng)          # run is 2.5x brighter than the store
        applied.append(t.observe(0, I, thr, mu_e, atom_ref, good))
    assert np.mean(applied[-20:]) == pytest.approx(1.0, abs=0.05)
    st = t.frame_stats(0)
    assert st['ewma'] == pytest.approx(2.5, rel=0.1)   # the offset lives in the reference
    assert st['n'] == 40 and st['n_applied'] >= 35


def test_tracker_reference_is_debiased_against_a_bad_first_shot():
    """Regression: a plain cold-started EWMA anchors on shot 1 and needs ~1/ema
    shots to forget it. Real data (data_20260727_185942, CV 0.27) opened with a
    0.30 gain, which pinned the reference low and slammed the correction into
    its upper rail for ~50 shots -- it removed only 29% of the wobble instead of
    ~65%. The debiased weight max(ema, 1/n) makes the reference the running mean
    until the exponential memory is filled."""
    n = 400
    rng = np.random.default_rng(31)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.1 * SEP)      # low cut: a very dim shot is still seen
    t = cm.CommonModeTracker(ema=0.02, warmup=3)
    I, _ = _shot(n, 0.7, 0.30, rng)         # pathological opener
    t.observe(0, I, thr, mu_e, atom_ref, good)
    applied = []
    for _ in range(12):
        I, _ = _shot(n, 0.7, 1.0, rng)
        applied.append(t.observe(0, I, thr, mu_e, atom_ref, good))
    # By shot ~6 the reference has absorbed the outlier: gains sit near 1, and
    # nothing is pinned at the rail.
    assert max(applied) < 1.6, applied
    assert np.mean(applied[-5:]) == pytest.approx(1.0, abs=0.15), applied


def test_tracker_clamps_absurd_gains_and_counts_skips():
    n = 400
    rng = np.random.default_rng(9)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    t = cm.CommonModeTracker(ema=0.01, warmup=1)
    for _ in range(6):
        I, _ = _shot(n, 0.7, 1.0, rng)
        t.observe(0, I, thr, mu_e, atom_ref, good)
    I, _ = _shot(n, 0.7, 20.0, rng)             # absurd: clipped, not applied raw
    assert t.observe(0, I, thr, mu_e, atom_ref, good) == pytest.approx(CM_NORM_GAIN_MAX)
    assert t.observe(0, np.zeros(n), thr, mu_e, atom_ref, good) == 1.0   # blank
    assert t.frame_stats(0)['n_skipped'] == 1
    # The LOW rail needs a cut low enough that a very dim shot is still
    # detected at all -- at the usual mid-valley cut a shot that dim simply
    # drops out of the estimator (-> gain 1.0, left alone), which is the
    # intended failure mode.
    t2 = cm.CommonModeTracker(ema=0.01, warmup=1)
    thr_low = np.full(n, MU_E + 0.1 * SEP)
    for _ in range(6):
        I, _ = _shot(n, 0.7, 1.0, rng)
        t2.observe(0, I, thr_low, mu_e, atom_ref, good)
    I, _ = _shot(n, 0.7, 0.3, rng)
    assert t2.observe(0, I, thr_low, mu_e, atom_ref, good) == pytest.approx(CM_NORM_GAIN_MIN)


def test_tracker_keeps_one_reference_per_frame():
    """The 3013 / 2198 / 2078 frames of a two-round rearrangement sit at
    different absolute brightness -- they must not share a reference."""
    n = 400
    rng = np.random.default_rng(13)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    t = cm.CommonModeTracker(ema=0.3, warmup=1)
    for _ in range(20):
        for key, g in ((0, 1.0), (1, 3.0)):
            I, _ = _shot(n, 0.7, g, rng)
            t.observe(key, I, thr, mu_e, atom_ref, good)
    assert t.frame_stats(0)['ewma'] == pytest.approx(1.0, rel=0.1)
    assert t.frame_stats(1)['ewma'] == pytest.approx(3.0, rel=0.1)
    snap = t.snapshot()
    assert set(snap['frames']) == {0, 1}
    assert snap['frames'][0]['cv'] is not None


def test_correction_reduces_misclassifications_end_to_end():
    """The headline claim: with a fixed (raw-fitted) per-site cut, dividing out
    the per-shot gain substantially reduces detection errors. (On the real
    3-pattern run data_20260727_185942 this was per-site median fidelity
    0.9874 -> 0.9960 on the 3013 loading frame and d' 3.74 -> 4.82.)"""
    n, n_shots = 300, 120
    rng = np.random.default_rng(17)
    mu_e, atom_ref, good = cm.reference_from_fits(_fits(n), n)
    thr = np.full(n, MU_E + 0.5 * SEP)
    t = cm.CommonModeTracker(ema=0.02, warmup=5)
    err_raw = err_norm = 0
    gain = 1.0
    for k in range(n_shots):
        # AR(1) wobble: CV ~24%, lag-1 correlation 0.7 (the measured behaviour)
        gain = 1.0 + 0.7 * (gain - 1.0) + rng.normal(0.0, 0.24 * np.sqrt(1 - 0.49))
        I, truth = _shot(n, 0.7, max(gain, 0.05), rng)
        g = t.observe(0, I, thr, mu_e, atom_ref, good)
        err_raw += int(np.sum((I > thr) != truth))
        err_norm += int(np.sum((cm.normalize(I, mu_e, good, g) > thr) != truth))
    assert err_norm < 0.7 * err_raw, (err_raw, err_norm)


# ---- C: DataManager wiring ---------------------------------------------------

def _stub(n_sites, fits=None, mid_fits=None, img2_fits=None):
    """A bare DataManager carrying only what the common-mode path reads."""
    d = DataManager.__new__(DataManager)
    d.num_sites = n_sites
    d.live_gauss_fits = None
    d.live_gauss_fits_img2 = None
    d.loaded_gauss_fits = fits
    d.loaded_gauss_fits_mid = mid_fits
    d.loaded_gauss_fits_img2 = img2_fits
    d.num_images_per_seq = 3
    d._pattern_names = {0: 'tri_3013', 1: 'kagome_2198', 2: 'kagome_2078'}
    d._cm = cm.CommonModeTracker(ema=0.3, warmup=1)
    d._cm_ref_cache = {}
    d._cm_logged = set()
    return d


def test_effective_gauss_fits_follows_effective_thresholds_precedence():
    d = _stub(4, fits=_fits(4), mid_fits=_fits(4), img2_fits=_fits(4))
    # No live fit yet -> each frame gets its own stored fits
    assert d._effective_gauss_fits('kagome_2198', d.loaded_gauss_fits_mid) is d.loaded_gauss_fits_mid
    assert d._effective_gauss_fits('kagome_2078', d.loaded_gauss_fits_img2) is d.loaded_gauss_fits_img2
    # img1's live refit wins for the LOADING pattern only
    d.live_gauss_fits = _fits(4)
    assert d._effective_gauss_fits('tri_3013', d.loaded_gauss_fits) is d.live_gauss_fits
    assert d._effective_gauss_fits('kagome_2198', d.loaded_gauss_fits_mid) is d.loaded_gauss_fits_mid
    # img2's own live refit wins for the img2 pattern
    d.live_gauss_fits_img2 = _fits(4)
    assert d._effective_gauss_fits('kagome_2078', d.loaded_gauss_fits_img2) is d.live_gauss_fits_img2


def test_cm_detect_flips_marginal_sites_and_leaves_intensities_raw():
    n = 400
    rng = np.random.default_rng(21)
    d = _stub(n, fits=_fits(n))
    thr = np.full(n, MU_E + 0.5 * SEP)
    for _ in range(5):                             # warm the reference at gain 1
        I, _ = _shot(n, 0.7, 1.0, rng)
        d._cm_detect(0, I, thr, d.loaded_gauss_fits, I > thr)
    I, truth = _shot(n, 0.7, 0.55, rng)            # a DIM shot: atoms fall under the cut
    raw_logicals = I > thr
    raw_copy = I.copy()
    out = d._cm_detect(0, I, thr, d.loaded_gauss_fits, raw_logicals)
    assert np.array_equal(I, raw_copy)             # intensities never mutated
    assert out.sum() > raw_logicals.sum()          # dim atoms recovered
    assert np.sum(out != truth) < np.sum(raw_logicals != truth)


def test_cm_detect_no_op_when_disabled_or_bypassed_but_still_tracks():
    n = 400
    rng = np.random.default_rng(23)
    d = _stub(n, fits=_fits(n))
    thr = np.full(n, MU_E + 0.5 * SEP)
    for _ in range(5):
        I, _ = _shot(n, 0.7, 1.0, rng)
        d._cm_detect(0, I, thr, d.loaded_gauss_fits, I > thr)
    I, _ = _shot(n, 0.7, 0.6, rng)
    raw = I > thr
    # bypass (the img2 spot-shape GMM decided this frame)
    assert np.array_equal(d._cm_detect(0, I, thr, d.loaded_gauss_fits, raw,
                                       bypass=True), raw)
    # global kill switch
    assert cm.set_enabled(False) is False
    try:
        assert np.array_equal(d._cm_detect(0, I, thr, d.loaded_gauss_fits, raw), raw)
    finally:
        cm.set_enabled(True)
    # ...but the wobble is still measured either way
    st = d._cm.frame_stats(0)
    assert st['n'] == 7 and st['cv'] is not None


def test_cm_detect_no_op_without_usable_fits():
    n = 400
    rng = np.random.default_rng(27)
    d = _stub(n, fits=None)
    thr = np.full(n, MU_E + 0.5 * SEP)
    I, _ = _shot(n, 0.7, 0.6, rng)
    raw = I > thr
    for _ in range(4):
        assert np.array_equal(d._cm_detect(0, I, thr, None, raw), raw)
    assert d._cm.frame_stats(0) is None            # nothing tracked, nothing applied
    # A width mismatch (thresholds vs intensities) is also a clean no-op.
    d2 = _stub(n, fits=_fits(n))
    assert np.array_equal(
        d2._cm_detect(0, I, np.full(n - 1, MU_E), d2.loaded_gauss_fits, raw), raw)


def test_cm_reference_cache_rebuilds_on_a_live_refit():
    n = 32
    d = _stub(n, fits=_fits(n))
    r1 = d._cm_reference(0, d.loaded_gauss_fits, n)
    assert d._cm_reference(0, d.loaded_gauss_fits, n) is r1       # cached
    new_fits = _fits(n, sep=9.0)
    r2 = d._cm_reference(0, new_fits, n)
    assert r2 is not r1 and np.allclose(r2[1], 9.0)
