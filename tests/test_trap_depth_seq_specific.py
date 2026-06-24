"""Seq-specific 556 push-out trap-depth analysis (run_analysis._trap_depth_from_pushout).

For ANY 556 push-out SURVIVAL scan (NumImages>=2 sweeping the push-out green
frequency) whose line is the trap-shifted |mj|=1 feature, the Sequence-specific
tab now shows a per-site trap-depth histogram + CV -- the trap-depth-feedback
uniformity metric -- regardless of the scan's name. These tests build synthetic
per-site survival curves and assert the analysis fires for the |mj|=1 case and is
correctly SKIPPED for the mj=0 calibration (line at f0), a non-push-out sweep, and
a non-survival scan.

Run in the yb_analysis env:

    python -m pytest yb_analysis/tests/test_trap_depth_seq_specific.py -v
"""

import numpy as np

from yb_analysis.analysis import run_analysis as RA


F0 = 107.7753e6   # mj=0 resonance (expConfig.Resonance556mj0Freq)


def _survival_curve(freqs, center, fwhm=0.6e6, amp=0.8):
    """1 - dip: survival drops to (1-amp) on resonance, recovers off it."""
    half = fwhm / 2
    dip = amp * half ** 2 / ((freqs - center) ** 2 + half ** 2)
    return 1.0 - dip


def _make_logicals(centers, freqs, n_reps=60):
    """(logic1, logic2) for a survival scan: every site loaded in img1, and
    a deterministic round(survival*n_reps) survivors in img2 -- so the recovered
    per-site survival == the injected curve (no RNG)."""
    n_sites, n_params = len(centers), len(freqs)
    logic1 = np.ones((n_sites, n_params, n_reps), dtype=bool)
    logic2 = np.zeros((n_sites, n_params, n_reps), dtype=bool)
    for s, c in enumerate(centers):
        surv = _survival_curve(freqs, c)
        for p in range(n_params):
            k = int(round(surv[p] * n_reps))
            logic2[s, p, :k] = True
    return logic1, logic2


def _scan(num_images=2, f0=F0):
    return {'NumImages': num_images, 'expConfig': {'Resonance556mj0Freq': f0}}


def _sweep(name='Pushout.Green.Freq', size=31):
    return {'cols': [name], 'dims': [size]}


# --------------------------------------------------------------------------- #


def test_trap_depth_fires_for_mj1_pushout(tmp_path):
    """|mj|=1 (line red of f0): a trap_depth panel with a real CV + histogram."""
    freqs = np.linspace(103.5e6, 106.5e6, 31)
    n_sites = 16
    # A deterministic spread of per-site centers -> a spread of trap depths.
    centers = 105.0e6 + np.linspace(-0.4e6, 0.4e6, n_sites)
    logic1, logic2 = _make_logicals(centers, freqs)
    seq_ids = np.arange(1, freqs.size * logic1.shape[2] + 1)

    res = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(), freqs, logic1, logic2, seq_ids,
        sweep_all=_sweep())

    assert res is not None
    assert res['type'] == 'trap_depth'
    assert res['source'] == 'mj1_lightshift'
    # Most sites fit cleanly.
    assert res['n_good'] >= n_sites - 2
    assert res['n_sites'] == n_sites
    assert len(res['depths_uK']) == res['n_good']
    # Trap depth is positive (line red of f0) and CV is finite + sane.
    assert res['mean_depth_uK'] > 0
    assert 0.0 < res['cv'] < 0.5
    assert np.isfinite(res['cv_pct'])
    # Array-averaged line sits red of f0 by ~ (f0 - 105 MHz).
    af = res['array_fit']
    assert af['shift_MHz'] > 0.5
    assert abs(af['center_MHz'] - 105.0) < 0.4
    assert res['f0_MHz'] == F0 / 1e6


def test_trap_depth_cv_tracks_injected_spread(tmp_path):
    """A WIDER center spread -> a larger CV (the uniformity metric responds)."""
    freqs = np.linspace(103.5e6, 106.5e6, 31)
    n_sites = 20
    tight = 105.0e6 + np.linspace(-0.10e6, 0.10e6, n_sites)
    wide = 105.0e6 + np.linspace(-0.50e6, 0.50e6, n_sites)

    r_tight = RA._trap_depth_from_pushout(
        str(tmp_path / 'a'), _scan(), freqs, *_make_logicals(tight, freqs),
        np.arange(1, 100), sweep_all=_sweep())
    r_wide = RA._trap_depth_from_pushout(
        str(tmp_path / 'b'), _scan(), freqs, *_make_logicals(wide, freqs),
        np.arange(1, 100), sweep_all=_sweep())

    assert r_tight is not None and r_wide is not None
    assert r_wide['cv'] > r_tight['cv']


def test_trap_depth_skips_mj0_calibration(tmp_path):
    """mj=0 calibration: the line sits AT f0 (window brackets it), so trap depth
    is not measurable -> no panel."""
    freqs = np.linspace(107.5e6, 107.9e6, 41)
    n_sites = 16
    centers = F0 + np.linspace(-0.01e6, 0.01e6, n_sites)   # essentially at f0
    logic1, logic2 = _make_logicals(centers, freqs, n_reps=60)

    res = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(), freqs, logic1, logic2, np.arange(1, 100),
        sweep_all=_sweep())

    assert res is None


def test_trap_depth_skips_non_pushout_sweep(tmp_path):
    """A survival scan that sweeps something else is not a 556 push-out scan."""
    freqs = np.linspace(103.5e6, 106.5e6, 31)
    centers = 105.0e6 + np.linspace(-0.4e6, 0.4e6, 16)
    logic1, logic2 = _make_logicals(centers, freqs)

    res = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(), freqs, logic1, logic2, np.arange(1, 100),
        sweep_all=_sweep(name='Ramsey.WaitTime'))

    assert res is None


def test_trap_depth_skips_single_image_scan(tmp_path):
    """NumImages=1 (loading scan) has no survival -> no trap-depth panel."""
    freqs = np.linspace(103.5e6, 106.5e6, 31)
    centers = 105.0e6 + np.linspace(-0.4e6, 0.4e6, 16)
    logic1, logic2 = _make_logicals(centers, freqs)

    res = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(num_images=1), freqs, logic1, logic2,
        np.arange(1, 100), sweep_all=_sweep())

    assert res is None


def test_trap_depth_skips_without_f0(tmp_path):
    """No Resonance556mj0Freq in the snapshot -> can't convert -> no panel."""
    freqs = np.linspace(103.5e6, 106.5e6, 31)
    centers = 105.0e6 + np.linspace(-0.4e6, 0.4e6, 16)
    logic1, logic2 = _make_logicals(centers, freqs)
    scan = {'NumImages': 2, 'expConfig': {}}

    res = RA._trap_depth_from_pushout(
        str(tmp_path), scan, freqs, logic1, logic2, np.arange(1, 100),
        sweep_all=_sweep())

    assert res is None


def test_trap_depth_caches_keyed_by_shots(tmp_path):
    """The expensive per-site fit is cached to trap_depth.json keyed by shot
    count; a second call returns the cached result, a new shot count refits."""
    freqs = np.linspace(103.5e6, 106.5e6, 31)
    centers = 105.0e6 + np.linspace(-0.4e6, 0.4e6, 16)
    logic1, logic2 = _make_logicals(centers, freqs)

    seq_ids = np.arange(1, 50)
    r1 = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(), freqs, logic1, logic2, seq_ids,
        sweep_all=_sweep())
    assert r1 is not None
    assert (tmp_path / RA.TRAP_DEPTH_JSON).is_file()
    assert r1['n_shots'] == seq_ids.size

    # Same shot count -> cached (identical result).
    r2 = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(), freqs, logic1, logic2, seq_ids,
        sweep_all=_sweep())
    assert r2['cv'] == r1['cv'] and r2['n_good'] == r1['n_good']

    # More shots -> cache invalidated, recomputed with the new count.
    r3 = RA._trap_depth_from_pushout(
        str(tmp_path), _scan(), freqs, logic1, logic2, np.arange(1, 200),
        sweep_all=_sweep())
    assert r3['n_shots'] == 199


def test_lightshift_positive_red_of_f0():
    """depth(delta_nu) is positive for a line red of f0 and scales with shift."""
    d_small = RA._trap_depth_from_lightshift(2 * (F0 - 106.0e6))
    d_large = RA._trap_depth_from_lightshift(2 * (F0 - 104.0e6))
    assert d_small > 0 and d_large > d_small
    # A line blue of f0 -> negative (filtered out by the good-site mask).
    assert RA._trap_depth_from_lightshift(2 * (F0 - 108.0e6)) < 0
