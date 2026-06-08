"""Analysis-dashboard improvements (per-run Analysis view).

Covers the backend pieces added for the dashboard rework:
  * per-shot error stats (per_shot_rate_stats)
  * _summary_stats carrying both per-site and per-shot error families
  * discrimination-infidelity + threshold provenance from the scan .mat
  * sweep n_dims
  * runs_list actual-shot tag formatting

Pure-function tests (no disk / no MATLAB). Run in the yb_analysis env:

    python -m pytest yb_analysis/tests/test_analysis_dashboard_improvements.py -v
"""

import numpy as np
import pytest

from yb_analysis.analysis import probabilities as P
from yb_analysis.analysis import run_analysis as RA
from yb_analysis.analysis import runs_list as RL
from yb_analysis.detection import dynamical_threshold as DT


# --- per_shot_rate_stats ---------------------------------------------------

def _make_logicals(n_sites=6, n_params=3, n_reps=10, seed=0):
    rng = np.random.default_rng(seed)
    l1 = rng.random((n_sites, n_params, n_reps)) < 0.6
    l2 = l1 & (rng.random((n_sites, n_params, n_reps)) < 0.85)
    return l1, l2


def test_per_shot_rate_stats_shapes_and_keys():
    l1, l2 = _make_logicals()
    st = P.per_shot_rate_stats(l1, l2)
    for k in ('loading_mean', 'loading_std_pershot', 'loading_sem_pershot',
              'loading_n_shots', 'survival_mean', 'survival_std_pershot',
              'survival_sem_pershot', 'survival_n_shots'):
        assert k in st, k
        assert len(st[k]) == l1.shape[1]


def test_per_shot_sem_is_std_over_sqrt_n():
    l1, l2 = _make_logicals(n_reps=12, seed=3)
    st = P.per_shot_rate_stats(l1, l2)
    for p in range(l1.shape[1]):
        n = st['survival_n_shots'][p]
        if n and n > 1:
            expected = st['survival_std_pershot'][p] / np.sqrt(n)
            assert st['survival_sem_pershot'][p] == pytest.approx(expected)


def test_per_shot_respects_reps_per_param():
    # Param 2 only ran 4 of its 10 rep slots; the padded (all-False) reps
    # must NOT be counted as genuine zero-loading shots.
    l1, l2 = _make_logicals(n_reps=10, seed=1)
    reps = np.array([10, 10, 4])
    st = P.per_shot_rate_stats(l1, l2, reps_per_param=reps)
    assert st['loading_n_shots'] == [10, 10, 4]
    # survival eligible count never exceeds the real rep count.
    assert st['survival_n_shots'][2] <= 4


def test_per_shot_loading_only_when_no_img2():
    l1, _ = _make_logicals()
    st = P.per_shot_rate_stats(l1, None)
    assert 'loading_mean' in st
    assert 'survival_mean' not in st


# --- _summary_stats: both error families -----------------------------------

def test_summary_stats_has_persite_and_pershot_families():
    l1, l2 = _make_logicals()
    reps = np.full(l1.shape[1], l1.shape[2], dtype=int)
    s = RA._summary_stats(l1, l2, reps)
    # per-site binomial SEM (existing convention)
    assert len(s['survival_sem']) == l1.shape[1]
    assert len(s['loading_rate_sem']) == l1.shape[1]
    # per-shot families (new default)
    assert len(s['survival_sem_pershot']) == l1.shape[1]
    assert len(s['loading_sem_pershot']) == l1.shape[1]
    assert s['survival_n_shots'] is not None


def test_summary_stats_loading_only():
    l1, _ = _make_logicals()
    reps = np.full(l1.shape[1], l1.shape[2], dtype=int)
    s = RA._summary_stats(l1, None, reps)
    assert all(np.isnan(v) for v in s['survival_mean'])
    assert len(s['loading_rate']) == l1.shape[1]
    # per-shot loading still computed; survival per-shot absent (None).
    assert s['loading_sem_pershot'] is not None
    assert s['survival_sem_pershot'] is None


# --- discrimination + threshold provenance ---------------------------------

def _fake_scan():
    n = 8
    rng = np.random.default_rng(2)
    return {
        'initInfidelities': (rng.random(n) * 0.01).tolist(),
        'initThresholds':   (200 + rng.random(n) * 5).tolist(),
        'initGridLocationsX': list(range(n)),
        'initGridLocationsY': list(range(n)),
        'NumImages': 1,
    }


def test_discrimination_info_from_scan_init():
    d = RA._discrimination_info(_fake_scan())
    assert d is not None
    assert d['source'] == 'scan_init'
    assert d['n_sites'] == 8
    assert len(d['per_site']) == 8
    assert 0 <= d['mean_infidelity'] <= 1
    assert len(d['x']) == 8 and len(d['y']) == 8


def test_discrimination_info_none_without_infidelities():
    assert RA._discrimination_info({'foo': 1}) is None


def test_thresholds_info_summary():
    ti = RA._thresholds_info(_fake_scan())
    assert ti is not None
    assert ti['source'] == 'scan_init'
    assert ti['n'] == 8
    assert ti['min'] <= ti['mean'] <= ti['max']
    assert ti['mean_infidelity'] is not None
    assert 'threshold.mat' in ti['source_note']


def test_thresholds_info_none_without_thresholds():
    assert RA._thresholds_info({'foo': 1}) is None


# --- sweep n_dims ----------------------------------------------------------

def test_build_sweep_n_dims_1d():
    sp = np.array([1.0, 2.0, 3.0, 4.0])
    sw = RA._build_sweep({}, sp)
    assert sw['n_dims'] == 1
    assert sw['dims'] == [4]


def test_build_sweep_n_dims_0d_single_point():
    sp = np.array([5.0])
    sw = RA._build_sweep({}, sp)
    # A single-point "sweep" is 0-D as far as the dashboard is concerned.
    assert sw['n_dims'] == 0


# --- recompute discrimination from this run's intensities ------------------

def test_thresholds_infidelities_well_separated_is_low():
    # Two clean populations (empty ~0, atom ~100) → tiny infidelity.
    rng = np.random.default_rng(7)
    n = 400
    empty = rng.normal(0, 3, n)
    atom = rng.normal(100, 5, n)
    col = np.concatenate([empty, atom])
    inten = np.column_stack([col, col])   # 2 identical sites
    thr, inf = DT.thresholds_infidelities_from_intensities(inten)
    assert inf.shape == (2,)
    assert np.all(inf[np.isfinite(inf)] < 0.05)
    # threshold lands between the two means.
    assert np.all((thr > 10) & (thr < 90))


def test_thresholds_infidelities_overlapping_is_higher():
    rng = np.random.default_rng(8)
    n = 400
    a = rng.normal(40, 20, n)
    b = rng.normal(60, 20, n)   # heavy overlap
    col = np.concatenate([a, b])
    inten = np.column_stack([col])
    _, inf_over = DT.thresholds_infidelities_from_intensities(inten)
    # Clean case for comparison.
    clean = np.concatenate([rng.normal(0, 3, n), rng.normal(100, 5, n)])
    _, inf_clean = DT.thresholds_infidelities_from_intensities(
        np.column_stack([clean]))
    assert float(inf_over[0]) > float(inf_clean[0])


def test_thresholds_infidelities_empty_input():
    thr, inf = DT.thresholds_infidelities_from_intensities(np.zeros((0, 0)))
    assert thr.size == 0 and inf.size == 0


def test_fit_run_infidelities_used_threshold():
    # Imaging fidelity = infidelity AT the used threshold (how good the run's
    # bitstrings were). A well-placed cut → low; a cut on the empty peak →
    # ~half the empties misread as atoms → high.
    rng = np.random.default_rng(5)
    n = 400
    col = np.concatenate([rng.normal(0, 3, n), rng.normal(100, 5, n)])
    inten = np.column_stack([col])
    _, opt_inf, used_good = DT.fit_run_infidelities(inten, [50.0])
    assert used_good[0] < 0.05            # good threshold ≈ optimal
    assert opt_inf[0] < 0.05
    _, _, used_bad = DT.fit_run_infidelities(inten, [0.0])
    assert used_bad[0] > 0.3              # threshold on the empty peak → bad
    # imaging_infidelity_at_thresholds delegates to the same fit.
    assert DT.imaging_infidelity_at_thresholds(inten, [0.0])[0] > 0.3


def test_live_scan_curve_target_aware():
    """compute_scan_curve uses per-shot diag targets (TP) when given, matching
    the Analysis tab; falls back to per-site survival otherwise (unchanged)."""
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    sl = [
        (1, np.array([1, 1, 1, 1], bool), np.array([1, 1, 0, 0], bool)),
        (2, np.array([1, 1, 1, 1], bool), np.array([1, 0, 0, 0], bool)),
    ]
    pidx = np.array([1, 2])
    sp = np.array([10.0, 20.0])
    base = compute_scan_curve(sl, pidx, sp, 2)
    assert base.get('target_aware') is None
    assert np.allclose(base['y_mean'], [0.5, 0.25])          # per-site survival
    # targets {0,1}: seq1 both survive→1.0, seq2 only site0→0.5
    tgt = {1: np.array([0, 1]), 2: np.array([0, 1])}
    ta = compute_scan_curve(sl, pidx, sp, 2, seq_targets=tgt)
    assert ta.get('target_aware') is True and ta['mode'] == 'survival'
    assert np.allclose(ta['y_mean'], [1.0, 0.5])
    # No matching shots → fall back (no false target-aware).
    assert compute_scan_curve(sl, pidx, sp, 2,
                              seq_targets={99: np.array([0])}).get('target_aware') is None


def test_per_iteration_fp_excludes_targets_for_rearrange():
    """Per-shot FP must exclude rearrangement target sites: atoms moved INTO
    targets aren't false positives. Without paths → plain empty→occupied FP."""
    # 1 shot, 4 sites. img1 loaded at site 0. img2 has atoms at site 1 (target,
    # legitimately filled) and site 2 (spurious, non-target).
    img1 = np.array([1, 0, 0, 0], dtype=bool)
    img2 = np.array([0, 1, 1, 0], dtype=bool)
    logicals = np.vstack([img1, img2])           # interleaved (NumImages=2)
    bundle = {'logicals': logicals, 'seq_ids': np.array([1]), 'mat_path': None}
    scan = {'NumImages': 2, 'Params': [1]}
    paths = {'paths_per_shot': [{'seq_id': 1, 'target_paired': [1],
                                 'target_site_indices': [1]}]}
    out = RA._per_iteration_time_order(scan, bundle, np.array([1]), None, None,
                                       paths_info=paths, diag_path=None)
    assert out['fp_source'] == 'rearrange'
    # non-target empty sites = {2, 3}; spurious atom only at 2 → 1/2.
    assert abs(out['fp_frac'][0] - 0.5) < 1e-9
    # No paths → naive empty→occupied over {1,2,3} = 2/3.
    out2 = RA._per_iteration_time_order(scan, bundle, np.array([1]), None, None)
    assert out2['fp_source'] == 'all_empty'
    assert abs(out2['fp_frac'][0] - 2.0 / 3.0) < 1e-9


def test_fit_run_infidelities_no_used_thresholds():
    rng = np.random.default_rng(6)
    col = np.concatenate([rng.normal(0, 3, 200), rng.normal(100, 5, 200)])
    opt_thr, opt_inf, used_inf = DT.fit_run_infidelities(
        np.column_stack([col]), None)
    assert opt_inf.size == 1 and np.isnan(used_inf[0])   # no used thr → NaN


# --- avg intensity histogram -----------------------------------------------

def test_intensity_hist_basic():
    rng = np.random.default_rng(11)
    # bimodal pooled intensities, 2 sites x 500 shots
    col = np.concatenate([rng.normal(0, 4, 250), rng.normal(120, 8, 250)])
    bundle = {'intensities': np.column_stack([col, col]), 'two_array': False}
    h = RA._intensity_hist(bundle, n_bins=50)
    assert h is not None
    assert len(h['counts']) == 50
    assert len(h['bin_centers']) == 50
    assert h['n_samples'] == col.size * 2


def test_intensity_hist_none_without_intensities():
    assert RA._intensity_hist({'intensities': None, 'two_array': False}) is None


# --- per-run affine scale (for survival-vs-distance knm conversion) ---------

def test_affine_scale_prefers_run_entry(monkeypatch):
    from yb_analysis.analysis import affine_transform as aff
    fake = {
        'current': {'last_scan_id': '20260603050000', 'scale_x': 3.0,
                    'scale_y': 3.0, 'det': 9.0},
        'history': [
            {'last_scan_id': '20260603043143', 'scale_x': 2.0,
             'scale_y': 2.0, 'det': 4.0},
            {'last_scan_id': '20260601000000', 'scale_x': 1.0,
             'scale_y': 1.0, 'det': 1.0},
        ],
    }
    monkeypatch.setattr(aff, '_read', lambda: fake)
    # exact run match -> that entry's scale (sqrt(2*2)=2), provenance 'run'
    s, src = RA._affine_scale_for_scan('20260603043143')
    assert src == 'run'
    assert s == pytest.approx(2.0)
    # no exact match, falls to most recent <= sid -> the 043143 entry
    s2, src2 = RA._affine_scale_for_scan('20260603044000')
    assert src2 == 'nearest'
    assert s2 == pytest.approx(2.0)
    # older than everything -> current
    s3, src3 = RA._affine_scale_for_scan('20260101000000')
    assert src3 == 'current'
    assert s3 == pytest.approx(3.0)


# --- Details panel: swept params + fixed params + pattern names ------------

def test_run_parameters_includes_swept_and_fixed():
    scan = {
        'SetParams': {'rearrange_kwargs_protocol': 'x', 'LAC_Amp': np.array([0.17])},
        'DefaultParams': {'SLM_VServo': np.array([6.0])},
    }
    sweep_all = {'cols': ['rearrange_kwargs.nsteps'],
                 'values': [[30.0, 50.0, 80.0]]}
    rp = RA._run_parameters(scan, sweep_all)
    swept = [p for p in rp if p['group'] == 'swept']
    assert len(swept) == 1
    assert swept[0]['name'] == 'rearrange_kwargs.nsteps'
    assert swept[0]['value'] == [30.0, 50.0, 80.0]
    # fixed params present, swept ones not duplicated as fixed
    names = {p['name'] for p in rp if p['group'] != 'swept'}
    assert 'LAC_Amp' in names and 'SLM_VServo' in names


def test_loading_pattern_names_parses_json():
    scan = {'imagePatternsJson':
            '[{"name":"33x33_uniform"},{"name":"33x33_uniform"},{"name":"ring"}]'}
    assert RA._loading_pattern_names(scan) == ['33x33_uniform', 'ring']


def test_loading_pattern_names_absent():
    assert RA._loading_pattern_names({}) == []


# --- filesystem analysis cache ---------------------------------------------

def test_analysis_cache_roundtrip_and_keying(tmp_path):
    d = tmp_path
    assert RA._read_analysis_cache(d) is None
    RA._write_analysis_cache(d, {'_version': RA.ANALYSIS_CACHE_VERSION,
                                 'n_shots': 100,
                                 'discrimination_recomputed': {'median_infidelity': 0.001}})
    got = RA._read_analysis_cache(d)
    assert got is not None and got['n_shots'] == 100
    # wrong version -> treated as miss
    RA._write_analysis_cache(d, {'_version': 999, 'n_shots': 100})
    assert RA._read_analysis_cache(d) is None


def test_invalidate_analysis_cache_removes_files(tmp_path):
    (tmp_path / RA.ANALYSIS_CACHE_JSON).write_text('{}')
    (tmp_path / RA.FOCUS_METRICS_JSON).write_text('{}')
    removed = RA.invalidate_analysis_cache(tmp_path)
    assert len(removed) == 2
    assert not (tmp_path / RA.ANALYSIS_CACHE_JSON).exists()
    assert not (tmp_path / RA.FOCUS_METRICS_JSON).exists()


# --- per-step survival-vs-distance: nsteps from the swept axis -------------

def test_seq_to_nsteps_map_from_swept_axis():
    # 2 scan points: nsteps=5 (param 1) and nsteps=10 (param 2). Two axes
    # (nsteps, step_period_ms); the helper must pick the nsteps column.
    scan = {'Params': np.array([1, 2, 1, 2])}   # seq k -> 1-indexed param
    scan_params_full = np.array([[5.0, 1.0], [10.0, 2.0]])
    sweep_all = {'cols': ['rearrange_kwargs.nsteps',
                          'rearrange_kwargs.step_period_ms']}
    seq_ids = np.array([1, 2, 3, 4])
    m = RA._seq_to_nsteps_map(scan, scan_params_full, sweep_all, seq_ids)
    assert m == {1: 5, 2: 10, 3: 5, 4: 10}


def test_seq_to_nsteps_map_no_nsteps_axis():
    scan = {'Params': np.array([1, 2])}
    sp = np.array([[1.0], [2.0]])
    sweep_all = {'cols': ['SomeOther.Param']}
    assert RA._seq_to_nsteps_map(scan, sp, sweep_all, np.array([1, 2])) == {}


# --- calibration-age (staleness) marker ------------------------------------

def test_human_duration_formats():
    assert RA._human_duration(2.5 * 86400) == "2.5 days"
    assert RA._human_duration(3 * 3600) == "3.0 h"
    assert RA._human_duration(120) == "2 min"
    assert RA._human_duration(5) == "5 s"


def test_calibration_age_from_stamped_timestamp(tmp_path):
    sd = tmp_path / "data_20260605_024928"
    sd.mkdir()
    info = RA._calibration_age_info(
        {"calibrationTimestamp": "2026-06-02T17:32:07"}, str(sd))
    assert info["calibration_age_basis"] == "stamped"
    assert info["calibration_age_human"] == "2.4 days"   # ~2.38 days
    assert info["calibration_age_s"] > 0


def test_calibration_age_from_pattern_file_mtime(tmp_path, monkeypatch):
    import os
    sd = tmp_path / "data_20260605_024928"
    sd.mkdir()
    pat = tmp_path / "yb_dashboard_state" / "patterns" / "33x33_uniform"
    pat.mkdir(parents=True)
    thr = pat / "threshold.mat"
    thr.write_text("x")
    # set the calibration file's mtime to a known time 1 day before run start
    import time as _t
    from datetime import datetime
    run = datetime(2026, 6, 5, 2, 49, 28).timestamp()
    os.utime(str(thr), (run - 86400, run - 86400))
    monkeypatch.setattr(RA._yb_cfg, "PATH_PREFIX", str(tmp_path))
    info = RA._calibration_age_info(
        {"calibrationSource": "pattern:33x33_uniform"}, str(sd))
    assert info["calibration_age_basis"] == "file_mtime"
    assert info["calibration_source"] == "pattern:33x33_uniform"
    assert info["calibration_age_human"] == "1.0 days"


def test_calibration_age_none_when_unresolvable(tmp_path):
    sd = tmp_path / "data_20260605_024928"
    sd.mkdir()
    # no timestamp, no source, no day-folder threshold.mat -> empty
    assert RA._calibration_age_info({}, str(sd)) == {}


# --- runs_list actual-shot tag ---------------------------------------------

def test_shots_tag_actual_equals_total():
    assert RL._shots_tag(100, 100) == '100 shots'


def test_shots_tag_aborted_shows_fraction():
    assert RL._shots_tag(42, 100) == '42/100 shots'


def test_shots_tag_actual_only():
    assert RL._shots_tag(42, None) == '42 shots'


def test_shots_tag_none():
    assert RL._shots_tag(None, None) is None
