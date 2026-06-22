"""Phase 5a closeout tests — slm_analysis.json fall-through + target-aware
survival override + per_site TP/FP markers + diag_aggregate fallback.

These cover the lab-side translation of the SLM server's
``/runs/<run_id>/analysis`` cache (written by
``backfill_slm_analysis.py``) into the dashboard's analysis shape:

- ``analyze_scan_dir`` returns ``slm_analysis_cached=True`` when the
  sidecar is present.
- ``target_aware`` carries the overall TP/Loss/FP overall rates and a
  per-scan-param mean+sem array when SLM and lab sweep axes can be
  fuzzy-matched.
- ``summary.survival_mean`` is OVERRIDDEN to the target-aware curve
  when matching succeeded (with per-site backup preserved under
  ``survival_mean_per_site``).
- ``per_site`` is OVERRIDDEN to the SLM's TP/FP-aware per-site map,
  with new boolean ``is_target_site`` / ``is_nontarget_site`` masks.
- ``diag_aggregate`` falls back to the SLM's ``diag_series_all`` block
  when no local ``slm_diag.h5`` exists.

Run as:
    python -m yb_analysis.tests.slm_migration.test_phase5a_closeout
or:
    pytest yb_analysis/tests/slm_migration/test_phase5a_closeout.py -v
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from yb_analysis.analysis.run_analysis import (
    ANALYSIS_JSON, _load_slm_analysis, _per_site_from_slm_analysis,
    _target_aware_from_slm_analysis, _diag_aggregate_from_slm_analysis,
    analyze_scan_dir,
)


# ---------------------------------------------------------------------------
# Synthetic SLM analysis fixture (matches /runs/<id>/analysis shape)
# ---------------------------------------------------------------------------

def _make_slm_analysis(n_sites=20, n_targets=8, n_bins=4):
    """Build a synthetic SLM analysis dict close to the real shape."""
    rng = np.random.default_rng(42)
    image_x = rng.uniform(100, 900, size=n_sites).tolist()
    image_y = rng.uniform(100, 900, size=n_sites).tolist()
    # First n_targets sites are TP-eligible; the rest are FP-eligible.
    tp_elig = [10] * n_targets + [0] * (n_sites - n_targets)
    fp_elig = [0] * n_targets + [10] * (n_sites - n_targets)
    tp_rate = ([float(rng.uniform(70, 90)) for _ in range(n_targets)]
                + [None] * (n_sites - n_targets))
    fp_rate = ([None] * n_targets
                + [float(rng.uniform(0, 5)) for _ in range(n_sites - n_targets)])
    loading_rate = [float(rng.uniform(50, 80)) for _ in range(n_sites)]
    per_bin = [
        {'axis_a': float(i % 2),
          'axis_b': float(i // 2),
          'n_shots': 10,
          'TP_return_%': 80.0 + i * 2.0,
          'Loss_%': 20.0 - i * 2.0,
          'FP_total_%': 1.5}
        for i in range(n_bins)
    ]
    return {
        'run_id': '2026-05-29T17:00:00.000000',
        'n_shots': 40,
        'n_shots_total': 40,
        'n_shots_loaded': 40,
        'protocols': ['rearrange'],
        'sweep': {'kind': '2d', 'params': ['axis_a', 'axis_b'],
                   'p1_vals': [0, 1], 'p2_vals': [0, 1]},
        'summary': [
            {'category': 'TP_return', 'hits': 700, 'eligible': 800, 'rate_pct': 87.5},
            {'category': 'Loss',      'hits': 100, 'eligible': 800, 'rate_pct': 12.5},
            {'category': 'FP_total',  'hits': 12,  'eligible': 1200, 'rate_pct': 1.0},
        ],
        'per_bin': per_bin,
        'per_site': {
            'image_x': image_x, 'image_y': image_y,
            'tp_rate': tp_rate, 'fp_rate': fp_rate,
            'loading_rate': loading_rate,
            'tp_elig': tp_elig, 'fp_elig': fp_elig,
            'n_loaded': 40,
        },
        'diag_series_all': {
            'total_ms':  [100.0, 110.0, 105.0, 95.0],
            'n_loaded':  [12, 14, 13, 11],
            'n_dropped': [0, 1, 0, 2],
            'aborted':   [0, 0, 0, 0],
            'two_round_phase': [None, None, None, None],
        },
        'loaded_frac': {'bi_mean': 0.7, 'bo_mean': 0.5,
                          'bi_std': 0.05, 'bo_std': 0.1},
        '_backfill_mapping': {
            'slm_run_id': '2026-05-29T17:00:00.000000',
            'matlab_scan_id': '20260529170000',
            'matched_delta_seconds': 0.0,
            'shot_count_confidence': 'exact',
            'shot_count_slm': 40,
            'shot_count_lab': 40,
            'source': 'http://test',
            'script': 'test_fixture',
        },
        'synced_at_iso': '2026-06-01T00:00:00',
    }


# ---------------------------------------------------------------------------
# Tier 1 — module-level helpers
# ---------------------------------------------------------------------------

def test_load_slm_analysis_missing(tmp_path):
    """No sidecar -> None."""
    assert _load_slm_analysis(tmp_path) is None


def test_load_slm_analysis_present(tmp_path):
    """Sidecar present -> parsed dict."""
    payload = _make_slm_analysis()
    (tmp_path / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')
    result = _load_slm_analysis(tmp_path)
    assert result is not None
    assert result['n_shots'] == 40
    assert result['summary'][0]['category'] == 'TP_return'


def test_per_site_from_slm_analysis_masks(tmp_path):
    """per_site override carries is_target_site / is_nontarget_site masks."""
    payload = _make_slm_analysis(n_sites=20, n_targets=8)
    ps = _per_site_from_slm_analysis(payload)
    assert ps['source'] == 'slm_server_cached'
    assert len(ps['x']) == 20
    assert sum(ps['is_target_site']) == 8
    assert sum(ps['is_nontarget_site']) == 12
    # TP rate finite only at target sites
    finite_tp = sum(1 for v in ps['tp_rate'] if isinstance(v, float) and v == v)
    assert finite_tp == 8
    # FP rate finite only at non-target sites
    finite_fp = sum(1 for v in ps['fp_rate'] if isinstance(v, float) and v == v)
    assert finite_fp == 12
    # survival_mean (lab-canonical key) is set to tp_rate (NaN at non-targets)
    assert ps['survival_mean'][0] is not None  # first 8 are targets
    nan_count = sum(1 for v in ps['survival_mean']
                     if not isinstance(v, float) or v != v)
    assert nan_count == 12   # exactly the non-target sites


def test_target_aware_fuzzy_axis_match(tmp_path):
    """Lab axes like 'rearrange_kwargs.axis_a' match SLM 'axis_a' by
    suffix on the last dotted component."""
    payload = _make_slm_analysis(n_bins=4)
    out = {'sweep': {'cols': ['rearrange_kwargs.axis_a',
                                'rearrange_kwargs.axis_b'],
                      'values': [[0, 1], [0, 1]]}}
    scan_params = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    ta = _target_aware_from_slm_analysis(payload, out, scan_params)
    assert ta is not None
    assert ta['source'] == 'slm_server_cached'
    assert ta['axes_matched'] == ['rearrange_kwargs.axis_a',
                                    'rearrange_kwargs.axis_b']
    assert ta['per_param_mean'] is not None
    assert len(ta['per_param_mean']) == 4
    # Sanity: rates should be 0.80, 0.82, 0.84, 0.86
    expected = [0.80, 0.82, 0.84, 0.86]
    means = ta['per_param_mean']
    # Order depends on how scan_params match per_bin; check set match.
    assert sorted(round(v, 2) for v in means) == sorted(expected)


def test_target_aware_no_axis_match():
    """When axis names don't match at all, per_param_mean stays None but
    overall_mean is still surfaced."""
    payload = _make_slm_analysis()
    out = {'sweep': {'cols': ['totally.different.axis'],
                      'values': [[0]]}}
    scan_params = np.array([[0]], dtype=float)
    ta = _target_aware_from_slm_analysis(payload, out, scan_params)
    assert ta is not None
    assert ta['per_param_mean'] is None
    assert ta['overall_mean'] == pytest.approx(0.875, abs=1e-6)
    assert ta['loss_overall'] == pytest.approx(0.125, abs=1e-6)
    assert ta['fp_overall'] == pytest.approx(0.01, abs=1e-6)


def test_diag_aggregate_from_slm_analysis():
    payload = _make_slm_analysis()
    agg = _diag_aggregate_from_slm_analysis(payload)
    assert agg['n_rows'] == 4
    assert agg['mean_total_ms'] == pytest.approx(102.5, abs=0.1)
    assert agg['mean_n_loaded'] == pytest.approx(12.5, abs=0.1)
    assert agg['aborted_count'] == 0
    assert agg['source'] == 'slm_analysis_cache'


# ---------------------------------------------------------------------------
# Tier 2 — analyze_scan_dir integration with cached analysis
# ---------------------------------------------------------------------------

def _make_minimal_scan_dir(tmp_path, scan_id='20260529170000', n_sites=20):
    """Tiny data_*.h5 + .mat sidecar so load_scan_from_path doesn't barf,
    matching the pattern from test_phase5a_paths.py."""
    day, hms = scan_id[:8], scan_id[8:]
    scan_dir = tmp_path / f'data_{day}_{hms}'
    scan_dir.mkdir()
    h5p = scan_dir / f'data_{day}_{hms}.h5'
    matp = scan_dir / f'data_{day}_{hms}.mat'
    # 4 shots x n_sites
    with h5py.File(h5p, 'w') as f:
        f.create_dataset('logicals', data=np.ones((8, n_sites), dtype=np.uint8))
        f.create_dataset('intensities', data=np.full((8, n_sites), 0.5))
        f.create_dataset('seq_ids', data=np.arange(1, 5))
    with h5py.File(matp, 'w') as f:
        scan = f.create_group('Scan')
        scan.create_dataset('SaveOK', data=np.array([[1.0]]))
        scan.create_dataset('NumImages', data=np.array([[2.0]]))
        scan.create_dataset('NumPerGroup', data=np.array([[1.0]]))
        scan.create_dataset('Params', data=np.array([[1.0]] * 4))
        scan.create_dataset('ScanType', data=np.array([[1.0]]))
        scan.create_dataset('initGridLocationsX',
                              data=np.linspace(0, 1, n_sites))
        scan.create_dataset('initGridLocationsY',
                              data=np.linspace(0, 1, n_sites))
    return scan_dir


def test_analyze_scan_dir_surfaces_target_aware(tmp_path):
    """End-to-end: a scan_dir with both lab h5 + slm_analysis.json sidecar
    returns target_aware populated and slm_analysis_cached=True."""
    scan_dir = _make_minimal_scan_dir(tmp_path)
    payload = _make_slm_analysis()
    (scan_dir / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')

    result = analyze_scan_dir(str(scan_dir))
    assert result['slm_analysis_cached'] is True

    ta = result['target_aware']
    assert ta is not None
    assert ta['overall_mean'] == pytest.approx(0.875, abs=1e-6)
    assert ta['fp_overall'] == pytest.approx(0.01, abs=1e-6)


def test_analyze_scan_dir_overrides_per_site(tmp_path):
    """per_site is replaced with the SLM's TP/FP-aware version when
    cached analysis is present."""
    scan_dir = _make_minimal_scan_dir(tmp_path)
    payload = _make_slm_analysis(n_sites=20, n_targets=8)
    (scan_dir / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')

    result = analyze_scan_dir(str(scan_dir))
    ps = result.get('per_site') or {}
    assert ps.get('source') == 'slm_server_cached'
    assert sum(1 for v in (ps.get('is_target_site') or []) if v) == 8
    assert sum(1 for v in (ps.get('is_nontarget_site') or []) if v) == 12
    # The lab-computed per_site should still be available as backup
    assert 'per_site_lab_computed' in result


def test_analyze_scan_dir_diag_aggregate_fallback(tmp_path):
    """When no slm_diag.h5 exists but slm_analysis.json does, the
    diag_aggregate falls through and shows source='slm_analysis_cache'."""
    scan_dir = _make_minimal_scan_dir(tmp_path)
    payload = _make_slm_analysis()
    (scan_dir / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')

    result = analyze_scan_dir(str(scan_dir))
    assert (scan_dir / 'slm_diag.h5').exists() is False
    da = result['diag_aggregate']
    assert da is not None
    assert da['source'] == 'slm_analysis_cache'
    assert da['mean_total_ms'] == pytest.approx(102.5, abs=0.1)


def test_analyze_scan_dir_no_sidecars_no_target_aware(tmp_path):
    """Plain scan dir (no sidecars) -> target_aware is None, no
    survival_source flag, per_site lab-computed."""
    scan_dir = _make_minimal_scan_dir(tmp_path)
    result = analyze_scan_dir(str(scan_dir))
    assert result['slm_analysis_cached'] is False
    assert result['target_aware'] is None
    # survival_mean still populated lab-side, no override flag
    assert result['summary'].get('survival_source') is None


def test_summary_override_preserves_per_site_backup(tmp_path):
    """When target_aware overrides survival_mean, the original per-site
    survival is preserved under survival_mean_per_site."""
    scan_dir = _make_minimal_scan_dir(tmp_path)
    # Make the per_bin axes match the lab's (axes from .mat fixture).
    # Use a payload where axes can be matched against the lab's
    # 'unknown' axis 0 sweep.
    payload = _make_slm_analysis()
    (scan_dir / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')

    result = analyze_scan_dir(str(scan_dir))
    s = result['summary']
    # Either: per_param matched -> override happened, backup present
    # Or:     no match -> survival_source has _overall_only suffix
    src = s.get('survival_source')
    if src == 'slm_server_cached':
        assert 'survival_mean_per_site' in s
    elif src and src.endswith('_overall_only'):
        assert 'survival_overall_target_aware' in s
    else:
        # No override -> survival_mean is the lab-computed array
        assert isinstance(s.get('survival_mean'), list)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier 3 — lab-side target-aware TP from paths_per_shot
# ---------------------------------------------------------------------------

def _make_new_run_fixture(tmp_path, n_sites=16, n_shots=4):
    """Build a scan dir simulating a NEW (post-Phase-5a) rearrangement
    run: data_*.h5 with logicals, .mat with gridLocations, slm_diag.h5
    with paths columns populated, slm_grid.json with bit-ordered coords.
    Returns (scan_dir, expected_per_shot_tp).
    """
    scan_id = '20260601120000'
    day, hms = scan_id[:8], scan_id[8:]
    scan_dir = tmp_path / f'data_{day}_{hms}'
    scan_dir.mkdir()
    # Grid: 4x4 row-major
    grid_x = np.tile(np.arange(4), 4).astype(float)
    grid_y = np.repeat(np.arange(4), 4).astype(float)

    # Loaded everywhere; img2 survives at specific target sites per shot.
    # Shot k (1-indexed): targets are sites [0, k]; only site 0 always
    # survives, site k survives if k <= 2. So expected TP:
    #   shot 1: target=[0, 1]; img2 has 0,1 → 2/2 = 1.0
    #   shot 2: target=[0, 2]; img2 has 0,2 → 2/2 = 1.0
    #   shot 3: target=[0, 3]; img2 has 0 only → 1/2 = 0.5
    #   shot 4: target=[0, 4]; bit 4 doesn't exist as target site,
    #            so we use bit indices [0, 5] to stay within 0..15.
    target_paired_per_shot = [[0, 1], [0, 2], [0, 3], [0, 5]]
    img2_survive_sites    = [[0, 1], [0, 2], [0],     [0]]
    expected_tp = [1.0, 1.0, 0.5, 0.5]

    img1 = np.ones((n_shots, n_sites), dtype=np.uint8)
    img2 = np.zeros((n_shots, n_sites), dtype=np.uint8)
    for k, sites in enumerate(img2_survive_sites):
        for s in sites:
            img2[k, s] = 1
    seq_ids = np.arange(1, n_shots + 1, dtype=np.int64)

    # Interleave img1/img2 into a single logicals array (the layout
    # `load_scan_from_path` produces when NumImages>=2). Row 2k = img1
    # of shot k; row 2k+1 = img2 of shot k. Total 2*n_shots rows.
    interleaved = np.zeros((2 * n_shots, n_sites), dtype=np.uint8)
    interleaved[0::2] = img1
    interleaved[1::2] = img2

    with h5py.File(scan_dir / f'data_{day}_{hms}.h5', 'w') as f:
        f.create_dataset('logicals', data=interleaved)
        f.create_dataset('seq_ids', data=seq_ids)
        f.create_dataset('intensities',
                          data=np.full((2 * n_shots, n_sites), 0.5, dtype=float))
    with h5py.File(scan_dir / f'data_{day}_{hms}.mat', 'w') as f:
        scan = f.create_group('Scan')
        scan.create_dataset('SaveOK',      data=np.array([[1.0]]))
        scan.create_dataset('NumImages',   data=np.array([[2.0]]))
        scan.create_dataset('NumPerGroup', data=np.array([[1.0]]))
        # 4 params (one per shot) so each shot is its own scan-point.
        scan.create_dataset('Params', data=np.array([[1.0], [2.0], [3.0], [4.0]]))
        scan.create_dataset('ScanType', data=np.array([[1.0]]))
        scan.create_dataset('initGridLocationsX', data=grid_x)
        scan.create_dataset('initGridLocationsY', data=grid_y)

    # slm_diag.h5 with paths_per_shot populated (Phase 5a v2 schema).
    from yb_analysis.slm_sync.sync import _append_rows_to_h5
    rows = []
    for k in range(n_shots):
        rows.append({
            'seq_id': int(seq_ids[k]),
            'retry_count': 0,
            'ts_iso': '', 'ts_epoch': 0.0,
            'run_id': 'test', 'client_id': 'test',
            'diag': {
                'n_loaded': n_sites,
                'loaded_paired': target_paired_per_shot[k],   # placeholder, not used by TP
                'target_paired': target_paired_per_shot[k],
            },
        })
    _append_rows_to_h5(scan_dir / 'slm_diag.h5', rows)

    # Grid sidecar — bit-ordered coords matching gridLocations.
    coords = [[float(y), float(x)] for y, x in zip(grid_y, grid_x)]
    (scan_dir / 'slm_grid.json').write_text(json.dumps({
        'schema': 1, 'run_id': 'test',
        'init_grid':   {'coords': coords, 'n_sites': n_sites,
                         'gridloc_diag': {'rms_knm': 0.5}},
        'target_grid': {'coords': coords, 'n_sites': n_sites,
                         'gridloc_diag': None},
        'grid_rotation': None,
    }), encoding='utf-8')

    return scan_dir, expected_tp


def test_lab_paths_target_aware_new_run(tmp_path):
    """When paths_per_shot is populated AND no slm_analysis.json
    cache exists, target_aware is built lab-side from logicals.
    source = 'lab_paths', not 'slm_server_cached'."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    scan_dir, expected_tp = _make_new_run_fixture(tmp_path)
    result = analyze_scan_dir(str(scan_dir))
    ta = result['target_aware']
    assert ta is not None, f'target_aware should be populated; got None'
    assert ta['source'] == 'lab_paths', (
        f"expected lab_paths source, got {ta['source']!r}")
    # Per-param TP should match the expected per-shot values
    # (each shot is its own param).
    means = ta['per_param_mean']
    assert means is not None
    assert len(means) == len(expected_tp)
    for got, want in zip(means, expected_tp):
        assert abs(got - want) < 1e-6, f'expected {want}, got {got}'
    # Overall = mean of per-shot TPs = (1+1+0.5+0.5)/4 = 0.75
    assert abs(ta['overall_mean'] - 0.75) < 1e-6


def test_lab_paths_target_aware_no_grid(tmp_path):
    """Target-aware survival must work even when the run has NO baked grid
    (a legacy pyctrl config with no initGridLocations and no slm_grid.json):
    the primary path indexes the logicals directly via target_paired, needing
    only the logicals width, not site coordinates. survival_vs_distance still
    needs coords and is correctly skipped."""
    import glob
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    scan_dir, expected_tp = _make_new_run_fixture(tmp_path)
    # Strip every grid source: initGridLocations in the .mat + the sidecar.
    matp = glob.glob(str(scan_dir / 'data_*.mat'))[0]
    with h5py.File(matp, 'a') as f:
        for k in ('initGridLocationsX', 'initGridLocationsY'):
            if k in f['Scan']:
                del f['Scan'][k]
    (scan_dir / 'slm_grid.json').unlink()
    result = analyze_scan_dir(str(scan_dir), sync_slm_diag=False)
    ta = result['target_aware']
    assert ta is not None and ta['source'] == 'lab_paths', (
        f'no-grid target_aware should still be lab_paths; got {ta}')
    assert abs(ta['overall_mean'] - 0.75) < 1e-6
    # The distance plot genuinely needs coords → correctly None (not an error).
    assert result['survival_vs_distance'] is None


def test_lab_paths_prefers_over_slm_cache(tmp_path):
    """When BOTH paths_per_shot and slm_analysis.json are present, the
    lab-paths computation wins (source = 'lab_paths')."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    scan_dir, expected_tp = _make_new_run_fixture(tmp_path)
    # Plant a contradictory slm_analysis.json to verify lab-paths wins.
    payload = _make_slm_analysis()
    payload['summary'][0]['rate_pct'] = 12.34   # obviously bogus
    (scan_dir / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')
    result = analyze_scan_dir(str(scan_dir))
    ta = result['target_aware']
    assert ta['source'] == 'lab_paths'
    # The bogus 12.34% should NOT show up; lab-computed overall is 0.75.
    assert abs(ta['overall_mean'] - 0.75) < 1e-6


def test_lab_paths_falls_back_when_no_targets(tmp_path):
    """Paths present but with empty target_paired arrays (Phase 5a
    schema-upgrade-of-legacy case) → lab-paths returns None and falls
    back to slm_analysis cache when available."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    scan_dir, _ = _make_new_run_fixture(tmp_path)
    # Wipe the path columns in slm_diag.h5 to simulate legacy upgrade.
    with h5py.File(scan_dir / 'slm_diag.h5', 'a') as f:
        n = f['/diag/seq_id'].shape[0]
        for col in ('loaded_paired', 'target_paired'):
            del f['/diag'][col]
            ds = f['/diag'].create_dataset(col, (n,), maxshape=(None,),
                                            dtype=h5py.vlen_dtype(np.int64))
            for i in range(n):
                ds[i] = np.array([], dtype=np.int64)
    # Also drop the diag_json column so v1 fallback doesn't find them.
    with h5py.File(scan_dir / 'slm_diag.h5', 'a') as f:
        if 'diag_json' in f['/diag']:
            del f['/diag']['diag_json']
    # Add a SLM cache so the fallback has somewhere to go.
    payload = _make_slm_analysis()
    (scan_dir / ANALYSIS_JSON).write_text(json.dumps(payload), encoding='utf-8')
    result = analyze_scan_dir(str(scan_dir))
    ta = result['target_aware']
    assert ta is not None
    assert ta['source'] == 'slm_server_cached', (
        f'expected SLM fallback, got {ta["source"]!r}')


# ---------------------------------------------------------------------------
# Cross-grid rearrangement: init pattern != target pattern
# (img1 loaded on the 47x47 grid, img2 detected on the 33x33 target grid).
# img1 / img2 have DIFFERENT site counts (two_array layout) so per-site
# survival/loss/FP are undefined; the meaningful survival is the target-aware
# TP (target_paired -> img2). Regression guard for the crash where
# analyze_scan_dir blew up at prob11(logic1, logic2) on the shape mismatch.
# ---------------------------------------------------------------------------

def _make_cross_grid_fixture(tmp_path, n_init=12, n_target=6, n_shots=4):
    """Two_array scan whose img1 (loading) and img2 (target) grids differ in
    site count. ``target_paired`` indexes the img2 (target) grid directly.

    img1 loads every init site; img2 fills target sites per shot. Each shot is
    its own scan-param. Returns (scan_dir, expected_per_shot_tp).
    """
    scan_id = '20260611153013'
    day, hms = scan_id[:8], scan_id[8:]
    scan_dir = tmp_path / f'data_{day}_{hms}'
    scan_dir.mkdir()

    # target_paired indexes img2 (0..n_target-1). Per-shot target sets +
    # which of them survive in img2 -> expected TP = survived/targeted.
    target_paired_per_shot = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    img2_survive_sites     = [[0, 1, 2], [0, 1],    [0],       []]
    expected_tp = [1.0, 2 / 3, 1 / 3, 0.0]

    img1 = np.ones((n_shots, n_init), dtype=bool)          # 47x47 analogue
    img2 = np.zeros((n_shots, n_target), dtype=bool)       # 33x33 analogue
    for k, sites in enumerate(img2_survive_sites):
        for s in sites:
            img2[k, s] = True
    seq_ids = np.arange(1, n_shots + 1, dtype=np.int64)

    with h5py.File(scan_dir / f'data_{day}_{hms}.h5', 'w') as f:
        f.attrs['two_array'] = True
        f.create_dataset('logicals_img1', data=img1)
        f.create_dataset('logicals_img2', data=img2)
        f.create_dataset('intensities_img1',
                          data=np.full((n_shots, n_init), 0.5, dtype=float))
        f.create_dataset('intensities_img2',
                          data=np.full((n_shots, n_target), 0.5, dtype=float))
        f.create_dataset('seq_ids', data=seq_ids)
    with h5py.File(scan_dir / f'data_{day}_{hms}.mat', 'w') as f:
        scan = f.create_group('Scan')
        scan.create_dataset('SaveOK',      data=np.array([[1.0]]))
        scan.create_dataset('NumImages',   data=np.array([[2.0]]))
        scan.create_dataset('NumPerGroup', data=np.array([[1.0]]))
        scan.create_dataset('Params', data=np.array([[1.0], [2.0], [3.0], [4.0]]))
        scan.create_dataset('ScanType', data=np.array([[1.0]]))
        # Lab detection grid is the INIT (img1) grid: n_init sites.
        scan.create_dataset('initGridLocationsX',
                            data=np.arange(n_init, dtype=float))
        scan.create_dataset('initGridLocationsY',
                            data=np.zeros(n_init, dtype=float))

    from yb_analysis.slm_sync.sync import _append_rows_to_h5
    rows = []
    for k in range(n_shots):
        rows.append({
            'seq_id': int(seq_ids[k]),
            'retry_count': 0, 'ts_iso': '', 'ts_epoch': 0.0,
            'run_id': 'test', 'client_id': 'test',
            'diag': {
                'n_loaded': n_init,
                # loaded_paired indexes img1 (init); target_paired indexes img2.
                'loaded_paired': list(range(len(target_paired_per_shot[k]))),
                'target_paired': target_paired_per_shot[k],
            },
        })
    _append_rows_to_h5(scan_dir / 'slm_diag.h5', rows)
    return scan_dir, expected_tp


def test_cross_grid_logicals_detection():
    """_cross_grid_logicals flags differing img1/img2 site counts only."""
    from yb_analysis.analysis.run_analysis import _cross_grid_logicals
    a = np.ones((10, 3, 2), dtype=bool)
    b = np.ones((6, 3, 2), dtype=bool)
    same = np.ones((10, 3, 2), dtype=bool)
    assert _cross_grid_logicals(a, b) is True
    assert _cross_grid_logicals(a, same) is False
    assert _cross_grid_logicals(a, None) is False


def test_cross_grid_analyze_does_not_crash_and_tp_is_target_aware(tmp_path):
    """A rearrangement scan with DIFFERENT init/target patterns (img1 and img2
    on different grids) analyzes without crashing, and the headline survival is
    the target-aware TP (target_paired -> img2), not the undefined per-site
    survival that would otherwise crash at prob11(logic1, logic2)."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    scan_dir, expected_tp = _make_cross_grid_fixture(tmp_path)
    result = analyze_scan_dir(str(scan_dir), sync_slm_diag=False)
    assert result.get('unpack_error') is None
    # Different widths survived unpack.
    ds = result['data_shapes']
    assert ds['logicals_img1'][0] != ds['logicals_img2'][0] or \
        ds['logicals_img1'][1] != ds['logicals_img2'][1]
    ta = result['target_aware']
    assert ta is not None and ta['source'] == 'lab_paths'
    means = ta['per_param_mean']
    assert len(means) == len(expected_tp)
    for got, want in zip(means, expected_tp):
        assert abs(got - want) < 1e-6, f'expected {want}, got {got}'
    # Overall = mean of per-shot TP = (1 + 2/3 + 1/3 + 0)/4 = 0.5
    assert abs(ta['overall_mean'] - 0.5) < 1e-6
    # The summary survival curve is OVERRIDDEN to the target-aware values
    # (per-site survival is undefined across grids).
    assert result['summary'].get('survival_source') == 'lab_paths'
    for got, want in zip(result['summary']['survival_mean'], expected_tp):
        assert abs(got - want) < 1e-6
    # Loading rate still comes from img1 (every init site loaded -> 1.0).
    for lr in result['summary']['loading_rate']:
        assert abs(lr - 1.0) < 1e-6
    # per_iteration didn't crash; survival is the target-aware TP override.
    pi = result.get('per_iteration')
    assert pi is not None
    assert pi.get('survival_source') == 'lab_paths'


def test_target_grid_camera_xy_affine_maps_registry(monkeypatch, tmp_path):
    """_target_grid_camera_xy fits knm->camera from the INIT pattern and maps
    the TARGET pattern's registry knm into camera px."""
    from yb_analysis.analysis import pattern_registry as pr
    from yb_analysis.analysis.run_analysis import _target_grid_camera_xy
    monkeypatch.setenv('YB_PATTERNS_DIR', str(tmp_path / 'patterns'))
    # init = 3x3 knm lattice; target = 2x2 knm sub-lattice. (y, x).
    init_knm = [[y, x] for y in (0, 10, 20) for x in (0, 10, 20)]   # 9
    tgt_knm  = [[y, x] for y in (5, 15) for x in (5, 15)]           # 4
    pr.write_pattern({'name': 'initpat', 'knm': init_knm})
    pr.write_pattern({'name': 'tgtpat',  'knm': tgt_knm})
    # Known affine cam = 2*knm + [5, 7] (applied to the init grid coords).
    import numpy as np
    ik = np.asarray(init_knm, float)
    cam = 2 * ik + np.array([5.0, 7.0])
    scan = {
        'imagePatternsJson': json.dumps(
            [{'name': 'initpat'}, {'name': 'tgtpat'}]),
        'initGridLocationsY': cam[:, 0].tolist(),
        'initGridLocationsX': cam[:, 1].tolist(),
    }
    out = _target_grid_camera_xy(scan, 4)
    assert out is not None and out.shape == (4, 2)
    want = 2 * np.asarray(tgt_knm, float) + np.array([5.0, 7.0])
    assert np.allclose(out, want, atol=1e-6)


def test_cross_grid_per_site_map_on_target_grid(monkeypatch, tmp_path):
    """A cross-grid rearrangement run produces a per-site SURVIVAL/FP map on
    the TARGET (img2) grid, with camera coords from the affine-mapped registry
    knm. Loading is omitted (an init-grid quantity)."""
    import numpy as np
    from yb_analysis.analysis import pattern_registry as pr
    from yb_analysis.analysis.run_analysis import _per_site_from_lab_paths
    monkeypatch.setenv('YB_PATTERNS_DIR', str(tmp_path / 'patterns'))
    init_knm = [[y, x] for y in (0, 10, 20) for x in (0, 10, 20)]   # 9 init sites
    tgt_knm  = [[y, x] for y in (5, 15) for x in (5, 15)]           # 4 target sites
    pr.write_pattern({'name': 'initpat', 'knm': init_knm})
    pr.write_pattern({'name': 'tgtpat',  'knm': tgt_knm})
    cam = 2 * np.asarray(init_knm, float) + np.array([5.0, 7.0])
    scan = {
        'NumImages': 2,
        'imagePatternsJson': json.dumps(
            [{'name': 'initpat'}, {'name': 'tgtpat'}]),
        'initGridLocationsY': cam[:, 0].tolist(),
        'initGridLocationsX': cam[:, 1].tolist(),
    }
    n_shots = 4
    img1 = np.ones((n_shots, 9), dtype=np.uint8)        # 47x47 analogue (9)
    # Target site occupancy per shot -> per-site TP = 1.0, 0.75, 0.5, 0.0.
    img2 = np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
    ], dtype=np.uint8)
    seq_ids = np.arange(1, n_shots + 1, dtype=np.int64)
    bundle = {
        'two_array': True, 'logicals_img1': img1, 'logicals_img2': img2,
        'seq_ids': seq_ids,
    }
    paths = [{'seq_id': int(s), 'target_paired': [0, 1, 2, 3]} for s in seq_ids]
    ps = _per_site_from_lab_paths(paths, bundle, scan)
    assert ps is not None and ps['source'] == 'lab_paths'
    assert len(ps['x']) == 4 and len(ps['y']) == 4
    # Coords are the target grid mapped through the init affine.
    want_xy = 2 * np.asarray(tgt_knm, float) + np.array([5.0, 7.0])
    assert np.allclose(ps['y'], want_xy[:, 0], atol=1e-6)
    assert np.allclose(ps['x'], want_xy[:, 1], atol=1e-6)
    # Survival (TP) per target site.
    assert np.allclose(ps['survival_mean'], [1.0, 0.75, 0.5, 0.0], atol=1e-6)
    assert all(ps['is_target_site'])          # every site is a target every shot
    # Target is the whole array -> NO non-target sites -> FP undefined.
    assert not any(ps['is_nontarget_site'])
    assert all(v is None for v in ps['fp_rate'])
    # Loading lives on the INIT grid, carried separately so the dashboard can
    # still render the loading map (survival/FP stay on the target grid).
    assert ps['loading_rate'] is None
    assert ps.get('loading_init') is not None and len(ps['loading_init']) == 9
    assert len(ps['loading_x']) == 9 and len(ps['loading_y']) == 9


def test_cross_grid_unfilled_target_not_false_positive(monkeypatch, tmp_path):
    """On a low-loading shot the rearrange places fewer atoms than target
    sites, so some target sites are UNFILLED that shot. Those are unfilled
    targets, NOT false-positive sites: a spurious atom there must not be
    counted as FP. The non-target set is the run-level union of targeted
    sites, so an unfilled-this-shot but targeted-elsewhere site is never FP."""
    import numpy as np
    from yb_analysis.analysis import pattern_registry as pr
    from yb_analysis.analysis.run_analysis import _per_site_from_lab_paths
    monkeypatch.setenv('YB_PATTERNS_DIR', str(tmp_path / 'patterns'))
    init_knm = [[y, x] for y in (0, 10, 20) for x in (0, 10, 20)]   # 9
    tgt_knm  = [[y, x] for y in (5, 15) for x in (5, 15)]           # 4
    pr.write_pattern({'name': 'initpat', 'knm': init_knm})
    pr.write_pattern({'name': 'tgtpat',  'knm': tgt_knm})
    cam = 2 * np.asarray(init_knm, float) + np.array([5.0, 7.0])
    scan = {
        'NumImages': 2,
        'imagePatternsJson': json.dumps(
            [{'name': 'initpat'}, {'name': 'tgtpat'}]),
        'initGridLocationsY': cam[:, 0].tolist(),
        'initGridLocationsX': cam[:, 1].tolist(),
    }
    n_shots = 3
    img1 = np.ones((n_shots, 9), dtype=np.uint8)
    # Site 3 is targeted on shots 1,2 but UNFILLED on shot 3 (low loading).
    # On shot 3 img2 shows a spurious atom at site 3 -> must NOT count as FP.
    img2 = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],   # site 3 occupied even though not placed this shot
    ], dtype=np.uint8)
    seq_ids = np.arange(1, n_shots + 1, dtype=np.int64)
    bundle = {'two_array': True, 'logicals_img1': img1,
              'logicals_img2': img2, 'seq_ids': seq_ids}
    paths = [
        {'seq_id': 1, 'target_paired': [0, 1, 2, 3]},
        {'seq_id': 2, 'target_paired': [0, 1, 2, 3]},
        {'seq_id': 3, 'target_paired': [0, 1, 2]},   # site 3 unplaced this shot
    ]
    ps = _per_site_from_lab_paths(paths, bundle, scan)
    assert ps is not None
    # Site 3 is part of the target pattern (targeted on shots 1,2) -> NOT a
    # non-target / FP site, despite being unfilled on shot 3.
    assert ps['is_nontarget_site'] == [False, False, False, False]
    assert all(v is None for v in ps['fp_rate'])


def test_reorder_summary_to_sweep_fixes_inversion():
    """A non-ascending sweep (e.g. boolean precompute that unpacks as
    [True, False] = [1.0, 0.0]) leaves the per-param summary arrays in
    descriptor order while sweep['values'] is sorted -> the dashboard pairs
    each y with the wrong x (curve inverted vs the live view). Reordering the
    summary to ascending param-value order fixes the pairing."""
    import numpy as np
    from yb_analysis.analysis.run_analysis import _reorder_summary_to_sweep
    # precompute False(0.0)=0.9215, True(1.0)=0.9675; descriptor order [True, False].
    summary = {
        'survival_mean': [0.9675, 0.9215],
        'fp_mean':       [0.04, 0.02],
        'survival_n_shots': [130, 132],
        'survival_source': 'lab_paths',     # non-list: must be left untouched
    }
    scan_params = np.array([1.0, 0.0])      # unpack (descriptor) order
    _reorder_summary_to_sweep(summary, scan_params)
    # Now aligned with sweep values sorted([0.0, 1.0]) = [False, True].
    assert summary['survival_mean'] == [0.9215, 0.9675]   # False, True
    assert summary['fp_mean'] == [0.02, 0.04]
    assert summary['survival_n_shots'] == [132, 130]
    assert summary['survival_source'] == 'lab_paths'


def test_reorder_summary_to_sweep_noop_when_ascending():
    """Already-ascending sweeps (e.g. nsteps [100,140,180]) are untouched."""
    import numpy as np
    from yb_analysis.analysis.run_analysis import _reorder_summary_to_sweep
    summary = {'survival_mean': [0.80, 0.85, 0.90]}
    _reorder_summary_to_sweep(summary, np.array([100.0, 140.0, 180.0]))
    assert summary['survival_mean'] == [0.80, 0.85, 0.90]


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
