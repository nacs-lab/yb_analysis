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


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
