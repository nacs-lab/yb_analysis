"""Phase 5a tests — per-shot rearrangement PATHS plumbing.

Phase 5a surfaces the SLM server's per-shot `loaded_paired` /
`target_paired` int lists (stamped by SLMnet's `pairing_extra`) from
`slm_diag.h5/diag_json` (Phase 2 catch-all column) to first-class vlen
columns plus a lab-side analysis join with `slm_grid.json`.

Bit-order invariant (load-bearing — see plan §Phase 5a):
  * init_grid[k] is the (y, x) coord of the site MATLAB bit k references.
  * target_grid[k] same for the target lattice.
  * loaded_paired[i] / target_paired[i] index into init_grid/target_grid
    of that row (parallel arrays — pair i is loaded_paired[i] →
    target_paired[i]).
  * If gridloc_diag is non-null in the sidecar, both grids are in
    camera-frame post-affine ordering; else knm-native.

Tiers:
1. Schema v2 round-trip — sync_scan writes loaded_paired/target_paired
   into the vlen columns when the diag rows carry them.
2. Back-compat — a v1 file produced by an older syncer (no path cols)
   is still readable; analysis falls through to diag_json.
3. Bit-order invariant — indices land where they should, no permutation.
4. Two-round disambiguation — two_round_phase / two_round_idx carried
   through both columns and analysis output.
5. Defensive — empty pairing rows (legacy / fast-path-abort) produce
   empty arrays, no crash.

Run as:
    python -m yb_analysis.tests.slm_migration.test_phase5a_paths
or:
    pytest yb_analysis/tests/slm_migration/test_phase5a_paths.py -v
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from yb_analysis.slm_sync import SlmSyncClient, sync_scan
from yb_analysis.slm_sync.sync import (
    SCHEMA_VERSION, DIAG_H5, _ensure_h5, _append_rows_to_h5,
)
from yb_analysis.analysis.run_analysis import analyze_scan_dir, _paths_per_shot
from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_grid_sidecar(scan_dir, *, init_coords, target_coords,
                        gridloc_applied=True):
    """Write a minimal slm_grid.json that satisfies _paths_per_shot.

    `gridloc_applied=True` puts a non-null `gridloc_diag` in init_grid,
    which the analysis reads as the 'camera_bitorder' frame flag.
    """
    payload = {
        'schema': 1,
        'run_id': 'fake-run',
        'init_grid': {
            'coords': [list(p) for p in init_coords],
            'n_sites': len(init_coords),
            'gridloc_diag': ({'rms_knm': 0.5, 'affine_rot': 0.0,
                              'n_bits': len(init_coords)}
                              if gridloc_applied else None),
        },
        'target_grid': {
            'coords': [list(p) for p in target_coords],
            'n_sites': len(target_coords),
            'gridloc_diag': None,
        },
        'grid_rotation': None,
    }
    (Path(scan_dir) / 'slm_grid.json').write_text(
        json.dumps(payload), encoding='utf-8')


def _add_pairing_row(fake, scan_id, seq_id, loaded_paired, target_paired,
                      **extra_diag):
    diag = {
        'total_ms':       42.0 + seq_id,
        'n_loaded':       len(loaded_paired),
        'protocol':       extra_diag.pop('protocol', 'block_tail'),
        'loaded_paired':  list(loaded_paired),
        'target_paired':  list(target_paired),
    }
    diag.update(extra_diag)
    fake.add_diag_row(scan_id, seq_id, diag)


# ---------------------------------------------------------------------------
# Tier 1 — schema v2 round-trip
# ---------------------------------------------------------------------------

def test_sync_scan_writes_v2_path_columns(tmp_path):
    """sync_scan writes loaded_paired/target_paired as vlen-int64 cols."""
    scan_dir = tmp_path / 'data_20260601_120000'
    scan_dir.mkdir()
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, 'A', 1, [3, 7], [12, 15])
        _add_pairing_row(fake, 'A', 2, [0, 1, 5], [2, 4, 6])
        client = SlmSyncClient(slm_url=fake.url)
        status = sync_scan('A', scan_dir, client=client,
                           sync_code=False, sync_grid=False)
    assert status['synced']
    assert status['rows_written'] == 2
    with h5py.File(scan_dir / DIAG_H5, 'r') as f:
        assert f['meta'].attrs['schema_version'] == SCHEMA_VERSION
        assert 'loaded_paired' in f['/diag']
        assert 'target_paired' in f['/diag']
        lp = f['/diag/loaded_paired'][:]
        tp = f['/diag/target_paired'][:]
        assert len(lp) == 2 and len(tp) == 2
        assert np.array_equal(np.asarray(lp[0], dtype=np.int64), [3, 7])
        assert np.array_equal(np.asarray(lp[1], dtype=np.int64), [0, 1, 5])
        assert np.array_equal(np.asarray(tp[0], dtype=np.int64), [12, 15])
        assert np.array_equal(np.asarray(tp[1], dtype=np.int64), [2, 4, 6])


def test_sync_scan_two_round_columns(tmp_path):
    """two_round_phase / two_round_idx land as their own columns."""
    scan_dir = tmp_path / 'data_20260601_120100'
    scan_dir.mkdir()
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, 'B', 1, [0], [0],
                          two_round_phase='initial', two_round_idx=0)
        _add_pairing_row(fake, 'B', 2, [1], [1],
                          two_round_phase='final', two_round_idx=1)
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('B', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    with h5py.File(scan_dir / DIAG_H5, 'r') as f:
        phases = [v.decode() if isinstance(v, bytes) else str(v)
                  for v in f['/diag/two_round_phase'][:]]
        idxs = list(f['/diag/two_round_idx'][:])
        assert phases == ['initial', 'final']
        assert idxs == [0.0, 1.0]


def test_sync_scan_empty_pairing(tmp_path):
    """Rows lacking pairing produce empty vlen arrays — no crash."""
    scan_dir = tmp_path / 'data_20260601_120200'
    scan_dir.mkdir()
    with FakeSlmServer() as fake:
        # No loaded_paired / target_paired keys (legacy or fast-path abort).
        fake.add_diag_row('C', 1, {'total_ms': 10.0, 'n_loaded': 0})
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('C', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    with h5py.File(scan_dir / DIAG_H5, 'r') as f:
        lp = f['/diag/loaded_paired'][:]
        tp = f['/diag/target_paired'][:]
        assert len(lp) == 1 and len(tp) == 1
        assert np.asarray(lp[0], dtype=np.int64).size == 0
        assert np.asarray(tp[0], dtype=np.int64).size == 0
        # two_round_idx default -1 sentinel:
        assert float(f['/diag/two_round_idx'][0]) == -1.0


# ---------------------------------------------------------------------------
# Tier 2 — schema upgrade (v1 → v2 in place)
# ---------------------------------------------------------------------------

def _build_v1_slm_diag_h5(path, rows):
    """Build a Phase-2-era v1 slm_diag.h5 (no path columns) for back-compat
    tests."""
    f = h5py.File(path, 'w')
    f.create_group('meta')
    f['meta'].attrs['schema_version'] = 1
    diag = f.create_group('diag')
    vlen_str = h5py.string_dtype(encoding='utf-8')
    diag.create_dataset('seq_id', (0,), maxshape=(None,), dtype='i8')
    diag.create_dataset('retry_count', (0,), maxshape=(None,), dtype='i8')
    diag.create_dataset('ts_epoch', (0,), maxshape=(None,), dtype='f8')
    diag.create_dataset('ts_iso', (0,), maxshape=(None,), dtype=vlen_str)
    diag.create_dataset('run_id', (0,), maxshape=(None,), dtype=vlen_str)
    diag.create_dataset('client_id', (0,), maxshape=(None,), dtype=vlen_str)
    # Only the v1 numeric/string columns (no two_round_idx).
    for k in ('total_ms', 'n_loaded'):
        diag.create_dataset(k, (0,), maxshape=(None,), dtype='f8')
    diag.create_dataset('diag_json', (0,), maxshape=(None,), dtype=vlen_str)

    n = len(rows)

    def _append(name, vals):
        ds = diag[name]
        ds.resize((n,))
        ds[:] = vals

    _append('seq_id', [r['seq_id'] for r in rows])
    _append('retry_count', [0] * n)
    _append('ts_epoch', [0.0] * n)
    _append('ts_iso', [''] * n)
    _append('run_id', [''] * n)
    _append('client_id', [''] * n)
    _append('total_ms', [(r.get('diag') or {}).get('total_ms', 0.0)
                          for r in rows])
    _append('n_loaded', [(r.get('diag') or {}).get('n_loaded', 0.0)
                          for r in rows])
    _append('diag_json', [json.dumps(r, default=str) for r in rows])
    f.close()


def test_v1_h5_upgrades_in_place_on_next_sync(tmp_path):
    """A v1 file gets the new columns added when a v2 syncer touches it."""
    scan_dir = tmp_path / 'data_20260601_120300'
    scan_dir.mkdir()
    path = scan_dir / DIAG_H5
    # Pre-existing v1 file with 2 rows.
    _build_v1_slm_diag_h5(path, [
        {'seq_id': 1, 'diag': {'total_ms': 11.0, 'n_loaded': 5,
                                'loaded_paired': [3, 7],
                                'target_paired': [12, 15]}},
        {'seq_id': 2, 'diag': {'total_ms': 12.0, 'n_loaded': 6}},
    ])

    # Trigger an upgrade by syncing one new row.
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, 'D', 3, [0], [0])
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('D', scan_dir, client=client,
                   sync_code=False, sync_grid=False)

    with h5py.File(path, 'r') as f:
        assert f['meta'].attrs['schema_version'] == SCHEMA_VERSION
        assert 'loaded_paired' in f['/diag']
        assert 'target_paired' in f['/diag']
        assert 'two_round_idx' in f['/diag']
        # The first 2 rows (pre-upgrade) should have EMPTY vlen arrays
        # (we don't backfill v1 row contents into the new columns;
        # those rows' path data still lives in diag_json).
        lp = f['/diag/loaded_paired'][:]
        assert len(lp) == 3
        assert np.asarray(lp[0], dtype=np.int64).size == 0
        assert np.asarray(lp[1], dtype=np.int64).size == 0
        assert list(np.asarray(lp[2], dtype=np.int64)) == [0]


def test_paths_per_shot_v1_fallback_via_diag_json(tmp_path):
    """When path cols missing (v1 file untouched), _paths_per_shot still
    extracts paths by parsing diag_json."""
    scan_dir = tmp_path / 'data_20260601_120400'
    scan_dir.mkdir()
    path = scan_dir / DIAG_H5
    _build_v1_slm_diag_h5(path, [
        {'seq_id': 1, 'diag': {'total_ms': 11.0, 'n_loaded': 2,
                                'loaded_paired': [3, 7],
                                'target_paired': [12, 15]}},
    ])
    # Minimal grid sidecar so the join produces coords.
    _make_grid_sidecar(
        scan_dir,
        init_coords=[(y, x) for y in range(4) for x in range(4)],   # 16 sites
        target_coords=[(y * 2.0, x * 2.0) for y in range(4) for x in range(4)],
        gridloc_applied=True)
    result = _paths_per_shot(path, scan_dir / 'slm_grid.json')
    assert result['paths_per_shot'] is not None
    assert result['paths_frame'] == 'camera_bitorder'
    assert result['paths_n_shots_with_pairing'] == 1
    row = result['paths_per_shot'][0]
    assert row['loaded_paired'] == [3, 7]
    assert row['target_paired'] == [12, 15]
    # init_grid[3] = (0, 3) given the row-major 4x4 fixture.
    assert row['init_xy'][0] == [0.0, 3.0]
    # init_grid[7] = (1, 3).
    assert row['init_xy'][1] == [1.0, 3.0]
    # target_grid[12] = (6.0, 0.0); target_grid[15] = (6.0, 6.0).
    assert row['target_xy'][0] == [6.0, 0.0]
    assert row['target_xy'][1] == [6.0, 6.0]


# ---------------------------------------------------------------------------
# Tier 3 — bit-order invariant
# ---------------------------------------------------------------------------

def test_bit_order_invariant_camera_frame(tmp_path):
    """Direct index correspondence: paths_per_shot[0]['init_xy'][0] ==
    init_grid[loaded_paired[0]]. No reorder, no swap, no remap."""
    scan_dir = tmp_path / 'data_20260601_120500'
    scan_dir.mkdir()
    # 5x5 init grid, 3x3 target grid.
    init = [(float(y), float(x)) for y in range(5) for x in range(5)]
    target = [(float(y * 10), float(x * 10))
              for y in range(3) for x in range(3)]
    _make_grid_sidecar(scan_dir, init_coords=init, target_coords=target,
                        gridloc_applied=True)
    with FakeSlmServer() as fake:
        # One shot, two pairs: bit 3 → target 0, bit 7 → target 4.
        _add_pairing_row(fake, 'BIT', 1, [3, 7], [0, 4])
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('BIT', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    result = _paths_per_shot(scan_dir / DIAG_H5, scan_dir / 'slm_grid.json')
    assert result['paths_frame'] == 'camera_bitorder'
    row = result['paths_per_shot'][0]
    # init_grid[3] = (0, 3) (row-major 5x5).
    assert row['init_xy'][0] == [0.0, 3.0]
    # init_grid[7] = (1, 2).
    assert row['init_xy'][1] == [1.0, 2.0]
    # target_grid[0] = (0, 0); target_grid[4] = (10, 10).
    assert row['target_xy'][0] == [0.0, 0.0]
    assert row['target_xy'][1] == [10.0, 10.0]


def test_bit_order_invariant_knm_native(tmp_path):
    """gridloc_applied=False → paths_frame says 'knm_native', same
    direct-index correspondence."""
    scan_dir = tmp_path / 'data_20260601_120600'
    scan_dir.mkdir()
    init = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    target = [(10.0, 10.0), (20.0, 20.0)]
    _make_grid_sidecar(scan_dir, init_coords=init, target_coords=target,
                        gridloc_applied=False)
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, 'KNM', 1, [2], [1])
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('KNM', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    result = _paths_per_shot(scan_dir / DIAG_H5, scan_dir / 'slm_grid.json')
    assert result['paths_frame'] == 'knm_native'
    row = result['paths_per_shot'][0]
    assert row['init_xy'] == [[3.0, 3.0]]
    assert row['target_xy'] == [[20.0, 20.0]]


def test_out_of_range_index_emits_none_placeholder(tmp_path):
    """If the diag carries an index that's outside the grid sidecar's
    coords array (corruption / version mismatch), the analyzer puts a
    [None, None] placeholder in init_xy/target_xy and keeps going."""
    scan_dir = tmp_path / 'data_20260601_120700'
    scan_dir.mkdir()
    _make_grid_sidecar(scan_dir,
                        init_coords=[(0, 0), (1, 1)],   # only 2 sites
                        target_coords=[(10, 10)],
                        gridloc_applied=True)
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, 'X', 1, [5], [99])
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('X', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    result = _paths_per_shot(scan_dir / DIAG_H5, scan_dir / 'slm_grid.json')
    row = result['paths_per_shot'][0]
    assert row['init_xy'][0] == [None, None]
    assert row['target_xy'][0] == [None, None]


# ---------------------------------------------------------------------------
# Tier 4 — two-round phase carried through
# ---------------------------------------------------------------------------

def test_two_round_phase_carries_through(tmp_path):
    """Two-round shots end up with distinct rows in paths_per_shot, each
    labelled with its phase."""
    scan_dir = tmp_path / 'data_20260601_120800'
    scan_dir.mkdir()
    _make_grid_sidecar(scan_dir,
                        init_coords=[(y, x) for y in range(3) for x in range(3)],
                        target_coords=[(y * 5, x * 5)
                                        for y in range(3) for x in range(3)],
                        gridloc_applied=True)
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, 'TR', 1, [0, 2], [0, 4],
                          two_round_phase='initial', two_round_idx=0,
                          protocol='two_round')
        _add_pairing_row(fake, 'TR', 2, [0], [0],
                          two_round_phase='final', two_round_idx=1,
                          protocol='two_round')
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('TR', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    result = _paths_per_shot(scan_dir / DIAG_H5, scan_dir / 'slm_grid.json')
    rows = result['paths_per_shot']
    assert len(rows) == 2
    assert [r['two_round_phase'] for r in rows] == ['initial', 'final']
    assert [r['two_round_idx'] for r in rows] == [0, 1]


# ---------------------------------------------------------------------------
# Tier 5 — analyze_scan_dir surfaces paths in its output
# ---------------------------------------------------------------------------

def test_analyze_scan_dir_surfaces_paths(tmp_path, monkeypatch):
    """analyze_scan_dir() output carries paths_per_shot, paths_frame,
    paths_n_shots_with_pairing keys when slm_diag.h5 is present."""
    scan_dir = tmp_path / 'data_20260601_120900'
    scan_dir.mkdir()
    # Minimal data_*.h5 so the loader doesn't barf.
    _make_minimal_lab_h5(scan_dir, '20260601120900')
    _make_grid_sidecar(scan_dir,
                        init_coords=[(0, 0), (1, 1)],
                        target_coords=[(10, 10)],
                        gridloc_applied=True)
    with FakeSlmServer() as fake:
        _add_pairing_row(fake, '20260601120900', 1, [0], [0])
        client = SlmSyncClient(slm_url=fake.url)
        sync_scan('20260601120900', scan_dir, client=client,
                   sync_code=False, sync_grid=False)
    result = analyze_scan_dir(str(scan_dir))
    assert result['paths_frame'] == 'camera_bitorder'
    assert result['paths_n_shots_with_pairing'] == 1
    assert result['paths_per_shot'] is not None
    assert result['paths_per_shot'][0]['init_xy'] == [[0.0, 0.0]]


def test_analyze_scan_dir_no_diag_returns_none(tmp_path):
    """No slm_diag.h5 → paths_per_shot is None, frame is None, count 0."""
    scan_dir = tmp_path / 'data_20260601_121000'
    scan_dir.mkdir()
    _make_minimal_lab_h5(scan_dir, '20260601121000')
    result = analyze_scan_dir(str(scan_dir))
    assert result['paths_per_shot'] is None
    assert result['paths_frame'] is None
    assert result['paths_n_shots_with_pairing'] == 0


# ---------------------------------------------------------------------------
# Test-only HDF5/mat fixture helper (mirrors test_phase4_dashboard.py shape)
# ---------------------------------------------------------------------------

def _make_minimal_lab_h5(scan_dir, scan_id):
    """Write a tiny data_*.h5 + sibling .mat so load_scan_from_path works.

    Mirrors the synthetic fixture shape Phase 4's tests use: v7.3 HDF5
    Scan struct with the bare minimum fields for analyze_scan_dir to
    return without unpack errors. Two img bits, two reps.
    """
    h5_path = scan_dir / f'data_{scan_id[:8]}_{scan_id[8:]}.h5'
    mat_path = scan_dir / f'data_{scan_id[:8]}_{scan_id[8:]}.mat'
    with h5py.File(h5_path, 'w') as f:
        # Two shots, 2-site logicals (img1/img2 = same length).
        f.create_dataset('logicals', data=np.array(
            [[1, 1], [1, 1], [1, 0], [1, 1]], dtype=np.uint8))
        f.create_dataset('intensities', data=np.array(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.0], [0.5, 0.5]], dtype=float))
        f.create_dataset('seq_ids', data=np.array([1, 2], dtype=np.int64))
    # Scan struct as v7.3 HDF5 .mat (h5py writes; load_scan_from_path
    # autodetects HDF5).
    with h5py.File(mat_path, 'w') as f:
        scan = f.create_group('Scan')
        # SaveOK gate
        scan.create_dataset('SaveOK', data=np.array([[1.0]]))
        scan.create_dataset('NumImages', data=np.array([[2.0]]))
        scan.create_dataset('NumPerGroup', data=np.array([[1.0]]))
        scan.create_dataset('Params', data=np.array([[1.0], [1.0]]))
        scan.create_dataset('ScanType', data=np.array([[1.0]]))
        # GridLocations for the per-site map.
        scan.create_dataset('initGridLocationsX',
                              data=np.array([0.0, 1.0]))
        scan.create_dataset('initGridLocationsY',
                              data=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Entrypoint for `python -m`
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
