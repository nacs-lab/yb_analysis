"""Phase 5.5 Track E lab-side tests — `received_bits` surfaced into the
synced `slm_diag.h5` (schema v3) and read back by the run_analysis
diagnostic helper.

Covers:
  * `slm_diag.h5` gets the new `/diag/received_bits` vlen-string column,
    populated from each ledger row's `diag['received_bits']`.
  * Legacy rows that lack the field yield empty strings, no crash.
  * v2 -> v3 in-place upgrade backfills `''` for pre-existing rows.
  * `read_slm_received_bits` + `compare_lab_vs_slm_bitstrings` helpers.

Run as:
    pytest yb_analysis/tests/slm_migration/test_phase5p5_received_bits.py -v
"""

import h5py
import numpy as np
import pytest

from yb_analysis.slm_sync.sync import (
    DIAG_H5, SCHEMA_VERSION, _append_rows_to_h5, _ensure_h5,
    _STRING_DIAG_COLUMNS,
)
from yb_analysis.analysis.run_analysis import (
    read_slm_received_bits, compare_lab_vs_slm_bitstrings,
)


def _row(seq_id, bits=None, **diag_extra):
    diag = dict(diag_extra)
    if bits is not None:
        diag['received_bits'] = bits
    return {
        'seq_id': seq_id,
        'retry_count': 0,
        'ts_epoch': 1.0 + seq_id,
        'ts_iso': f'2026-06-02T00:00:0{seq_id}',
        'run_id': 'r1',
        'client_id': 'c1',
        'diag': diag,
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_version_is_v3():
    assert SCHEMA_VERSION == 3


def test_received_bits_is_a_string_column():
    assert 'received_bits' in _STRING_DIAG_COLUMNS


# ---------------------------------------------------------------------------
# sync surfacing
# ---------------------------------------------------------------------------


def test_sync_surfaces_received_bits(tmp_path):
    """slm_diag.h5 gets /diag/received_bits populated from the ledger
    rows' diag['received_bits']."""
    path = tmp_path / DIAG_H5
    rows = [
        _row(1, bits='1010', n_loaded=2),
        _row(2, bits='1111', n_loaded=4),
        _row(3, bits='0000', n_loaded=0),
    ]
    _append_rows_to_h5(path, rows)

    with h5py.File(path, 'r') as f:
        assert f['/meta'].attrs['schema_version'] == 3
        col = f['/diag/received_bits'][:]
        got = [v.decode() if isinstance(v, bytes) else str(v) for v in col]
    assert got == ['1010', '1111', '0000']


def test_legacy_rows_without_field_yield_empty_string(tmp_path):
    """Rows that predate Track E (no received_bits in diag) surface as
    empty strings — no crash."""
    path = tmp_path / DIAG_H5
    rows = [
        _row(1, bits='1100'),       # has the field
        _row(2),                    # legacy: no received_bits
        _row(3, bits='0011'),
    ]
    _append_rows_to_h5(path, rows)

    got = read_slm_received_bits(path)
    assert got == ['1100', '', '0011']


def test_v2_to_v3_upgrade_backfills_empty(tmp_path):
    """An existing v2 file (no received_bits column) is upgraded in
    place: old rows get '' backfilled, new rows carry their bits."""
    path = tmp_path / DIAG_H5

    # Build a v2-style file by creating the schema then deleting the
    # received_bits column and stamping schema_version=2.
    f = _ensure_h5(path)
    f['meta'].attrs['schema_version'] = 2
    if 'received_bits' in f['diag']:
        del f['diag']['received_bits']
    f.close()

    # Manually append two "old" rows to the core columns so existing_n>0.
    with h5py.File(path, 'a') as f:
        d = f['diag']
        for name, vals in (('seq_id', [1, 2]), ('retry_count', [0, 0])):
            ds = d[name]
            ds.resize((2,))
            ds[:] = vals

    # Now a fresh append triggers the v2->v3 upgrade path in _ensure_h5,
    # which must backfill received_bits='' for the 2 pre-existing rows.
    _append_rows_to_h5(path, [_row(3, bits='1001')])

    with h5py.File(path, 'r') as f:
        assert f['/meta'].attrs['schema_version'] == 3
        col = [v.decode() if isinstance(v, bytes) else str(v)
               for v in f['/diag/received_bits'][:]]
    # 2 backfilled empties + 1 real (the new append).
    assert col == ['', '', '1001']


def test_read_helper_handles_missing_file_and_column(tmp_path):
    """read_slm_received_bits is best-effort: missing file -> [],
    file without the column -> []."""
    missing = tmp_path / 'nope.h5'
    assert read_slm_received_bits(missing) == []

    path = tmp_path / DIAG_H5
    f = _ensure_h5(path)
    del f['diag']['received_bits']
    f.close()
    assert read_slm_received_bits(path) == []


# ---------------------------------------------------------------------------
# compare_lab_vs_slm_bitstrings helper
# ---------------------------------------------------------------------------


def test_compare_basic_hamming_and_mask():
    lab = ['1010', '1111']
    slm = ['1000', '1110']   # differ in 1 site each
    out = compare_lab_vs_slm_bitstrings(lab, slm)
    assert out['n_shots'] == 2
    assert out['n_comparable'] == 2
    assert out['hamming'] == [1, 1]
    assert out['disagreement'] == [[0, 0, 1, 0], [0, 0, 0, 1]]
    assert out['total_disagreements'] == 2
    assert out['mean_hamming'] == 1.0
    assert out['n_sites'] == 4
    # Per-site rate: site 2 disagreed in 1/2 shots, site 3 in 1/2.
    assert out['per_site_disagree_rate'] == [0.0, 0.0, 0.5, 0.5]
    assert out['skipped'] == []


def test_compare_accepts_array_lab_bits():
    """Lab bits as logicals/arrays compare against SLM '0'/'1' strings."""
    lab = [np.array([1, 0, 1, 0]), [True, True, True, True]]
    slm = ['1010', '1011']
    out = compare_lab_vs_slm_bitstrings(lab, slm)
    assert out['hamming'] == [0, 1]
    assert out['n_comparable'] == 2


def test_compare_skips_length_mismatch_and_empty():
    lab = ['1010', '', '111']
    slm = ['1010', '1111', '110011']
    out = compare_lab_vs_slm_bitstrings(lab, slm)
    # shot 0 comparable; shot 1 lab empty -> skipped; shot 2 length
    # mismatch -> skipped.
    assert out['hamming'][0] == 0
    assert out['hamming'][1] is None
    assert out['hamming'][2] is None
    assert out['n_comparable'] == 1
    reasons = {s['shot']: s['reason'] for s in out['skipped']}
    assert reasons[1] == 'unparseable_or_empty'
    assert 'length_mismatch' in reasons[2]
    # Mixed comparable lengths -> per_site_rate None, n_sites None.
    # (Only one comparable shot here, so length set is singleton -> rate
    #  IS computed; assert it reflects the single comparable shot.)
    assert out['per_site_disagree_rate'] == [0.0, 0.0, 0.0, 0.0]


def test_compare_truncates_to_shorter_list():
    out = compare_lab_vs_slm_bitstrings(['10', '10', '10'], ['10'])
    assert out['n_shots'] == 1


def test_compare_end_to_end_with_h5(tmp_path):
    """Read SLM received_bits from a synced h5 and diff against lab bits."""
    path = tmp_path / DIAG_H5
    _append_rows_to_h5(path, [
        _row(1, bits='1010'),
        _row(2, bits='0101'),
    ])
    slm_bits = read_slm_received_bits(path)
    lab_bits = ['1010', '0001']   # shot 2 differs in one site
    out = compare_lab_vs_slm_bitstrings(lab_bits, slm_bits)
    assert out['hamming'] == [0, 1]
    assert out['total_disagreements'] == 1


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
