"""Incremental enrichment cache for the runs list (Analysis/Sequence pickers).

The cache must (a) skip re-enriching a scan whose metadata is unchanged, (b)
enrich a NEW scan on the next pass, (c) re-enrich a scan whose .h5 mtime changed
(the live-growing current scan), and (d) re-enrich everything on force=True.

Run: yb_analysis-env python -m pytest yb_analysis/tests/test_runs_list_cache.py -v
"""
import os

import pytest

from yb_analysis.analysis import runs_list as rl


def _make_scan(root, day, hms):
    d = os.path.join(str(root), 'Data', day, f'data_{day}_{hms}')
    os.makedirs(d)
    name = os.path.basename(d)
    open(os.path.join(d, f'{name}.h5'), 'wb').close()        # empty h5 (probe -> None, ok)
    with open(os.path.join(d, f'{name}.json'), 'w') as f:    # config sidecar -> complete
        f.write('{}')
    return d


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(rl._yb_cfg, 'PATH_PREFIX', str(tmp_path))
    # Isolate the persisted enrich cache to the tmp dir so the test never
    # reads/writes/deletes the operator's real LOCALAPPDATA cache file.
    monkeypatch.setenv('YB_RUNS_CACHE_PATH',
                       str(tmp_path / 'runs_enrich_cache.json'))
    rl._ENRICH_CACHE_LOADED = False   # force a fresh (empty) lazy load in-test
    rl.clear_enrich_cache()
    return tmp_path


def test_cache_skips_unchanged_enriches_new_and_force(data_root, monkeypatch):
    calls = []
    orig = rl._enrich_meta
    monkeypatch.setattr(
        rl, '_enrich_meta',
        lambda sd, row: (calls.append(row['scan_id']), orig(sd, row))[1])

    _make_scan(data_root, '20260101', '120000')

    rl.list_runs(use_cache=True)            # cold: enrich the one scan
    assert len(calls) == 1

    rl.list_runs(use_cache=True)            # warm: cache hit, no re-enrich
    assert len(calls) == 1

    _make_scan(data_root, '20260101', '130000')
    rl.list_runs(use_cache=True)            # only the NEW scan is enriched
    assert len(calls) == 2

    n_before = len(calls)
    rl.list_runs(use_cache=True, force=True)  # Full rescan: both re-enriched
    assert len(calls) == n_before + 2


def test_mtime_change_reenriches(data_root, monkeypatch):
    calls = []
    orig = rl._enrich_meta
    monkeypatch.setattr(
        rl, '_enrich_meta',
        lambda sd, row: (calls.append(row['scan_id']), orig(sd, row))[1])

    d = _make_scan(data_root, '20260101', '120000')
    rl.list_runs(use_cache=True)
    assert len(calls) == 1

    # Simulate the live scan growing: bump the .h5 mtime forward.
    h5 = os.path.join(d, os.path.basename(d) + '.h5')
    st = os.stat(h5)
    os.utime(h5, (st.st_atime + 100, st.st_mtime + 100))

    rl.list_runs(use_cache=True)            # mtime changed -> re-enrich
    assert len(calls) == 2


def test_legacy_path_unaffected(data_root, monkeypatch):
    """use_cache defaults to False: every call enriches (old behavior)."""
    calls = []
    orig = rl._enrich_meta
    monkeypatch.setattr(
        rl, '_enrich_meta',
        lambda sd, row: (calls.append(row['scan_id']), orig(sd, row))[1])

    _make_scan(data_root, '20260101', '120000')
    rl.list_runs()
    rl.list_runs()
    assert len(calls) == 2   # no caching when use_cache is not set


def test_list_dates_newest_first(data_root):
    """list_dates returns every data day, newest-first, with no scan walk."""
    assert rl.list_dates() == []                 # empty Data/ -> []
    _make_scan(data_root, '20260101', '120000')
    _make_scan(data_root, '20260103', '090000')
    _make_scan(data_root, '20251231', '235959')
    # A non-date dir (e.g. a stray folder) must be ignored.
    os.makedirs(os.path.join(str(data_root), 'Data', 'notaday'))
    assert rl.list_dates() == ['20260103', '20260101', '20251231']


def test_date_str_restricts_to_one_day(data_root):
    """date_str returns only the chosen day's scans (the picker's date jump)."""
    _make_scan(data_root, '20260101', '120000')
    _make_scan(data_root, '20260103', '090000')
    _make_scan(data_root, '20260103', '100000')

    rows = rl.list_runs(date_str='20260103')
    assert {r['scan_id'] for r in rows} == {'20260103090000', '20260103100000'}

    # A day with no folder -> empty, not an error.
    assert rl.list_runs(date_str='20200101') == []


def test_date_str_only_enriches_that_day(data_root, monkeypatch):
    """The whole point: a date jump must NOT stat/enrich the other days."""
    calls = []
    orig = rl._enrich_meta
    monkeypatch.setattr(
        rl, '_enrich_meta',
        lambda sd, row: (calls.append(row['scan_id']), orig(sd, row))[1])

    _make_scan(data_root, '20260101', '120000')
    _make_scan(data_root, '20260101', '130000')
    _make_scan(data_root, '20260103', '090000')

    rl.list_runs(date_str='20260103')
    assert calls == ['20260103090000']           # only the chosen day enriched


def test_persisted_cache_survives_restart(data_root, monkeypatch):
    """After a listing, a simulated restart (wipe in-memory cache, keep the
    on-disk file) must serve every scan from the persisted file WITHOUT any
    file-opening enrich."""
    _make_scan(data_root, '20260101', '120000')
    _make_scan(data_root, '20260101', '130000')

    rl.list_runs(use_cache=True)                  # cold: enrich + persist to disk
    cache_file = os.environ['YB_RUNS_CACHE_PATH']
    assert os.path.exists(cache_file)             # the small file was written

    # Simulate a run_monitor restart: fresh process = empty in-memory cache,
    # file untouched. The next listing must load the file and enrich nothing.
    rl._ENRICH_CACHE.clear()
    rl._ENRICH_CACHE_LOADED = False
    calls = []
    orig = rl._enrich_meta
    monkeypatch.setattr(
        rl, '_enrich_meta',
        lambda sd, row: (calls.append(row['scan_id']), orig(sd, row))[1])

    rows = rl.list_runs(use_cache=True)
    assert len(rows) == 2
    assert calls == []                            # zero file-opening enriches


def test_dehydrated_scan_is_skipped_not_opened(data_root, monkeypatch):
    """A OneDrive cloud-only placeholder scan (not already cached) must be
    listed from stat only -- never enriched (opening it would block/fail while
    OneDrive is off) -- and flagged dehydrated."""
    _make_scan(data_root, '20260101', '120000')

    monkeypatch.setattr(rl, '_dehydrated_scan', lambda sd: True)
    calls = []
    orig = rl._enrich_meta
    monkeypatch.setattr(
        rl, '_enrich_meta',
        lambda sd, row: (calls.append(row['scan_id']), orig(sd, row))[1])

    rows = rl.list_runs(use_cache=True)
    assert len(rows) == 1
    assert calls == []                            # never opened the file
    assert rows[0].get('dehydrated') is True
    # A dehydrated scan that WAS enriched before (cache hit) must still serve
    # its cached metadata -- add an entry with the matching .h5 mtime and
    # confirm it wins over the dehydrated skip.
    sid = rows[0]['scan_id']
    h5 = os.path.join(rows[0]['scan_dir'], f"data_{rows[0]['date']}_{rows[0]['time']}.h5")
    rl._ENRICH_CACHE[sid] = {
        'mtime': os.stat(h5).st_mtime, 'enriched': {'name': 'CachedScan'}}
    rows2 = rl.list_runs(use_cache=True)
    assert rows2[0].get('name') == 'CachedScan'   # cache hit beats the skip
