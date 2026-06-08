"""Per-scan diagnostic sync from SLM PC → lab-PC HDF5 sidecar.

When a scan finishes, `sync_scan(scan_id, scan_dir)` pulls every diag row
from the SLM PC's ledger and writes them into `<scan_dir>/slm_diag.h5`
alongside the regular MATLAB-written `data_*.h5`. The code-snapshot
manifest is fetched separately and saved as `<scan_dir>/slm_code.json`
— hashes only; bytes stay on the SLM PC, retrievable on demand via
`ondemand.get_protocol_source(scan_id)`.

Design (matches the plan's §Phase 2 schema commitments):

- Sidecar lives next to the scan's HDF5 (`<scan_dir>/slm_diag.h5`) so
  it travels with the data when scans are archived.
- One resizable HDF5 dataset per scalar diag field. Variable-shape
  fields (arrays, dicts) go into a single `diag_json` vlen-string
  column carrying the full row as a JSON blob. Lets us add new diag
  fields server-side without lab-side schema migrations.
- Idempotent: rerunning the sync produces the same files. Resumable:
  if a partial sync wrote rows 0–N, the next attempt fetches
  `?since_seq_id=N` and appends.
- Old scans (pre-Phase-1 deploy) return
  `{synced: False, reason: 'legacy_run'}` — the SLM PC has no ledger
  for them, by design.
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from yb_analysis.slm_sync.client import SlmSyncClient

logger = logging.getLogger(__name__)


# Filenames written into each scan_dir.
DIAG_H5 = 'slm_diag.h5'
CODE_JSON = 'slm_code.json'
GRID_JSON = 'slm_grid.json'   # Phase 4: per-run grid sidecar from SLM PC
# Tiny resume-state sidecar so an interrupted sync can pick up exactly
# where it left off without parsing the HDF5.
SYNC_STATE = '.slm_sync_state.json'

# Numeric diag fields surfaced as their own HDF5 columns for easy
# `pd.read_hdf` / `h5py` access. Anything not in this list still lands
# in `diag_json` so nothing is lost — this is just a convenience layer
# for the common fields. Extend as needed when analysis tooling wants
# direct vectorized access to a new field.
_NUMERIC_DIAG_COLUMNS = (
    'total_ms', 'compute_total_ms', 'paced_total_ms',
    'bits_decode_ms', 'lock_check_ms', 'phase_load_ms',
    'n_loaded', 'n_loaded_total', 'n_dropped', 'n_total_sites',
    'n_sites_model', 'nsteps', 'step_period_ms',
    'aborted_at_frame', 'aborted_at_ms',
    # Phase 5a — two_round disambiguation. -1 sentinel when absent so
    # the column is always finite-int (h5py-friendly).
    'two_round_idx',
)
_BOOL_DIAG_COLUMNS = (
    'aborted', 'wrote_final_phase', 'used_cuda_graph',
    'handoff_idle', 'handoff_protocol',
)
_STRING_DIAG_COLUMNS = (
    'protocol', 'handoff_reason', 'handoff_idle_reason',
    # Phase 5a — "initial" / "final" / "" for non-two-round shots.
    'two_round_phase',
    # Phase 5.5 Track E — the canonical RECEIVED bitstring the SLM
    # server decoded for this shot ('0'/'1' chars, length ==
    # n_init_sites). Lets analysts diff SLM-side vs lab-side bitstrings
    # (different gridLocations / threshold pipelines). OPTIONAL: empty
    # string for legacy shots that predate the field (the surfacing
    # machinery already defaults missing string columns to '').
    'received_bits',
)

# Phase 5a — per-shot rearrangement PATHS, stamped by SLMnet via
# `pairing_extra(loaded_paired, target_paired)` and surfaced here as
# parallel vlen-int64 columns. Their entries index into the SAME row's
# slm_grid.json::init_grid / target_grid arrays respectively (both in
# MATLAB bit order; see plan §Bit-order invariant). Empty arrays for
# rows that didn't carry pairing (legacy / fast-path abort).
_VLEN_INT_PATH_COLUMNS = (
    'loaded_paired',   # idx into init_grid; loaded_paired[i] -> bit-k source site
    'target_paired',   # idx into target_grid; target_paired[i] -> bit-k target site
)

# Schema version history:
#   1 -> 2  (Phase 5a): per-shot path vlen columns + two_round columns.
#   2 -> 3  (Phase 5.5 Track E): /diag/received_bits vlen-string column.
# Readers MUST consult /meta/schema_version (or fall back to feature
# probes like `'two_round_idx' in /diag` / `'received_bits' in /diag`)
# and fall through to `diag_json` JSON-decode for older files. The v2->v3
# upgrade is purely additive: the column is backfilled with '' for rows
# written before the column existed, so legacy files keep working.
SCHEMA_VERSION = 3


def sync_scan(scan_id, scan_dir, *, client=None, sync_code=True,
              sync_grid=True):
    """Pull SLM-side diag + code-snapshot for ``scan_id`` into ``scan_dir``.

    Args:
        scan_id: 14-digit YYYYMMDDHHMMSS from MATLAB (passed verbatim to
                 the SLM endpoint as a string).
        scan_dir: Path to the scan's directory (where data_*.h5 lives).
                  ``slm_diag.h5`` and ``slm_code.json`` will be written
                  here.
        client: Optional ``SlmSyncClient`` instance. Created with defaults
                if omitted.
        sync_code: Whether to also fetch the code-snapshot manifest.

    Returns a status dict::

        {
          'synced':       bool,   # True iff at least one row landed on disk
          'reason':       str,    # 'ok' | 'legacy_run' | 'slm_offline' | 'no_data' | …
          'scan_id':      str,
          'rows_written': int,
          'total_rows':   int,    # current count in the SLM ledger
          'overflow':     bool,
          'diag_path':    str | None,
          'code_path':    str | None,
        }
    """
    scan_dir = Path(scan_dir)
    scan_id = str(scan_id)
    if client is None:
        client = SlmSyncClient()

    diag_path = scan_dir / DIAG_H5
    state_path = scan_dir / SYNC_STATE
    code_path = scan_dir / CODE_JSON
    grid_path = scan_dir / GRID_JSON

    status = {
        'synced': False,
        'reason': '',
        'scan_id': scan_id,
        'rows_written': 0,
        'total_rows': 0,
        'overflow': False,
        'diag_path': None,
        'code_path': None,
        'grid_path': None,    # Phase 4: per-run grid sidecar
    }

    # Where did the last sync leave off?
    since_seq_id = _read_resume_state(state_path)

    diag = client.get_diag(scan_id, since_seq_id=since_seq_id)
    if diag is None:
        status['reason'] = 'slm_offline'
        logger.info('slm_sync %s: SLM PC unreachable', scan_id)
        return status

    entries = diag.get('entries') or []
    status['total_rows'] = diag.get('count', 0)
    status['overflow'] = bool(diag.get('overflow', False))

    if not entries and status['total_rows'] == 0 and since_seq_id is None:
        # Server confirms there's never been any diag for this scan_id.
        # Could be a pre-Phase-1 run, or a scan that didn't use the SLM.
        status['reason'] = 'no_data'
        return status

    if entries:
        scan_dir.mkdir(parents=True, exist_ok=True)
        _append_rows_to_h5(diag_path, entries)
        last_seq_id = max(
            (r.get('seq_id') for r in entries
             if isinstance(r.get('seq_id'), int)),
            default=since_seq_id or 0)
        _write_resume_state(state_path, last_seq_id)
        status['rows_written'] = len(entries)
        status['diag_path'] = str(diag_path)

    # Code-snapshot manifest (optional, runs after the diag is safe).
    if sync_code:
        try:
            manifest = client.get_code_manifest(scan_id)
        except Exception as e:
            logger.warning('slm_sync %s: code manifest fetch failed: %s',
                           scan_id, e)
            manifest = None
        if manifest is not None:
            try:
                scan_dir.mkdir(parents=True, exist_ok=True)
                _write_code_json(code_path, manifest)
                status['code_path'] = str(code_path)
            except OSError as e:
                logger.warning('slm_sync %s: code json write failed: %s',
                               scan_id, e)

    # Per-run grid sidecar (Phase 4 addition). The endpoint is only
    # present on SLM builds that include the rearrange_grid_sidecar
    # writer (upstream 2b4e179). Missing endpoint -> None, no error.
    if sync_grid:
        try:
            grid_payload = client.get_grid_sidecar(scan_id)
        except Exception as e:
            logger.warning('slm_sync %s: grid sidecar fetch failed: %s',
                           scan_id, e)
            grid_payload = None
        if grid_payload is not None:
            try:
                scan_dir.mkdir(parents=True, exist_ok=True)
                _write_grid_json(grid_path, grid_payload)
                status['grid_path'] = str(grid_path)
            except OSError as e:
                logger.warning('slm_sync %s: grid json write failed: %s',
                               scan_id, e)

    status['synced'] = (status['rows_written'] > 0
                        or status['code_path'] is not None
                        or status['grid_path'] is not None)
    status['reason'] = 'ok' if status['synced'] else 'no_data'
    return status


def mark_legacy_run(scan_dir):
    """Stamp the resume state with `reason='legacy_run'` so future sync
    attempts on pre-Phase-1 scans short-circuit instead of polling the
    SLM PC on every dashboard tick."""
    Path(scan_dir).mkdir(parents=True, exist_ok=True)
    state_path = Path(scan_dir) / SYNC_STATE
    state_path.write_text(
        json.dumps({'last_seq_id': None, 'reason': 'legacy_run'}),
        encoding='utf-8')


# ---------------------------------------------------------------------------
# Schedule-from-DataManager helper
# ---------------------------------------------------------------------------


_SYNC_LOCK = threading.Lock()  # one sync at a time across the lab-PC process


def sync_scan_async(scan_id, scan_dir, **kw):
    """Run sync_scan in a daemon thread so DataManager.save_data isn't
    blocked by network I/O.

    Returns the Thread so the caller can `.join` in tests. In production
    the thread is fire-and-forget.
    """
    def _run():
        with _SYNC_LOCK:
            try:
                status = sync_scan(scan_id, scan_dir, **kw)
                logger.info('slm_sync %s: %s (rows=%d)',
                            scan_id, status['reason'], status['rows_written'])
            except Exception:
                logger.exception('slm_sync %s: unexpected failure', scan_id)
    t = threading.Thread(
        target=_run, name=f'slm-sync-{scan_id}', daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# HDF5 sidecar I/O
# ---------------------------------------------------------------------------


def _ensure_h5(path):
    """Open `slm_diag.h5` for append; create empty datasets if absent.

    Datasets layout (schema v2 — Phase 5a):
      /diag/seq_id          int64, resizable
      /diag/retry_count     int64, resizable
      /diag/ts_epoch        float64, resizable
      /diag/ts_iso          vlen str, resizable
      /diag/run_id          vlen str, resizable
      /diag/client_id       vlen str, resizable
      /diag/<numeric>       float64, resizable     (per _NUMERIC_DIAG_COLUMNS)
      /diag/<bool>          uint8, resizable       (per _BOOL_DIAG_COLUMNS)
      /diag/<str>           vlen str, resizable    (per _STRING_DIAG_COLUMNS)
      /diag/loaded_paired   vlen int64, resizable  (per-shot src bit indices)
      /diag/target_paired   vlen int64, resizable  (per-shot tgt bit indices)
      /diag/diag_json       vlen str, resizable    (full row JSON-encoded)
      /meta/scan_id         attr
      /meta/schema_version  attr  (1 = pre-Phase-5a, 2 = with path columns)

    v1 → v2 upgrade: existing files get the new columns added with
    defaults backfilled for older rows. Path columns get empty vlen
    arrays for legacy rows; two_round_idx gets -1; two_round_phase
    gets the empty string. Idempotent under multiple sync runs.
    """
    f = h5py.File(path, 'a')
    if 'meta' not in f:
        f.create_group('meta')
        f['meta'].attrs['schema_version'] = SCHEMA_VERSION
    if 'diag' not in f:
        diag = f.create_group('diag')
        vlen_str = h5py.string_dtype(encoding='utf-8')
        vlen_int = h5py.vlen_dtype(np.int64)
        # Core columns
        diag.create_dataset('seq_id', (0,), maxshape=(None,), dtype='i8')
        diag.create_dataset('retry_count', (0,), maxshape=(None,), dtype='i8')
        diag.create_dataset('ts_epoch', (0,), maxshape=(None,), dtype='f8')
        diag.create_dataset('ts_iso', (0,), maxshape=(None,), dtype=vlen_str)
        diag.create_dataset('run_id', (0,), maxshape=(None,), dtype=vlen_str)
        diag.create_dataset('client_id', (0,), maxshape=(None,), dtype=vlen_str)
        # Surfaced numeric / bool / string diag columns
        for k in _NUMERIC_DIAG_COLUMNS:
            diag.create_dataset(k, (0,), maxshape=(None,), dtype='f8')
        for k in _BOOL_DIAG_COLUMNS:
            diag.create_dataset(k, (0,), maxshape=(None,), dtype='u1')
        for k in _STRING_DIAG_COLUMNS:
            diag.create_dataset(k, (0,), maxshape=(None,), dtype=vlen_str)
        # Phase 5a — per-shot path columns. vlen int64; one row per
        # ledger entry, each row an array of variable length (= number
        # of paired atoms in that shot). See plan §Bit-order invariant.
        for k in _VLEN_INT_PATH_COLUMNS:
            diag.create_dataset(k, (0,), maxshape=(None,), dtype=vlen_int)
        # Full row as JSON — captures everything not surfaced above
        # (arrays like compute_ms[], Zernike coeffs, future fields, …).
        diag.create_dataset('diag_json', (0,), maxshape=(None,), dtype=vlen_str)
    else:
        # Existing file: upgrade in place when columns are missing
        # (v1 → v2 schema). Read-only callers see the new columns
        # populated only for rows appended after the upgrade; older
        # rows keep their empty defaults. Safe under multiple sync
        # runs because h5py raises if the dataset already exists.
        diag = f['diag']
        vlen_str = h5py.string_dtype(encoding='utf-8')
        vlen_int = h5py.vlen_dtype(np.int64)
        existing_n = diag['seq_id'].shape[0] if 'seq_id' in diag else 0
        for k in _NUMERIC_DIAG_COLUMNS:
            if k not in diag:
                ds = diag.create_dataset(
                    k, (existing_n,), maxshape=(None,), dtype='f8')
                if existing_n:
                    # `two_round_idx` defaults to -1 (sentinel: not a
                    # two-round shot) so the column stays integer-valued.
                    ds[...] = -1.0 if k == 'two_round_idx' else np.nan
        for k in _BOOL_DIAG_COLUMNS:
            if k not in diag:
                ds = diag.create_dataset(
                    k, (existing_n,), maxshape=(None,), dtype='u1')
                if existing_n:
                    ds[...] = 0
        for k in _STRING_DIAG_COLUMNS:
            if k not in diag:
                ds = diag.create_dataset(
                    k, (existing_n,), maxshape=(None,), dtype=vlen_str)
                if existing_n:
                    ds[...] = [''] * existing_n
        for k in _VLEN_INT_PATH_COLUMNS:
            if k not in diag:
                ds = diag.create_dataset(
                    k, (existing_n,), maxshape=(None,), dtype=vlen_int)
                if existing_n:
                    empty = np.array([], dtype=np.int64)
                    for i in range(existing_n):
                        ds[i] = empty
        if 'meta' in f and f['meta'].attrs.get('schema_version', 1) < SCHEMA_VERSION:
            f['meta'].attrs['schema_version'] = SCHEMA_VERSION
    return f


def _append_rows_to_h5(path, entries):
    """Append `entries` to the HDF5 sidecar. Resizes every dataset by
    len(entries) and writes the new tail slice. Atomic per-dataset; if
    the write fails halfway through, the next sync resumes via
    `since_seq_id`.
    """
    if not entries:
        return
    n = len(entries)
    with _ensure_h5(path) as f:
        diag = f['diag']

        def _append(name, values):
            ds = diag[name]
            old = ds.shape[0]
            ds.resize((old + n,))
            ds[old:old + n] = values

        _append('seq_id', [_safe_int(r.get('seq_id')) for r in entries])
        _append('retry_count',
                [_safe_int(r.get('retry_count'), default=0) for r in entries])
        _append('ts_epoch',
                [float(r.get('ts_epoch') or 0.0) for r in entries])
        _append('ts_iso', [str(r.get('ts_iso') or '') for r in entries])
        _append('run_id', [str(r.get('run_id') or '') for r in entries])
        _append('client_id', [str(r.get('client_id') or '') for r in entries])

        for col in _NUMERIC_DIAG_COLUMNS:
            # `two_round_idx` carries an integer index per shot; absent
            # rows default to -1 (not NaN) so the column stays h5py-int
            # friendly downstream. Other numeric columns NaN-default.
            default = -1.0 if col == 'two_round_idx' else None
            _append(col, [_safe_float(((r.get('diag') or {}).get(col)),
                                       default=default)
                          for r in entries])
        for col in _BOOL_DIAG_COLUMNS:
            _append(col, [1 if (r.get('diag') or {}).get(col) else 0
                          for r in entries])
        for col in _STRING_DIAG_COLUMNS:
            _append(col, [str((r.get('diag') or {}).get(col) or '')
                          for r in entries])
        # Phase 5a — per-shot path index arrays. Build a list-of-arrays
        # and assign element-by-element so h5py writes them as vlen.
        # Bit-order invariant: each int indexes into the SAME row's
        # slm_grid.json::init_grid (loaded_paired) or target_grid
        # (target_paired); see plan §Bit-order invariant.
        for col in _VLEN_INT_PATH_COLUMNS:
            ds = diag[col]
            old = ds.shape[0]
            ds.resize((old + n,))
            for j, r in enumerate(entries):
                ds[old + j] = _safe_int_array((r.get('diag') or {}).get(col))
        # Full row JSON — preserves arrays + unknown future fields.
        _append('diag_json',
                [json.dumps(r, default=str) for r in entries])


def _safe_int(v, default=None):
    if v is None:
        return default if default is not None else -1
    try:
        return int(v)
    except (TypeError, ValueError):
        return default if default is not None else -1


def _safe_float(v, default=None):
    if v is None:
        return np.nan if default is None else float(default)
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return np.nan if default is None else float(default)
    except (TypeError, ValueError):
        return np.nan if default is None else float(default)


def _safe_int_array(v):
    """Coerce a diag list field into an int64 numpy array, robustly.

    Used for Phase 5a's vlen-int path columns. Accepts list / tuple /
    np.ndarray; returns an empty int64 array for None / wrong shape /
    non-numeric content (legacy runs, fast-path-abort shots, …).
    """
    if v is None:
        return np.array([], dtype=np.int64)
    try:
        arr = np.asarray(v).ravel()
        if arr.size == 0:
            return np.array([], dtype=np.int64)
        return arr.astype(np.int64, casting='unsafe')
    except (TypeError, ValueError):
        return np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# Resume state + code JSON
# ---------------------------------------------------------------------------


def _read_resume_state(path):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if d.get('reason') == 'legacy_run':
        # Sentinel: signal to sync_scan that this is a pre-Phase-1 run.
        # We return a special marker; sync_scan checks d['reason'] via
        # _is_marked_legacy.
        return None
    last = d.get('last_seq_id')
    return last if isinstance(last, int) else None


def _write_resume_state(path, last_seq_id):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        json.dumps({'last_seq_id': last_seq_id,
                    'updated_iso': datetime.now().isoformat(
                        timespec='seconds')}),
        encoding='utf-8')
    os.replace(tmp, path)


def _write_code_json(path, manifest_response):
    """Write the SLM-PC code-snapshot manifest to disk.

    The body we get from `/slm/runs/{scan_id}/code` includes a `manifest`
    sub-dict with the full hashes + git_state + per-file metadata. We
    persist the response verbatim plus a top-level `synced_at_iso` so an
    analyst can tell when the snapshot was fetched.
    """
    payload = dict(manifest_response)
    payload['synced_at_iso'] = datetime.now().isoformat(timespec='seconds')
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str),
                   encoding='utf-8')
    os.replace(tmp, path)


def _write_grid_json(path, grid_payload):
    """Write the SLM-PC per-run grid sidecar to disk (Phase 4).

    The body we get from `/slm/runs/{scan_id}/grid_sidecar` carries the
    EXACT derived+reordered grid (init/target knm coords in bit order +
    gridLocations reference + grid_rotation + affine diag) the SLM
    server commanded for this run. We persist it verbatim plus a top-
    level ``synced_at_iso`` so lab-side run_analysis.py can replay the
    scoring lattice without re-deriving from the WGS phase.
    """
    payload = dict(grid_payload)
    payload['synced_at_iso'] = datetime.now().isoformat(timespec='seconds')
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str),
                   encoding='utf-8')
    os.replace(tmp, path)
