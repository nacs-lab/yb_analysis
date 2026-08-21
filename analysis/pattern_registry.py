"""Per-loading-pattern grid registry (loading-pattern affine migration).

Each loadable WGS pattern (e.g. ``33x33_uniform``, ``3270_z4eq4``) gets a
record holding the SLM-server-extracted trap positions + per-site phases
(in the shared column-major order), the lattice geometry, and the inputs
needed to re-derive it. This is the **source of truth** for "where are the
traps for pattern X" — the global SLM→camera affine then maps the stored
knm positions into camera pixels (see :mod:`affine_transform`).

Records live one-per-pattern under
``<PATH_PREFIX>/yb_dashboard_state/patterns/<name>/record.json`` (override
the base dir via ``$YB_PATTERNS_DIR``). Persistence mirrors
:mod:`yb_analysis.analysis.run_groups` (thread-safe, atomic write).

``record.json`` schema::

    {"name": "33x33_uniform",
     "base_phase_path": "phase/base/33x33_uniform.pt",  # server-side
     "legacy_zerniked": false, "baked_zernike": null,   # re-derive inputs
     "base_sha256": "8506...", "default_loading_zernike": null,
     "order": "col", "fft_shape": [4096, 4096],
     "threshold": 0.30, "min_dist": null,
     "n_sites": 1068,
     "knm": [[y, x], ...],            # (N,2) knm-1024-space, (y,x)
     "knm_offset": [dy, dx] | null,   # OPTIONAL per-pattern registration offset,
                                      # knm-px, (y,x). See below.
     "knm_offset_note": "..." | null, # provenance: who measured it, from which scan
     "phases": [...],                 # (N,) per-site phase (rad)
     "lattice": {rows, cols, n_rows, n_cols, pitch_x, pitch_y,
                 row_basis, col_basis, tilt_deg, n_missing, x0, y0},
     # --- 3-D (present/non-null ONLY for a 3-D extraction; None for 2-D) ---
     "planes_z_rad": [...] | null,    # declared layer depths (ANSI rad)
     "is_3d": false,                  # true iff planes_z_rad was supplied
     "z_rad": [...] | null,           # (N,) per-site depth (rad), layer-major
     "positions_knm3d": [[y, x, z], ...] | null,   # (N,3), layer-major
     "n_per_plane": [n0, n1, ...] | null,          # sites per declared layer
     "plane_of_site": [...] | null,                # (N,) layer index per site
     "source_endpoint": "/slm/initialize_loading_pattern",
     "created_iso": "...", "updated_iso": "..."}

The 2-D ``knm`` stays (N,2) in BOTH cases — every existing consumer reads
only that, so a 3-D record is a strict superset and 2-D detection is
unchanged. Site order for a 3-D record is LAYER-MAJOR (declared plane order,
then within-layer ``order``).

``knm_offset`` -- the per-pattern registration offset
----------------------------------------------------
The GLOBAL affine maps knm -> camera for EVERY pattern, so it can only ever
carry effects that are common to all of them (camera/optics drift). A residual
that belongs to ONE pattern -- its extracted coordinate origin sitting a couple
of knm-px away from where the physical array actually lands -- has nowhere to
live in that model, and dragging the shared affine to absorb it makes every
OTHER pattern wrong by the same amount. Measured 2026-08-11: with one affine
fitting both ``17x17_20um`` (0.35 px rms) and ``33x33_feedback11`` (0.47 px rms
over 903 sites) at the SAME scale (0.06 %) and rotation (0.0013 deg), the two
still needed translations 4.94 cam-px apart in Y -- a per-pattern translation
and nothing else.

``knm_offset`` is that translation, in knm-px, registry ``(y, x)`` order, and
it is **added to the stored ``knm`` at read time**::

    knm_effective = knm + knm_offset

Applied by :func:`affine_transform.canonical_knm` and its engine-side mirror
``pyctrl.lib.pattern_grid._canonical_knm``, i.e. BEFORE anything -- grid build,
affine fit, bootstrap, run_analysis -- ever sees the coordinates. So the affine
is fit **blind to it**: a pattern carrying offset ``(10, 0)`` is fit exactly as
if its sites had genuinely been extracted 10 knm-px away, and the affine stays a
pure global-drift fit. Nothing downstream needs to know the offset exists.

Null/absent means ``(0, 0)`` -- every existing record is unchanged. The offset
is PRESERVED across a server refresh (the server derives positions from the
phase and knows nothing about it) and is deliberately NOT part of the refresh
cache key.

Set it with :func:`set_knm_offset`, which is additive-aware and records
provenance in ``knm_offset_note``.

Public API::

    list_patterns()            -> {name: <compact metadata>}
    get_pattern(name)          -> full record | None
    write_pattern(record)      -> None
    delete_pattern(name)       -> bool
    fetch_or_refresh_pattern(name, *, base_phase_path, ...) -> record | None
    get_knm_offset(record)     -> (2,) float ndarray, (dy, dx); zeros if unset
    set_knm_offset(name, dy, dx, *, note=None, add=False) -> record
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from yb_analysis import config as _yb_cfg

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_RECORD_FILENAME = 'record.json'
# Big per-site arrays omitted from list_patterns() metadata. The 3-D arrays
# (positions_knm3d/z_rad/plane_of_site) are N-length too, so they're dropped
# from the compact view alongside the 2-D ones.
_BIG_KEYS = ('knm', 'phases', 'positions_knm3d', 'z_rad', 'plane_of_site')


def _patterns_dir() -> Path:
    """Base dir holding ``<name>/record.json``. Override $YB_PATTERNS_DIR."""
    env = os.environ.get('YB_PATTERNS_DIR')
    if env:
        return Path(env)
    return Path(_yb_cfg.PATH_PREFIX) / 'yb_dashboard_state' / 'patterns'


def _sanitize_name(name: str) -> str:
    """Filesystem-safe pattern name: keep [A-Za-z0-9._-], no separators."""
    name = (name or '').strip()
    if not name:
        raise ValueError('pattern name required')
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    if safe in ('', '.', '..'):
        raise ValueError(f'invalid pattern name: {name!r}')
    return safe


def _pattern_dir(name: str) -> Path:
    return _patterns_dir() / _sanitize_name(name)


def _record_path(name: str) -> Path:
    return _pattern_dir(name) / _RECORD_FILENAME


def pattern_threshold_path(name: str) -> Path:
    """Per-pattern threshold store (same layout as the day-folder
    threshold.mat: thresholds, infidelities, gaussFitsStruct)."""
    return _pattern_dir(name) / 'threshold.mat'


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _read(name: str) -> Optional[dict]:
    p = _record_path(name)
    if not p.is_file():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning('pattern_registry: read %s failed: %s', p, ex)
        return None


def _write(record: dict) -> None:
    p = _record_path(record['name'])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, p)


def _compact(record: dict) -> dict:
    """Metadata view for list endpoints — drop big per-site arrays and
    trim the lattice's N-length row/col lists."""
    meta = {k: v for k, v in record.items() if k not in _BIG_KEYS}
    lat = record.get('lattice')
    if isinstance(lat, dict):
        meta['lattice'] = {k: lat.get(k) for k in (
            'n_rows', 'n_cols', 'n_missing', 'pitch_x', 'pitch_y', 'tilt_deg')}
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_patterns() -> Dict[str, dict]:
    """{name: compact-metadata} for every registered pattern."""
    base = _patterns_dir()
    out: Dict[str, dict] = {}
    if not base.is_dir():
        return out
    with _LOCK:
        for child in sorted(base.iterdir()):
            rec_path = child / _RECORD_FILENAME
            if not rec_path.is_file():
                continue
            try:
                with open(rec_path, 'r', encoding='utf-8') as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(rec, dict) and rec.get('name'):
                out[rec['name']] = _compact(rec)
    return out


def get_pattern(name: str) -> Optional[dict]:
    """Full record (incl. knm + phases) or None."""
    with _LOCK:
        return _read(name)


def write_pattern(record: dict) -> None:
    if not record.get('name'):
        raise ValueError('record must have a name')
    with _LOCK:
        _write(record)


def get_knm_offset(record) -> 'np.ndarray':
    """This pattern's registration offset as ``(dy, dx)`` knm-px; zeros if unset.

    Accepts a record dict or a pattern NAME. Never raises on a malformed value --
    a bad offset must degrade to "no offset", not break grid resolution for the
    whole run (the same discipline :func:`affine_transform.canonical_knm` uses).
    """
    import numpy as np
    if isinstance(record, str):
        record = get_pattern(record) or {}
    raw = record.get('knm_offset') if isinstance(record, dict) else None
    if raw is None:
        return np.zeros(2, dtype=np.float64)
    try:
        off = np.asarray(raw, dtype=np.float64).reshape(2)
    except (ValueError, TypeError):
        logger.warning('pattern_registry: malformed knm_offset %r on %r; ignoring',
                       raw, (record or {}).get('name'))
        return np.zeros(2, dtype=np.float64)
    if not np.isfinite(off).all():
        logger.warning('pattern_registry: non-finite knm_offset %r on %r; ignoring',
                       raw, (record or {}).get('name'))
        return np.zeros(2, dtype=np.float64)
    return off


def set_knm_offset(name: str, dy: float, dx: float, *, note=None,
                   add: bool = False) -> dict:
    """Set (or accumulate) a pattern's registration offset, in knm-px ``(y, x)``.

    ``add=True`` ADDS to the existing offset, which is what you want after
    measuring a residual on top of an already-offset pattern -- measuring again
    and calling with ``add=False`` would double-count whatever the first offset
    already removed.

    ``note`` should say who measured it and from which scan; it is stored in
    ``knm_offset_note`` so the number is never a bare magic constant.
    """
    import numpy as np
    with _LOCK:
        rec = _read(name)
        if rec is None:
            raise KeyError('pattern %r not in the registry' % (name,))
        base = np.zeros(2, dtype=np.float64)
        if add:
            base = get_knm_offset(rec)
        off = base + np.asarray([float(dy), float(dx)], dtype=np.float64)
        if not np.isfinite(off).all():
            raise ValueError('knm_offset must be finite; got %r' % (off.tolist(),))
        rec['knm_offset'] = [float(off[0]), float(off[1])]
        rec['knm_offset_note'] = note
        rec['updated_iso'] = _now_iso()
        _write(rec)
    logger.info('pattern_registry: %s knm_offset -> (%.4f, %.4f) knm-px%s',
                name, off[0], off[1], (' [%s]' % note) if note else '')
    return rec


def delete_pattern(name: str) -> bool:
    with _LOCK:
        p = _record_path(name)
        if not p.is_file():
            return False
        try:
            p.unlink()
            # Remove the now-empty pattern dir (best-effort).
            try:
                p.parent.rmdir()
            except OSError:
                pass
            return True
        except OSError as ex:
            logger.warning('pattern_registry: delete %s failed: %s', name, ex)
            return False


def _params_match(rec: dict, *, base_phase_path, order, fft_shape, threshold,
                  min_dist, legacy_zerniked, baked_zernike,
                  planes_z_rad=None) -> bool:
    """True if a cached record was derived with the same inputs. The base
    `.pt` content (sha) can only change server-side, so a same-path record
    is treated as valid unless the caller passes force=True."""
    if rec.get('base_phase_path') != base_phase_path:
        return False
    if str(rec.get('order')) != str(order):
        return False
    if list(rec.get('fft_shape') or []) != list(fft_shape):
        return False
    if float(rec.get('threshold', -1)) != float(threshold):
        return False
    if rec.get('min_dist') != min_dist:
        return False
    if bool(rec.get('legacy_zerniked', False)) != bool(legacy_zerniked):
        return False
    if list(rec.get('baked_zernike') or []) != list(baked_zernike or []):
        return False
    # 3-D layer depths are part of the extraction identity: a 2-D cache hit
    # must NOT satisfy a 3-D request (and vice versa). Normalise None/missing
    # -> [] so a pre-3-D record still matches a 2-D request (no needless
    # re-derive on deploy). Compare element-wise as floats to ignore
    # int-vs-float / list-vs-tuple drift.
    cached_p = [float(z) for z in (rec.get('planes_z_rad') or [])]
    want_p = [float(z) for z in (planes_z_rad or [])]
    if cached_p != want_p:
        return False
    return bool(rec.get('base_sha256'))  # only trust a successfully-derived rec


def load_pattern_thresholds(name):
    """Load per-pattern detection thresholds, or None. Mirrors the day-folder
    threshold.mat parsing in data_manager._load_from_disk. Returns
    {'thresholds': (N,), 'infidelities': (N,), 'gauss_fits': list|None}."""
    p = pattern_threshold_path(name)
    if not p.is_file():
        return None
    try:
        import numpy as np
        from scipy.io import loadmat
        from yb_analysis.io.preload import _parse_gauss_fits_struct
        td = loadmat(str(p), squeeze_me=True)
        thr = np.asarray(td['thresholds'], dtype=np.float64).ravel()
        inf = np.asarray(td.get('infidelities', np.full(len(thr), np.nan)),
                         dtype=np.float64).ravel()
        gf = _parse_gauss_fits_struct(td.get('gaussFitsStruct'))
        return {'thresholds': thr, 'infidelities': inf, 'gauss_fits': gf}
    except Exception as ex:  # noqa: BLE001
        logger.warning('load_pattern_thresholds(%s) failed: %s', name, ex)
        return None


def save_pattern_thresholds(name, mat_dict):
    """Write <pattern>/threshold.mat (keys: thresholds, infidelities,
    gaussFitsStruct) — the same dict data_manager writes to the day folder, so
    the formats stay identical and the live refit self-calibrates per pattern."""
    try:
        from scipy.io import savemat
        p = pattern_threshold_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(p), mat_dict)
    except Exception as ex:  # noqa: BLE001
        logger.warning('save_pattern_thresholds(%s) failed: %s', name, ex)


def pattern_site_mask_path(name: str) -> Path:
    """Per-pattern site-mask .npy store (bool[n_sites] or int indices),
    alongside record.json. Used when a pattern's record `site_mask` is the
    literal string 'file' (i.e. the mask lives in this .npy)."""
    return _pattern_dir(name) / 'site_mask.npy'


def load_pattern_site_mask(name):
    """Return this pattern's configured site-mask SPEC, or None.

    The spec is whatever ``record.json['site_mask']`` holds -- a registered mask
    NAME, a .npy path, or the literal 'file' (meaning the mask lives in this
    pattern's own ``site_mask.npy``, whose absolute path is returned). None /
    missing key -> None (analyze the full array). The spec is resolved to a bool
    array later by ``site_mask.resolve_site_mask`` against the scan's n_sites."""
    if not name:
        return None
    rec = get_pattern(name)
    if not rec:
        return None
    spec = rec.get('site_mask')
    if spec is None:
        return None
    if isinstance(spec, str) and spec.strip().lower() == 'file':
        p = pattern_site_mask_path(name)
        return str(p) if p.is_file() else None
    return spec


def set_pattern_site_mask(name, spec):
    """Set (or clear, spec=None) this pattern's site-mask spec in record.json.
    ``spec`` is a registered name, a .npy path, 'file' (mask in the pattern's
    own site_mask.npy), or None to remove. Leaves other record fields intact."""
    with _LOCK:
        rec = _read(name) or {'name': name}
        if spec is None:
            rec.pop('site_mask', None)
        else:
            rec['site_mask'] = spec
        rec['updated_iso'] = _now_iso()
        _write(rec)


def save_pattern_site_mask_file(name, mask):
    """Write ``mask`` (bool[n_sites], pattern detection-column order) to this
    pattern's own ``site_mask.npy`` and point ``record.json['site_mask']`` at
    it (the literal 'file'). Returns the .npy path. The dashboard's lasso
    mask editor saves through here."""
    import numpy as np
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0 or not m.any():
        raise ValueError('site mask must keep at least one site')
    p = pattern_site_mask_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), m)
    set_pattern_site_mask(name, 'file')
    return p


def fetch_or_refresh_pattern(name, *, base_phase_path,
                             default_loading_zernike=None, order='col',
                             fft_shape=(4096, 4096), threshold=0.30,
                             min_dist=None, legacy_zerniked=False,
                             baked_zernike=None, planes_z_rad=None,
                             client=None, force=False):
    """Return the registry record for ``name``, deriving it from the SLM
    server only when needed.

    If a cached record exists with matching inputs and ``force`` is False,
    it is returned WITHOUT any network call. Otherwise POST
    ``/slm/initialize_loading_pattern`` (extraction only — ``write_to_slm``
    is the scan's job, not the registry's), build and persist the record,
    and return it. On SLM error, fall back to the last-known-good record if
    present (logged); otherwise re-raise. The one exception that is NEVER
    masked by the cache is :class:`~yb_analysis.slm_sync.client.SlmPhaseNotFound`
    (the server has no such phase file) — it always propagates so a misspelled
    phase path is surfaced rather than silently served a stale grid.

    ``planes_z_rad`` (OPTIONAL, list of ANSI ``2*rho^2-1`` radians): when
    given (non-empty) the server does a 3-D, per-plane extraction and the
    record gains the 3-D fields (``is_3d``/``z_rad``/``positions_knm3d``/
    ``planes_z_rad``/``n_per_plane``/``plane_of_site``). It is part of the
    cache key, so changing it forces a re-derive; ``None`` is the legacy 2-D
    path and leaves the 3-D fields null.
    """
    fft_shape = (int(fft_shape[0]), int(fft_shape[1]))
    # Normalise once so the cache key, the request, and the stored record all
    # agree (an empty list means "no planes" == 2-D).
    planes = ([float(z) for z in planes_z_rad] if planes_z_rad else None)
    existing = get_pattern(name)
    if existing and not force and _params_match(
            existing, base_phase_path=base_phase_path, order=order,
            fft_shape=fft_shape, threshold=threshold, min_dist=min_dist,
            legacy_zerniked=legacy_zerniked, baked_zernike=baked_zernike,
            planes_z_rad=planes):
        return existing

    from yb_analysis.slm_sync.client import SlmPhaseNotFound
    if client is None:
        from yb_analysis.slm_sync.client import SlmSyncClient
        client = SlmSyncClient()

    try:
        resp = client.initialize_loading_pattern(
            phase_path=base_phase_path, loading_zernike=None,
            baked_zernike=baked_zernike, legacy_zerniked=legacy_zerniked,
            order=order, fft_shape=fft_shape, threshold=threshold,
            min_dist=min_dist, write_to_slm=False, name=name,
            planes_z_rad=planes)
    except SlmPhaseNotFound:
        # The phase file genuinely does not exist on the SLM server. NEVER mask
        # this with a stale cache: a cached grid is for a phase that is no longer
        # there, so silently reusing it is exactly the failure we want surfaced.
        raise
    except Exception as ex:  # noqa: BLE001 — network/HTTP; fall back if we can
        if existing:
            logger.warning('pattern_registry: refresh %s failed (%s); '
                           'using last-known-good record', name, ex)
            return existing
        raise

    if resp is None:  # SLM unreachable
        if existing:
            logger.warning('pattern_registry: SLM unreachable refreshing %s; '
                           'using last-known-good record', name)
            return existing
        return None

    record = {
        'name': name,
        'base_phase_path': base_phase_path,
        'legacy_zerniked': bool(legacy_zerniked),
        'baked_zernike': list(baked_zernike) if baked_zernike else None,
        'base_sha256': resp.get('base_sha256'),
        'default_loading_zernike': (list(default_loading_zernike)
                                    if default_loading_zernike else None),
        'order': resp.get('order', order),
        'fft_shape': list(fft_shape),
        'threshold': float(threshold),
        'min_dist': min_dist,
        'n_sites': resp.get('n_sites'),
        'knm': resp.get('positions_knm'),
        # PRESERVED across the refresh: the server derives positions from the phase
        # and knows nothing about this pattern's registration offset, so a refresh
        # must not silently drop it (that would re-open the misregistration the
        # offset was measured to close). It is also NOT part of the cache key --
        # changing it must not force a re-extraction.
        'knm_offset': (existing or {}).get('knm_offset'),
        'knm_offset_note': (existing or {}).get('knm_offset_note'),
        'phases': resp.get('phases'),
        'lattice': resp.get('lattice'),
        # 3-D fields: echo what the server returned. For a 2-D request the
        # server leaves these null (or omits them), so this is a no-op vs the
        # old record shape. ``planes_z_rad`` mirrors the REQUESTED planes (the
        # cache key) so a refresh round-trips identically.
        'planes_z_rad': planes,
        'is_3d': bool(resp.get('is_3d')),
        'z_rad': resp.get('z_rad'),
        'positions_knm3d': resp.get('positions_knm3d'),
        'n_per_plane': resp.get('n_per_plane'),
        'plane_of_site': resp.get('plane_of_site'),
        'source_endpoint': '/slm/initialize_loading_pattern',
        'created_iso': (existing or {}).get('created_iso') or _now_iso(),
        'updated_iso': _now_iso(),
    }
    write_pattern(record)
    return record
