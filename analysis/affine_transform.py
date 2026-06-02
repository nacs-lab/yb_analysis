"""Global SLM(knm) -> camera(absolute pixel) affine transform.

One affine ``A`` maps a pattern's simulated knm positions into **absolute
full-sensor** camera pixels. The per-scan crop ROI offset is applied
SEPARATELY (the ROI may change), so ``A`` itself never encodes the crop.
``A`` auto-updates from recent scans behind guardrails (coverage / RMS /
rotation / scale gates, EMA blend, bounded history, rollback).

Convention (reuses :func:`hist_init.fit_affine_csv_to_image`):
    source = knm ``[x, y]``  ->  output = absolute camera ``[Y_abs, X_abs]``
    ``[Y_abs, X_abs]^T = A @ [x_knm, y_knm, 1]^T``
Registry knm is stored as ``[y, x]``; we swap to ``[x, y]`` exactly once via
:func:`_knm_to_xy` (axis-order discipline — the #1 source of bugs here).

Crop discipline: ``apply_affine_cropped(knm, A, roi) = apply_affine(...) -
[Yoff, Xoff]`` where ``roi = [Xoff, Yoff, W, H]``. When fitting from a scan,
detected positions are in CROPPED pixels, so lift them to absolute with
``+ [Yoff, Xoff]`` before the fit.

Persisted at ``<PATH_PREFIX>/yb_dashboard_state/affine_transform.json``
(override ``$YB_AFFINE_PATH``); persistence mirrors
:mod:`yb_analysis.analysis.run_groups`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from yb_analysis import config as _yb_cfg
from yb_analysis.detection.hist_init import (
    fit_affine_csv_to_image, project_csv, detect_grid, sort_grid,
)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# ---- Guardrail constants (module-level so tests/dashboard can read them) ----
MIN_COVERAGE = 0.85          # matched fraction of pattern sites
MIN_PAIRS = 50               # absolute floor (relaxed for tiny arrays)
MAX_RMS_PX = 2.0             # fit residual ceiling (absolute pixels)
MAX_ROT_DEV_DEG = 2.0        # vs current affine
MAX_SCALE_DEV_FRAC = 0.05    # vs current affine
EMA_WEIGHT = 0.25            # blend weight for an accepted (full-refit) candidate
SHIFT_EMA_WEIGHT = 0.5       # blend weight for a translation (drift) update
HISTORY_DEPTH = 10
ROLLBACK_COVERAGE = 0.5
ROLLBACK_RMS_PX = 3.0
# Per-scan drift update (mirrors the live locate_atom grid-shift):
SHIFT_SEARCH_RANGE = 12      # px; cross-correlation search half-width
SHIFT_MIN_SNR = 5.0          # peak-vs-far-shift SNR floor; rejects bad/empty runs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _affine_path() -> Path:
    env = os.environ.get('YB_AFFINE_PATH')
    if env:
        return Path(env)
    return (Path(_yb_cfg.PATH_PREFIX) / 'yb_dashboard_state'
            / 'affine_transform.json')


def _read() -> dict:
    p = _affine_path()
    if not p.is_file():
        return {'current': None, 'history': []}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'current': None, 'history': []}
        data.setdefault('current', None)
        data.setdefault('history', [])
        return data
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning('affine_transform: read %s failed: %s', p, ex)
        return {'current': None, 'history': []}


def _write(data: dict) -> None:
    p = _affine_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def load_affine() -> Optional[dict]:
    """Return the current affine entry ({A, metrics, ...}) or None."""
    with _LOCK:
        return _read().get('current')


def load_matrix() -> Optional[np.ndarray]:
    """Just the current 2x3 matrix as an ndarray, or None."""
    cur = load_affine()
    if not cur or cur.get('A') is None:
        return None
    return np.asarray(cur['A'], dtype=np.float64).reshape(2, 3)


# ---------------------------------------------------------------------------
# Core math (axis-order discipline lives here)
# ---------------------------------------------------------------------------

def _knm_to_xy(knm) -> np.ndarray:
    """Registry knm is ``[y, x]``; affine math wants ``[x, y]``. Swap ONCE."""
    knm = np.asarray(knm, dtype=np.float64).reshape(-1, 2)
    return knm[:, [1, 0]]


def fit_affine(knm_xy, cam_abs_yx):
    """Least-squares 2x3 ``A`` with ``[Y,X]^T = A @ [x,y,1]^T``.

    Returns ``(A (2,3), rms_px float, residuals (N,))``.
    """
    A, resid = fit_affine_csv_to_image(np.asarray(knm_xy), np.asarray(cam_abs_yx))
    rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float('inf')
    return A, rms, resid


def apply_affine(knm_xy, A) -> np.ndarray:
    """knm ``[x,y]`` -> absolute camera ``[Y,X]`` (N,2)."""
    return project_csv(np.asarray(knm_xy), np.asarray(A))


def apply_affine_cropped(knm_xy, A, roi) -> np.ndarray:
    """knm ``[x,y]`` -> CROPPED-frame camera ``[Y,X]`` for ``roi =
    [Xoff,Yoff,W,H]``. The crop offset is applied here, never baked into A."""
    yx = apply_affine(knm_xy, A)
    xoff, yoff = float(roi[0]), float(roi[1])
    return yx - np.array([yoff, xoff], dtype=np.float64)


def decompose(A) -> dict:
    """Rough rotation/scale/shear of the 2x2 linear part (for drift
    monitoring + guardrails — values are consistent, not canonical)."""
    A = np.asarray(A, dtype=np.float64).reshape(2, 3)
    M = A[:, :2]
    sx = float(np.hypot(M[0, 0], M[1, 0]))   # response to knm-x
    sy = float(np.hypot(M[0, 1], M[1, 1]))   # response to knm-y
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    det = float(np.linalg.det(M))
    return {'rotation_deg': rot, 'scale_x': sx, 'scale_y': sy, 'det': det}


# ---------------------------------------------------------------------------
# Correspondence (knm <-> detected camera spots)
# ---------------------------------------------------------------------------

# The four 90°-family axis-swapping orientations mapping knm ``[x,y]`` ->
# camera ``[Y,X]`` (anchored to grid_rotation=90 — we do NOT try 0°/180°).
# Two are proper rotations (det +1), two are reflections (det -1); rotations
# are listed first so ties break toward a non-mirrored fit.
_NINETY_M = (
    np.array([[0.0, 1.0], [-1.0, 0.0]]),   # det +1
    np.array([[0.0, -1.0], [1.0, 0.0]]),   # det +1
    np.array([[0.0, 1.0], [1.0, 0.0]]),    # det -1 (reflection)
    np.array([[0.0, -1.0], [-1.0, 0.0]]),  # det -1 (reflection)
)


def _seed_predict(knm_xy, cam_abs_yx, M, scale, center_knm, center_cam,
                  extra_rot_deg=0.0):
    """Coarse similarity seed mapping knm ``[x,y]`` -> predicted camera
    ``[Y,X]`` for orthonormal ``M`` (a 90°-family orientation), using a
    robust (median) center and a lattice-pitch ``scale``, with a small
    in-plane ``extra_rot_deg`` to absorb the lattice TILT (so even edge
    sites land nearer their true spot than a neighbour). Used only to seed
    nearest-neighbour matching; the final A comes from ``fit_affine``."""
    P = np.asarray(knm_xy, dtype=np.float64)            # [x, y]
    base = (P - center_knm) @ (scale * np.asarray(M, dtype=np.float64)).T
    if extra_rot_deg:
        th = np.radians(extra_rot_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        base = base @ R.T
    return base + center_cam                            # [Y,X]


def _match(pred_yx, cam_abs_yx, tol):
    """Greedy nearest-neighbour match pred->cam within ``tol`` px. Returns
    ``(knm_idx, cam_idx)`` int arrays; each cam spot used at most once
    (closest predicted wins)."""
    from scipy.spatial import cKDTree
    cam = np.asarray(cam_abs_yx, dtype=np.float64)
    pred = np.asarray(pred_yx, dtype=np.float64)
    tree = cKDTree(cam)
    d, j = tree.query(pred, k=1)
    # Resolve collisions: if two predicted map to the same cam, keep closest.
    best = {}
    for i, (di, ji) in enumerate(zip(d, j)):
        if di > tol:
            continue
        if ji not in best or di < best[ji][1]:
            best[ji] = (i, di)
    knm_idx = np.array([v[0] for v in best.values()], dtype=int)
    cam_idx = np.array(list(best.keys()), dtype=int)
    order = np.argsort(knm_idx)
    return knm_idx[order], cam_idx[order]


def _correspond_and_fit(knm_xy, cam_abs_yx, *, seed_A=None,
                        pitch_px=None, iters=3):
    """Match knm sites to detected camera spots and fit ``A``.

    If ``seed_A`` is given (an existing affine), it seeds matching directly
    (no orientation search). Otherwise try the four 90°-family orientations
    (:data:`_NINETY_M`) and keep the one with the most inliers — orientation
    anchored to ``grid_rotation=90`` (we do NOT try 0°/180°; for symmetric
    arrays this still gives a deterministic, consistent result via the listed
    order). Then ICP-refine (coarse-to-fine) for ``iters`` rounds.

    Returns ``(A, rms, knm_idx, cam_idx)`` or ``(None, inf, [], [])``.
    """
    knm_xy = np.asarray(knm_xy, dtype=np.float64)
    cam = np.asarray(cam_abs_yx, dtype=np.float64)
    cam_pitch = pitch_px if pitch_px is not None else _median_spacing(cam)
    knm_pitch = _median_spacing(knm_xy) or 1.0
    # Robust seed: lattice-pitch ratio (insensitive to a few outlier/bright
    # blobs that would inflate an RMS-radius scale) + median centers.
    scale_seed = cam_pitch / knm_pitch
    c_knm = np.median(knm_xy, axis=0)
    c_cam = np.median(cam, axis=0)
    # Two tolerances: fit on a TIGHT inlier set (low rms, drops stray bright
    # blobs); report COVERAGE as the fraction of knm sites the final A places
    # near SOME detected spot (forward match) — robust to detection-centroid
    # imprecision on faint/defocused spots, where a strict fit tol would
    # under-count perfectly-good sites.
    ransac_px = max(2.5, 0.05 * cam_pitch)
    cov_tol = max(4.0, 0.10 * cam_pitch)

    # Coarse-to-fine tolerance: start loose so a seed a few degrees off the
    # true rotation still pairs each site with its CORRECT nearest neighbour
    # (lattice spacing >> seed error), fit to correct the rotation, then
    # tighten.
    tol_factors = [1.5, 1.0, 0.7] + [0.5] * max(0, iters - 3)

    def _refine(pred_yx):
        from scipy.spatial import cKDTree
        A = None
        rms = float('inf')
        for tf in tol_factors:
            ki, ci = _match(pred_yx, cam, tf * cam_pitch)
            if len(ki) < 3:
                continue
            A, rms, _ = fit_affine(knm_xy[ki], cam[ci])
            pred_yx = apply_affine(knm_xy, A)
        if A is None:
            return None, float('inf'), np.array([], int), np.array([], int)
        tree = cKDTree(cam)
        # Tight inliers -> tight fit (low rms).
        d, j = tree.query(apply_affine(knm_xy, A), k=1)
        tight = np.where(d <= ransac_px)[0]
        if len(tight) >= 3:
            A, rms, _ = fit_affine(knm_xy[tight], cam[j[tight]])
        # Forward coverage with the final A.
        d, j = tree.query(apply_affine(knm_xy, A), k=1)
        matched = np.where(d <= cov_tol)[0]
        return A, rms, matched, j[matched]

    if seed_A is not None:
        return _refine(apply_affine(knm_xy, seed_A))

    best = (None, float('inf'), np.array([], dtype=int), np.array([], dtype=int))
    # Tilt search around each 90° orientation: the lattice tilt (a few deg)
    # would otherwise displace edge sites by >half a pitch and mis-pair them.
    for M in _NINETY_M:
        for dt in (0.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0):
            pred = _seed_predict(knm_xy, cam, M, scale_seed, c_knm, c_cam, dt)
            A, rms, ki, ci = _refine(pred)
            # Prefer most inliers, then lowest rms (deterministic; _NINETY_M
            # lists rotations first so ties favour a non-mirrored fit).
            if (len(ki), -rms) > (len(best[2]), -best[1]):
                best = (A, rms, ki, ci)
    return best


def _median_spacing(pts):
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 1.0
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


# ---------------------------------------------------------------------------
# Candidate construction + guardrails
# ---------------------------------------------------------------------------

def _make_candidate(A, rms, n_pairs, n_sites, scan_id, *, bootstrap):
    """Build a candidate dict and decide accept/reject against the gates.

    Bootstrap (or no current affine) relaxes the rotation/scale DEVIATION
    gates (there's nothing to compare to); coverage / pairs / rms /
    non-degenerate always apply.
    """
    coverage = (n_pairs / n_sites) if n_sites else 0.0
    min_pairs = min(MIN_PAIRS, max(3, int(0.5 * n_sites))) if n_sites else MIN_PAIRS
    cand = {
        'A': (np.asarray(A).reshape(2, 3).tolist() if A is not None else None),
        'rms_px': float(rms), 'coverage': float(coverage),
        'n_pairs': int(n_pairs), 'n_sites': int(n_sites),
        'scan_id': scan_id, 'accept': False, 'reason': None,
    }
    if A is None or n_pairs < 3:
        cand['reason'] = 'reject_too_few_pairs'
        return cand
    cand.update(decompose(A))
    if abs(cand['det']) <= 1e-6:
        cand['reason'] = 'reject_degenerate'
        return cand
    if n_pairs < min_pairs:
        cand['reason'] = 'reject_too_few_pairs'
        return cand
    if coverage < MIN_COVERAGE:
        cand['reason'] = 'reject_low_coverage'
        return cand
    if rms > MAX_RMS_PX:
        cand['reason'] = 'reject_high_rms'
        return cand
    cur = load_affine() if not bootstrap else None
    if cur and cur.get('A') is not None:
        c0 = decompose(cur['A'])
        d_rot = abs((cand['rotation_deg'] - c0['rotation_deg'] + 180) % 360 - 180)
        d_scale = max(abs(cand['scale_x'] / max(c0['scale_x'], 1e-9) - 1),
                      abs(cand['scale_y'] / max(c0['scale_y'], 1e-9) - 1))
        if d_rot > MAX_ROT_DEV_DEG:
            cand['reason'] = 'reject_rotation_drift'
            return cand
        if d_scale > MAX_SCALE_DEV_FRAC:
            cand['reason'] = 'reject_scale_drift'
            return cand
    cand['accept'] = True
    cand['reason'] = 'accepted'
    return cand


def propose_update(knm_yx, cam_abs_yx, scan_id, *, bootstrap=False,
                   pitch_px=None):
    """Build the correspondence and a gated candidate affine.

    ``knm_yx`` is the pattern's stored knm positions ``[y,x]`` (registry
    order); ``cam_abs_yx`` are detected atom centroids in ABSOLUTE camera
    pixels ``[Y,X]`` (already lifted by the ROI offset). Returns the
    candidate dict from :func:`_make_candidate`.
    """
    knm_xy = _knm_to_xy(knm_yx)
    n_sites = knm_xy.shape[0]
    seed = None if bootstrap else load_matrix()
    A, rms, ki, ci = _correspond_and_fit(
        knm_xy, cam_abs_yx, seed_A=seed, pitch_px=pitch_px)
    return _make_candidate(A, rms, len(ki), n_sites, scan_id, bootstrap=bootstrap)


def propose_scan_update(images, knm_yx, roi, mask_mat, scan_id, *,
                        search_range=SHIFT_SEARCH_RANGE,
                        min_snr=SHIFT_MIN_SNR):
    """Per-scan DRIFT update: find the global pixel shift of the atoms vs.
    the current affine's predicted grid using the SAME cross-correlation as
    the live grid tracker (:func:`locate_atom.locate_atom_update`), and
    update only the affine TRANSLATION. Rotation/scale are held — recalibrate
    those with :func:`bootstrap_from_scan`.

    This mirrors the proven live grid-shift: ``images`` is the recent image
    stack (the same buffer the live updater uses), ``mask_mat`` the detection
    mask. Safeties against a bad/empty run ruining the affine: reject a shift
    that rails the search window, or a low-prominence heatmap peak; the commit
    is additionally EMA-damped.

    Returns a candidate dict (``mode='shift'``) for :func:`commit_update`.
    """
    from yb_analysis.detection.locate_atom import locate_atom_update
    cand = {'A': None, 'scan_id': scan_id, 'mode': 'shift',
            'n_pairs': None, 'coverage': None, 'rms_px': None,
            'accept': False, 'reason': None}
    A = load_matrix()
    if A is None:
        cand['reason'] = 'reject_no_affine'
        return cand
    knm_xy = _knm_to_xy(knm_yx)
    pred = apply_affine_cropped(knm_xy, A, roi)           # cropped [Y,X]
    img = np.asarray(images, dtype=np.float64)
    img = img[None] if img.ndim == 2 else img
    _, _, dy, dx, heat = locate_atom_update(img, pred, search_range, mask_mat)
    heat = np.asarray(heat, dtype=np.float64)
    # SNR of the peak vs the far-shift (heatmap-edge) level: a real lattice
    # aligns atoms only near the true shift, so the peak sits well above the
    # background-shift ring; a blank/empty run does not.
    if heat.shape[0] > 2 and heat.shape[1] > 2:
        ring = np.concatenate([heat[0, :], heat[-1, :],
                               heat[1:-1, 0], heat[1:-1, -1]])
    else:
        ring = heat.ravel()
    mad = float(np.median(np.abs(ring - np.median(ring)))) * 1.4826 + 1e-9
    snr = float((heat.max() - np.median(ring)) / mad)
    cand.update({'shift_dy': int(dy), 'shift_dx': int(dx), 'snr': snr})
    if abs(dy) >= search_range or abs(dx) >= search_range:
        cand['reason'] = 'reject_shift_railed'
        return cand
    if snr < min_snr:
        cand['reason'] = 'reject_low_snr'
        return cand
    A_new = np.asarray(A, dtype=np.float64).copy()
    A_new[:, 2] += np.array([dy, dx], dtype=np.float64)   # translation [Y,X]
    cand['A'] = A_new.tolist()
    cand.update(decompose(A_new))
    cand['accept'] = True
    cand['reason'] = 'accepted'
    return cand


def commit_update(candidate, *, ema_weight=EMA_WEIGHT) -> dict:
    """Persist an accepted candidate. If a current affine exists, EMA-blend
    the 2x3 elementwise (valid: the deviation gates bound the difference)
    and push the old one onto the bounded history. Returns the new current."""
    if not candidate or not candidate.get('accept'):
        raise ValueError('refusing to commit a non-accepted candidate')
    A_cand = np.asarray(candidate['A'], dtype=np.float64).reshape(2, 3)
    with _LOCK:
        data = _read()
        cur = data.get('current')
        if cur and cur.get('A') is not None:
            A_cur = np.asarray(cur['A'], dtype=np.float64).reshape(2, 3)
            A_new = (1 - ema_weight) * A_cur + ema_weight * A_cand
            data.setdefault('history', []).append(cur)
            data['history'] = data['history'][-HISTORY_DEPTH:]
            created = cur.get('created_iso') or _now_iso()
        else:
            A_new = A_cand
            created = _now_iso()
        entry = {
            'A': A_new.tolist(),
            'created_iso': created, 'updated_iso': _now_iso(),
            'last_scan_id': candidate.get('scan_id'),
            'n_pairs': candidate.get('n_pairs'),
            'coverage': candidate.get('coverage'),
            'rms_px': candidate.get('rms_px'),
        }
        entry.update(decompose(A_new))
        data['current'] = entry
        _write(data)
        return entry


def rollback() -> bool:
    """Restore the most recent history entry as current (e.g. after a bad
    run). Returns True if a rollback happened."""
    with _LOCK:
        data = _read()
        hist = data.get('history') or []
        if not hist:
            return False
        prev = hist.pop()
        prev = dict(prev)
        prev['rolled_back'] = True
        prev['updated_iso'] = _now_iso()
        data['current'] = prev
        data['history'] = hist
        _write(data)
        return True


# ---------------------------------------------------------------------------
# Bootstrap from a scan's averaged loading image
# ---------------------------------------------------------------------------

def bootstrap_from_scan(avg_image, pattern_record, roi, *,
                        spot_sigma=5.0, min_distance=None, scan_id=None,
                        detected_yx=None, n_det_factor=1.1):
    """Fit the FIRST affine: pattern knm <-> detected atoms in ``avg_image``.

    Defocus robustness: spots may be blurred/sidelobed, so detection uses a
    GENEROUS ``spot_sigma`` and the LoG centroid (intensity-weighted) — the
    spot's mean position is preserved under defocus. Pass ``detected_yx`` to
    skip detection (e.g. for tests or a manual-anchor fit).

    Detected positions are in CROPPED pixels; we lift them to absolute via
    the ROI offset before fitting (A lives in absolute sensor coordinates).

    Returns the candidate dict (NOT committed — caller commits if accepted).
    """
    knm_yx = np.asarray(pattern_record['knm'], dtype=np.float64)
    n_sites = knm_yx.shape[0]
    if detected_yx is None:
        n_det = int(round(n_det_factor * n_sites))
        if min_distance is None:
            min_distance = max(3, int(round(0.6 * _expected_cam_pitch(
                pattern_record, roi))))
        detected_yx = detect_grid(np.asarray(avg_image, dtype=np.float64),
                                  num_tweezers=n_det, spot_sigma=spot_sigma,
                                  min_distance=min_distance, sort=False,
                                  refine_subpixel=True)
    detected_yx = np.asarray(detected_yx, dtype=np.float64).reshape(-1, 2)
    # cropped -> absolute
    xoff, yoff = float(roi[0]), float(roi[1])
    cam_abs = detected_yx + np.array([yoff, xoff], dtype=np.float64)
    return propose_update(knm_yx, cam_abs, scan_id, bootstrap=True)


def _expected_cam_pitch(pattern_record, roi):
    """Rough camera pitch (px) from the knm lattice pitch scaled by the
    frame: knm spans ~1024; the crop width sets the camera scale. Used only
    to pick a detection ``min_distance`` — a loose estimate is fine."""
    lat = pattern_record.get('lattice') or {}
    knm_pitch = lat.get('pitch_x') or lat.get('pitch_y') or 24.5
    w = float(roi[2]) if roi is not None and len(roi) >= 3 else 2100.0
    return max(4.0, knm_pitch * (w / 1024.0))
