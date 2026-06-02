"""Unit tests for the global SLM->camera affine module (no GPU / no SLM).

Covers: fit/apply round-trip, axis-order swap, crop separation, the
guardrail gates (accept / low-coverage / too-few / high-rms / degenerate /
rotation-drift / scale-drift), bootstrap correspondence under a ~90°
rotation, and EMA commit + rollback.
"""

import os

import numpy as np
import pytest

import yb_analysis.analysis.affine_transform as aff


@pytest.fixture(autouse=True)
def _affine_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv('YB_AFFINE_PATH', str(tmp_path / 'affine.json'))


# A ~90°-rotation + scale ~2.5 + translation similarity (knm[x,y] -> cam[Y,X]).
A_TRUE = np.array([[0.1, 2.5, 1000.0],
                   [-2.5, 0.1, 800.0]])


def _grid_xy(n=10, pitch=25.0):
    g = np.array([[i * pitch, j * pitch] for j in range(n) for i in range(n)],
                 dtype=np.float64)
    return g  # [x, y]


def _asym_grid_xy(n=10, pitch=25.0):
    """A square grid with a 2-wide x 4-tall corner notch removed. The notch
    is NOT symmetric under any rotation/reflection, so the orientation is
    unique (a square or diagonally-symmetric notch would leave the fit
    ambiguous — e.g. a reflected affine would fit equally well)."""
    return np.array([[i * pitch, j * pitch]
                     for j in range(n) for i in range(n)
                     if not (i >= n - 2 and j >= n - 4)], dtype=np.float64)


def test_fit_apply_roundtrip():
    knm_xy = _grid_xy()
    cam = aff.apply_affine(knm_xy, A_TRUE)            # [Y,X]
    A, rms, resid = aff.fit_affine(knm_xy, cam)
    assert rms < 1e-6
    np.testing.assert_allclose(A, A_TRUE, atol=1e-6)


def test_knm_to_xy_swaps_once():
    knm_yx = np.array([[5.0, 9.0], [1.0, 2.0]])       # [y,x]
    xy = aff._knm_to_xy(knm_yx)
    np.testing.assert_array_equal(xy, [[9.0, 5.0], [2.0, 1.0]])


def test_crop_separation_offset_only():
    knm_xy = _grid_xy(4)
    abs_yx = aff.apply_affine(knm_xy, A_TRUE)
    r1 = aff.apply_affine_cropped(knm_xy, A_TRUE, [1000, 100, 2100, 2100])
    r2 = aff.apply_affine_cropped(knm_xy, A_TRUE, [1200, 150, 2100, 2100])
    # cropped = absolute - [Yoff, Xoff]
    np.testing.assert_allclose(r1, abs_yx - [100, 1000])
    # changing ROI shifts the grid by exactly the offset delta; A untouched
    assert np.allclose(r1 - r2, [150 - 100, 1200 - 1000])


def test_bootstrap_recovers_affine_under_90deg():
    knm_xy = _asym_grid_xy(10)                          # asymmetric -> unique
    knm_yx = knm_xy[:, [1, 0]]                         # registry order [y,x]
    cam_abs = aff.apply_affine(knm_xy, A_TRUE)         # absolute [Y,X]
    # detected positions are CROPPED; lift happens inside bootstrap via roi.
    roi = [1000, 100, 2100, 2100]
    detected = cam_abs - [100, 1000]
    rec = {'knm': knm_yx.tolist(),
           'lattice': {'pitch_x': 25.0, 'pitch_y': 25.0}}
    cand = aff.bootstrap_from_scan(None, rec, roi, detected_yx=detected,
                                   scan_id='boot')
    assert cand['accept'] is True, cand['reason']
    assert cand['coverage'] == 1.0
    np.testing.assert_allclose(np.array(cand['A']), A_TRUE, atol=1e-4)


# ---- guardrail gates (exercise _make_candidate directly) ----------------

def test_gate_accept_bootstrap():
    c = aff._make_candidate(A_TRUE, rms=0.2, n_pairs=100, n_sites=100,
                            scan_id='s', bootstrap=True)
    assert c['accept'] and c['reason'] == 'accepted'


def test_gate_too_few_pairs():
    c = aff._make_candidate(A_TRUE, rms=0.2, n_pairs=20, n_sites=100,
                            scan_id='s', bootstrap=True)
    assert not c['accept'] and c['reason'] == 'reject_too_few_pairs'


def test_gate_low_coverage():
    c = aff._make_candidate(A_TRUE, rms=0.2, n_pairs=60, n_sites=100,
                            scan_id='s', bootstrap=True)
    assert not c['accept'] and c['reason'] == 'reject_low_coverage'


def test_gate_high_rms():
    c = aff._make_candidate(A_TRUE, rms=5.0, n_pairs=100, n_sites=100,
                            scan_id='s', bootstrap=True)
    assert not c['accept'] and c['reason'] == 'reject_high_rms'


def test_gate_degenerate():
    A_deg = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # det 0
    c = aff._make_candidate(A_deg, rms=0.1, n_pairs=100, n_sites=100,
                            scan_id='s', bootstrap=True)
    assert not c['accept'] and c['reason'] == 'reject_degenerate'


def test_gate_rotation_and_scale_drift():
    # Commit a current affine first.
    base = aff._make_candidate(A_TRUE, rms=0.1, n_pairs=100, n_sites=100,
                               scan_id='s0', bootstrap=True)
    aff.commit_update(base)
    # Rotate A_TRUE's linear part by 10° -> rotation drift.
    th = np.radians(10)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A_rot = A_TRUE.copy()
    A_rot[:, :2] = R @ A_TRUE[:, :2]
    c = aff._make_candidate(A_rot, rms=0.1, n_pairs=100, n_sites=100,
                            scan_id='s1', bootstrap=False)
    assert not c['accept'] and c['reason'] == 'reject_rotation_drift'
    # Scale up 20% -> scale drift.
    A_sc = A_TRUE.copy()
    A_sc[:, :2] = 1.2 * A_TRUE[:, :2]
    c2 = aff._make_candidate(A_sc, rms=0.1, n_pairs=100, n_sites=100,
                             scan_id='s2', bootstrap=False)
    assert not c2['accept'] and c2['reason'] == 'reject_scale_drift'


# ---- commit (EMA) + rollback --------------------------------------------

def test_commit_ema_and_rollback():
    base = aff._make_candidate(A_TRUE, rms=0.1, n_pairs=100, n_sites=100,
                               scan_id='s0', bootstrap=True)
    aff.commit_update(base)
    cur0 = aff.load_matrix()
    np.testing.assert_allclose(cur0, A_TRUE)           # first commit = candidate

    # A small in-gate drift: translate by a few px.
    A2 = A_TRUE.copy()
    A2[:, 2] += [4.0, 4.0]
    cand2 = aff._make_candidate(A2, rms=0.1, n_pairs=100, n_sites=100,
                                scan_id='s1', bootstrap=False)
    assert cand2['accept'], cand2['reason']
    aff.commit_update(cand2, ema_weight=0.25)
    cur1 = aff.load_matrix()
    # EMA: new = 0.75*A_TRUE + 0.25*A2  -> translation moved by +1 px.
    np.testing.assert_allclose(cur1[:, 2], A_TRUE[:, 2] + [1.0, 1.0], atol=1e-9)

    # rollback restores the pre-blend (bootstrap) matrix.
    assert aff.rollback() is True
    np.testing.assert_allclose(aff.load_matrix(), A_TRUE, atol=1e-9)
    assert aff.load_affine().get('rolled_back') is True


def _mask(b=11, s=2.0):
    ax = np.arange(b) - b // 2
    xx, yy = np.meshgrid(ax, ax)
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2))


def _spots_image(centers_yx, H=700, W=700, sigma=2.0, amp=50.0):
    img = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    for y, x in centers_yx:
        if 0 <= y < H and 0 <= x < W:
            img += amp * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
    return img


def _commit_known_affine():
    A = np.array([[0.0, 2.0, 300.0], [2.0, 0.0, 300.0]])  # knm[x,y]->cam[Y,X]
    aff.commit_update(aff._make_candidate(A, 0.1, 100, 100, 's0', bootstrap=True))
    return A


def test_propose_scan_update_detects_shift():
    A = _commit_known_affine()
    roi = [0, 0, 700, 700]                       # no crop offset for simplicity
    knm_xy = np.array([[i * 10.0, j * 10.0] for j in range(6) for i in range(6)])
    knm_yx = knm_xy[:, [1, 0]]
    pred = aff.apply_affine_cropped(knm_xy, A, roi)      # cropped [Y,X]
    shift = np.array([3.0, -4.0])
    img = _spots_image(pred + shift)
    cand = aff.propose_scan_update(img, knm_yx, roi, _mask(), 's1', search_range=12)
    assert cand['accept'], cand['reason']
    assert cand['shift_dy'] == 3 and cand['shift_dx'] == -4
    # commit moves the affine translation by ema_weight*shift
    before = aff.load_matrix()[:, 2].copy()
    aff.commit_update(cand, ema_weight=0.5)
    after = aff.load_matrix()[:, 2]
    np.testing.assert_allclose(after - before, 0.5 * shift, atol=1e-6)


def test_propose_scan_update_rejects_blank():
    A = _commit_known_affine()
    roi = [0, 0, 700, 700]
    knm_xy = np.array([[i * 10.0, j * 10.0] for j in range(6) for i in range(6)])
    img = np.random.default_rng(0).normal(0, 1, (700, 700))  # no lattice
    cand = aff.propose_scan_update(img, knm_xy[:, [1, 0]], roi, _mask(), 's1')
    assert not cand['accept'] and cand['reason'] == 'reject_low_snr'


def test_propose_scan_update_rejects_railed():
    A = _commit_known_affine()
    roi = [0, 0, 700, 700]
    # Coarse pitch (camera ~60 px) so a 30 px drift can't alias into the
    # search window (real arrays have pitch >> 2*search_range).
    knm_xy = np.array([[i * 30.0, j * 30.0] for j in range(5) for i in range(5)])
    knm_yx = knm_xy[:, [1, 0]]
    pred = aff.apply_affine_cropped(knm_xy, A, roi)
    img = _spots_image(pred + np.array([30.0, 0.0]))     # shift well beyond range
    cand = aff.propose_scan_update(img, knm_yx, roi, _mask(), 's1', search_range=12)
    assert not cand['accept'] and cand['reason'] == 'reject_shift_railed'


@pytest.mark.skipif(not os.environ.get('YB_IT_SCAN_ID'),
                    reason='set YB_IT_SCAN_ID (+YB_SLM_URL) for the live '
                           'bootstrap integration test')
def test_bootstrap_integration_real_scan():
    """Data+SLM-gated: bootstrap from a real scan via /eval (the same path
    yb_analysis.scripts.bootstrap_affine drives). Opt-in only."""
    from yb_analysis.scripts import bootstrap_affine as boot
    scan_id = os.environ['YB_IT_SCAN_ID']
    phase = os.environ.get('YB_IT_PHASE', 'phase/33x33_uniform.pt')
    res = boot._knm_via_eval(phase, [], 'col', os.environ.get('YB_SLM_URL'))
    rec = {'knm': res['knm'],
           'lattice': {'pitch_x': res.get('pitch_x'), 'pitch_y': res.get('pitch_y')}}
    avg, roi = boot._avg_image_and_roi(scan_id)
    cand = aff.bootstrap_from_scan(avg, rec, roi, scan_id=scan_id)
    assert cand['accept'], cand['reason']
    assert cand['coverage'] >= 0.9
    assert abs(abs(cand['rotation_deg']) - 90) < 5   # grid_rotation=90 conv.
