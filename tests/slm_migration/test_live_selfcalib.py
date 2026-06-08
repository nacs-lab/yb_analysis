"""Live per-N-shots self-calibration: EWMA affine (translation) + cheap per-pattern
threshold tracker, both shot-stamped to the audit log.

These exercise the DataManager helpers in isolation (a bare instance with only the
state each path touches) so they need no hardware / engine / camera.
"""

import json
import os

import numpy as np
import pytest


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Isolate PATH_PREFIX so affine/threshold/log writes land in tmp."""
    monkeypatch.setenv('YB_PATH_PREFIX', str(tmp_path))
    import yb_analysis.config as cfg
    monkeypatch.setattr(cfg, 'PATH_PREFIX', str(tmp_path))
    return tmp_path


def _bare_dm(num_sites, day_dir, scan_id='20260101000000', seq_total=0):
    from yb_analysis.acquisition.data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    dm.num_sites = num_sites
    dm.scan_id = scan_id
    dm._seq_total = seq_total
    dm._pattern_names = {0: 'testpat'}
    dm._pattern_grids = {0: np.zeros((num_sites, 2))}
    dm.live_thresholds = None
    dm.live_infidelities = None
    dm.live_gauss_fits = None
    dm.loaded_thresholds = np.full(num_sites, 125.0)
    dm.loaded_infidelities = np.full(num_sites, np.nan)
    dm.loaded_gauss_fits = None
    dm._thr_place_ratio = None
    dm._day_dir = str(day_dir)
    dm.is_two_array = False
    return dm


def _bimodal(rng, n_sites, k, empty_mu, atom_mu, load=0.6):
    out = []
    for _ in range(k):
        loaded = rng.random(n_sites) < load
        out.append(np.where(loaded, rng.normal(atom_mu, 15, n_sites),
                            rng.normal(empty_mu, 8, n_sites)))
    return out


def test_placement_ratio_midpoint_for_symmetric(state):
    dm = _bare_dm(8, state)
    fits = [{'params': np.array([50, 8, .4, 200, 15, .6])} for _ in range(8)]
    r = dm._placement_ratio(fits, np.full(8, 125.0))
    assert np.allclose(r, 0.5, atol=1e-6)
    # degenerate / missing fits -> 0.5 fallback
    r2 = dm._placement_ratio([{'params': None}] * 8, np.full(8, 999.0))
    assert np.allclose(r2, 0.5)


def test_cheap_threshold_is_stable_and_tracks_drift(state):
    rng = np.random.default_rng(0)
    dm = _bare_dm(16, state)
    dm._thr_place_ratio = np.full(16, 0.5)
    dm.live_thresholds = np.full(16, 125.0)
    # stable populations -> threshold stays near the 125 midpoint
    dm._intensity_accum = _bimodal(rng, 16, 120, 50, 200)
    for _ in range(20):
        dm._update_thresholds_live_cheap()
    assert 115 < dm.thresholds.mean() < 135
    # both peaks drift up by +60 -> threshold follows up
    dm._intensity_accum = _bimodal(rng, 16, 120, 110, 260)
    for _ in range(30):
        dm._update_thresholds_live_cheap()
    assert dm.thresholds.mean() > 160


def test_cheap_threshold_saves_per_pattern_and_logs(state):
    rng = np.random.default_rng(1)
    dm = _bare_dm(10, state, scan_id='20260102123456', seq_total=77)
    dm._thr_place_ratio = np.full(10, 0.5)
    dm.live_thresholds = np.full(10, 125.0)
    dm._intensity_accum = _bimodal(rng, 10, 100, 50, 200)
    dm._update_thresholds_live_cheap()
    # per-pattern threshold.mat written
    import yb_analysis.analysis.pattern_registry as reg
    assert reg.pattern_threshold_path('testpat').is_file()
    # shot-stamped audit log
    logp = state / 'yb_dashboard_state' / 'update_logs' / 'thresholds' / 'testpat.jsonl'
    rec = json.loads(logp.read_text().strip().splitlines()[-1])
    assert rec['scan_id'] == '20260102123456' and rec['seq_no'] == 77
    assert rec['source'] == 'cheap' and 'ts' in rec
    assert len(rec['thresholds']) == 10


def test_cheap_save_preserves_existing_gauss_fits(state):
    """A cheap update BEFORE the first full fit (live_gauss_fits is None) must not
    wipe the per-pattern gaussFitsStruct the dashboard/analysis rely on."""
    from scipy.io import loadmat
    rng = np.random.default_rng(2)
    n = 12
    dm = _bare_dm(n, state)
    dm.live_thresholds = np.full(n, 125.0)
    dm.live_infidelities = np.full(n, 0.01)
    dm.live_gauss_fits = None  # before first full fit
    dm.loaded_gauss_fits = [{'params': np.array([50, 8, .4, 200, 15, .6])}
                            for _ in range(n)]
    dm._thr_place_ratio = np.full(n, 0.5)
    dm._intensity_accum = _bimodal(rng, n, 80, 50, 200)
    dm._update_thresholds_live_cheap()
    import yb_analysis.analysis.pattern_registry as reg
    gs = np.atleast_1d(loadmat(str(reg.pattern_threshold_path('testpat')),
                               squeeze_me=True)['gaussFitsStruct'])
    n_fit = sum(1 for s in range(len(gs)) if np.ravel(gs[s]['params']).size >= 6)
    assert n_fit == n


def test_affine_live_ewma_translation_and_log(state):
    """A synthetic +dy/+dx image shift commits ema*shift to the affine TRANSLATION
    only (rotation/scale frozen) and appends a shot-stamped record."""
    import collections
    import yb_analysis.analysis.affine_transform as aff
    import yb_analysis.config as cfg
    from yb_analysis.acquisition.data_manager import _gaussian_mask

    n_side = 6
    pitch = 30.0
    ys, xs = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing='ij')
    knm = np.column_stack([ys.ravel(), xs.ravel()]).astype(float)   # [y,x]
    # seed an affine: identity-ish scale, place grid in a 400x400 frame
    A = np.array([[0.0, pitch, 60.0], [pitch, 0.0, 60.0]])
    aff.commit_update(aff._make_candidate(A, 0.1, len(knm), len(knm), 's0',
                                          bootstrap=True))
    roi = [0.0, 0.0, 400.0, 400.0]
    grid = aff.apply_affine_cropped(aff._knm_to_xy(knm), aff.load_matrix(), roi)

    def render(shift_y, shift_x):
        img = np.zeros((400, 400), float)
        for (gy, gx) in grid:
            yy, xx = int(round(gy + shift_y)), int(round(gx + shift_x))
            if 2 <= yy < 398 and 2 <= xx < 398:
                img[yy - 1:yy + 2, xx - 1:xx + 2] += 100.0
        return img

    dm = type('D', (), {})()  # not needed; use real DM
    from yb_analysis.acquisition.data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    dm.scan_id = '20260103000000'
    dm._seq_total = 5
    dm._roi = roi
    dm._pattern_knm = {0: knm}
    dm._pattern_names = {0: 'testpat'}
    dm.num_images_per_seq = 1
    dm.is_two_array = False
    dm.num_sites = len(knm)
    dm.grid_locations = np.zeros((len(knm), 2))
    dm._affine_grid0 = None
    dm.grid_shift_history = collections.deque(maxlen=50)
    dm.mask_mat = _gaussian_mask(7, 2.0)
    dm._affine_update_running = False

    t0 = aff.load_matrix()[:, 2].copy()
    imgs = np.array([render(2, 3) for _ in range(8)])
    dm._affine_live_worker('testpat', knm, imgs, 5)
    A1 = aff.load_matrix()
    dt = A1[:, 2] - t0
    # EWMA weight applied to the detected (2,3) shift; rotation/scale unchanged
    assert np.allclose(dt, cfg.AFFINE_LIVE_EMA * np.array([2, 3]), atol=0.5)
    d0, d1 = aff.decompose(A), aff.decompose(A1)
    assert abs(d0['rotation_deg'] - d1['rotation_deg']) < 1e-6
    assert abs(d0['scale_x'] - d1['scale_x']) < 1e-6
    # live grid refreshed from the new affine
    assert dm.grid_locations.shape == (len(knm), 2) and np.any(dm.grid_locations)
    # shot-stamped affine log
    logp = state / 'yb_dashboard_state' / 'update_logs' / 'affine.jsonl'
    rec = json.loads(logp.read_text().strip().splitlines()[-1])
    assert rec['scan_id'] == '20260103000000' and rec['seq_no'] == 5
    assert rec['accepted'] is True and 'ts' in rec and 'tx' in rec
