"""DataManager loading-pattern grid integration (no scan / no SLM).

Exercises _image_pattern_specs (imagePatternsJson + warmup_kwargs parsing,
incl. MATLAB uint16 strings) and _build_pattern_grids (affine-mapped,
ROI-cropped per-image grids) in isolation via DataManager.__new__.
"""

import json

import numpy as np
import pytest

from yb_analysis.acquisition.data_manager import DataManager
import yb_analysis.analysis.affine_transform as aff
import yb_analysis.analysis.pattern_registry as reg


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    monkeypatch.setenv('YB_AFFINE_PATH', str(tmp_path / 'affine.json'))
    monkeypatch.setenv('YB_PATTERNS_DIR', str(tmp_path / 'patterns'))
    return tmp_path


def _bare_dm(config, pSeq=1):
    dm = DataManager.__new__(DataManager)
    dm.scan_id = 1
    dm.config = config
    dm.num_images_per_seq = pSeq
    dm.loaded_thresholds = np.full(100, 5.0)
    dm.loaded_infidelities = np.full(100, np.nan)
    dm.loaded_gauss_fits = None
    dm.mask_mat = np.ones((11, 11))
    dm.grid_locations = np.zeros((100, 2))
    dm.num_sites = 100
    dm.is_two_array = False
    dm.grid_locations_img2 = None
    dm.num_sites_img2 = 0
    dm.loaded_thresholds_img2 = None
    dm.loaded_infidelities_img2 = None
    dm._pattern_grids = None
    dm._pattern_names = {}
    dm._roi = None
    dm._affine_grid0 = None
    return dm


def _knm_5x5():
    return [[float(j), float(i)] for j in range(5) for i in range(5)]  # [y,x]


def _register(name='p', knm=None):
    reg.write_pattern({
        'name': name, 'base_phase_path': f'phase/base/{name}.pt',
        'legacy_zerniked': False, 'baked_zernike': None,
        'base_sha256': 'a' * 64, 'default_loading_zernike': None,
        'order': 'col', 'fft_shape': [4096, 4096], 'threshold': 0.30,
        'min_dist': None, 'n_sites': len(knm), 'knm': knm, 'phases': None,
        'lattice': {}, 'source_endpoint': 't',
        'created_iso': 'i', 'updated_iso': 'i'})


# ---- _image_pattern_specs ------------------------------------------------

def test_specs_json_recycles_last(tmp_state):
    cfg = {'imagePatternsJson': json.dumps(
        [{'name': 'p', 'base_phase_path': 'phase/base/p.pt', 'order': 'col'}])}
    specs = _bare_dm(cfg, pSeq=2)._image_pattern_specs()
    assert len(specs) == 2
    assert specs[0]['name'] == 'p'
    assert specs[0]['base_phase_path'] == 'phase/base/p.pt'
    assert specs[1]['name'] == 'p'           # reused for frame 1


def test_specs_json_uint16_string(tmp_state):
    # MATLAB v7.3 stores char arrays as uint16 — must decode.
    s = json.dumps([{'name': 'p', 'base_phase_path': 'phase/base/p.pt'}])
    cfg = {'imagePatternsJson': np.array([ord(c) for c in s], dtype=np.uint16)}
    specs = _bare_dm(cfg, pSeq=1)._image_pattern_specs()
    assert specs and specs[0]['name'] == 'p'


def test_specs_warmup_kwargs(tmp_state):
    cfg = {'warmup_kwargs': {
        'initial_phase': 'phase/3270_z4eq4.pt',
        'final_phase': 'phase/3270_z4eq4.pt',
        'extras': {'initial_phase_zernike': [0, 0, 0, 0, -4]}}}
    specs = _bare_dm(cfg, pSeq=2)._image_pattern_specs()
    assert specs[0]['name'] == '3270_z4eq4'
    assert specs[0]['legacy_zerniked'] is True
    assert specs[0]['baked_zernike'] == [0, 0, 0, 0, -4]
    assert specs[1]['name'] == '3270_z4eq4'   # final_phase -> final frame
    assert specs[1]['legacy_zerniked'] is False  # no final_phase_zernike


def test_specs_none(tmp_state):
    assert _bare_dm({}, pSeq=1)._image_pattern_specs() is None


def test_specs_json_carries_planes(tmp_state):
    # A 2-D entry has planes_z_rad None; a 3-D entry threads the depths.
    cfg = {'imagePatternsJson': json.dumps([
        {'name': 'flat', 'base_phase_path': 'phase/base/flat.pt'},
        {'name': 'stack', 'base_phase_path': 'phase/base/stack.pt',
         'planes_z_rad': [-3.07, 3.07]}])}
    specs = _bare_dm(cfg, pSeq=2)._image_pattern_specs()
    assert specs[0]['planes_z_rad'] is None
    assert specs[1]['planes_z_rad'] == [-3.07, 3.07]


def test_specs_warmup_planes_shared_and_per_phase(tmp_state):
    cfg = {'warmup_kwargs': {
        'initial_phase': 'phase/a.pt', 'final_phase': 'phase/b.pt',
        'extras': {'planes_z_rad': [-1.0, 1.0],
                   'final_phase_planes_z_rad': [-2.0, 2.0]}}}
    specs = _bare_dm(cfg, pSeq=2)._image_pattern_specs()
    assert specs[0]['planes_z_rad'] == [-1.0, 1.0]    # shared applies to img1
    assert specs[1]['planes_z_rad'] == [-2.0, 2.0]    # per-phase wins for img2


# ---- _build_pattern_grids ------------------------------------------------

def test_build_grids_applies_affine(tmp_state):
    A = np.array([[0.0, 2.0, 300.0], [2.0, 0.0, 300.0]])
    aff.commit_update(aff._make_candidate(A, 0.1, 100, 100, 's0', bootstrap=True))
    knm = _knm_5x5()
    _register('p', knm)
    roi = [1000, 100, 2100, 2100]
    cfg = {'roi': roi, 'imagePatternsJson': json.dumps(
        [{'name': 'p', 'base_phase_path': 'phase/base/p.pt', 'order': 'col'}])}
    dm = _bare_dm(cfg, pSeq=1)
    dm._build_pattern_grids()
    assert dm._pattern_grids is not None
    expected = aff.apply_affine_cropped(np.asarray(knm)[:, [1, 0]], A, roi)
    np.testing.assert_allclose(dm.grid_locations, expected)
    assert dm.num_sites == 25
    assert len(dm.loaded_thresholds) == 25     # resized to match new grid
    assert dm._pattern_names[0] == 'p'
    assert dm._affine_grid0 is not None


def test_build_grids_3d_record_grid_stays_2d(tmp_state):
    """A 3-D registry record (knm still (N,2), 3-D fields alongside) yields the
    SAME (N,2) projected detection grid as a 2-D record — detection unchanged."""
    A = np.array([[0.0, 2.0, 300.0], [2.0, 0.0, 300.0]])
    aff.commit_update(aff._make_candidate(A, 0.1, 100, 100, 's0', bootstrap=True))
    knm = _knm_5x5()
    reg.write_pattern({
        'name': 's', 'base_phase_path': 'phase/base/s.pt',
        'legacy_zerniked': False, 'baked_zernike': None,
        'base_sha256': 'a' * 64, 'default_loading_zernike': None,
        'order': 'col', 'fft_shape': [4096, 4096], 'threshold': 0.30,
        'min_dist': None, 'n_sites': len(knm), 'knm': knm, 'phases': None,
        'lattice': {}, 'source_endpoint': 't',
        # 3-D fields ride alongside; knm itself stays (N,2).
        'planes_z_rad': [-3.07, 3.07], 'is_3d': True,
        'z_rad': [(-3.07 if j < 13 else 3.07) for j in range(len(knm))],
        'positions_knm3d': [[p[0], p[1], 0.0] for p in knm],
        'n_per_plane': [13, 12], 'plane_of_site': [0] * 13 + [1] * 12,
        'created_iso': 'i', 'updated_iso': 'i'})
    roi = [1000, 100, 2100, 2100]
    cfg = {'roi': roi, 'imagePatternsJson': json.dumps(
        [{'name': 's', 'base_phase_path': 'phase/base/s.pt', 'order': 'col',
          'planes_z_rad': [-3.07, 3.07]}])}
    dm = _bare_dm(cfg, pSeq=1)
    dm._build_pattern_grids()   # cache hit (planes match) -> no network
    expected = aff.apply_affine_cropped(np.asarray(knm)[:, [1, 0]], A, roi)
    np.testing.assert_allclose(dm.grid_locations, expected)
    assert dm.num_sites == 25


def test_build_grids_two_array_final_frame(tmp_state):
    A = np.array([[0.0, 2.0, 300.0], [2.0, 0.0, 300.0]])
    aff.commit_update(aff._make_candidate(A, 0.1, 100, 100, 's0', bootstrap=True))
    _register('load', _knm_5x5())
    _register('targ', _knm_5x5())
    roi = [1000, 100, 2100, 2100]
    cfg = {'roi': roi, 'imagePatternsJson': json.dumps([
        {'name': 'load', 'base_phase_path': 'phase/base/load.pt'},
        {'name': 'targ', 'base_phase_path': 'phase/base/targ.pt'}])}
    dm = _bare_dm(cfg, pSeq=2)
    dm._build_pattern_grids()
    assert dm.is_two_array is True
    assert dm.grid_locations_img2 is not None and dm.num_sites_img2 == 25
    assert len(dm.loaded_thresholds_img2) == 25
    assert dm._pattern_names[1] == 'targ'


def test_build_grids_no_affine_keeps_day_grid(tmp_state):
    # no affine committed -> legacy day-folder grid kept, name still recorded
    _register('p', _knm_5x5())
    cfg = {'roi': [1000, 100, 2100, 2100], 'imagePatternsJson': json.dumps(
        [{'name': 'p', 'base_phase_path': 'phase/base/p.pt'}])}
    dm = _bare_dm(cfg, pSeq=1)
    day = dm.grid_locations.copy()
    dm._build_pattern_grids()
    assert dm._pattern_grids is None
    np.testing.assert_array_equal(dm.grid_locations, day)
    assert dm._pattern_names.get(0) == 'p'     # surfaced for the dashboard


def _save_thr(reg, name, n, base=10.0):
    gs = np.empty(n, dtype=[('params', 'O')])
    for s in range(n):
        gs[s]['params'] = np.array([])
    reg.save_pattern_thresholds(name, {
        'thresholds': np.arange(n, dtype=float) + base,
        'infidelities': np.full(n, 0.01),
        'gaussFitsStruct': gs})


def test_pattern_thresholds_roundtrip(tmp_state):
    _save_thr(reg, 'p', 25)
    td = reg.load_pattern_thresholds('p')
    assert td is not None and len(td['thresholds']) == 25
    np.testing.assert_allclose(td['thresholds'], np.arange(25) + 10.0)
    assert len(td['infidelities']) == 25


def test_build_grids_uses_pattern_thresholds(tmp_state):
    A = np.array([[0.0, 2.0, 300.0], [2.0, 0.0, 300.0]])
    aff.commit_update(aff._make_candidate(A, 0.1, 100, 100, 's0', bootstrap=True))
    _register('p', _knm_5x5())          # 25 sites
    _save_thr(reg, 'p', 25, base=7.0)   # per-pattern thresholds
    cfg = {'roi': [1000, 100, 2100, 2100], 'imagePatternsJson': json.dumps(
        [{'name': 'p', 'base_phase_path': 'phase/base/p.pt'}])}
    dm = _bare_dm(cfg, pSeq=1)
    dm._build_pattern_grids()
    assert len(dm.loaded_thresholds) == 25
    np.testing.assert_allclose(dm.loaded_thresholds, np.arange(25) + 7.0)


def test_build_grids_no_pattern_is_legacy(tmp_state):
    aff.commit_update(aff._make_candidate(
        np.array([[0., 2., 300.], [2., 0., 300.]]), 0.1, 100, 100, 's', bootstrap=True))
    dm = _bare_dm({'roi': [1000, 100, 2100, 2100]}, pSeq=1)  # no patterns
    day = dm.grid_locations.copy()
    dm._build_pattern_grids()
    assert dm._pattern_grids is None
    np.testing.assert_array_equal(dm.grid_locations, day)
