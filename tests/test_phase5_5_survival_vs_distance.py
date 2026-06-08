"""Phase 5.5 Track B — lab-side survival_vs_distance.

Exercises run_analysis._survival_vs_distance directly with hand-rolled
fixtures (no scan dir / HDF5 needed).

    python -m pytest yb_analysis/tests/test_phase5_5_survival_vs_distance.py -v
"""

import numpy as np

from yb_analysis.analysis.run_analysis import _survival_vs_distance


def _scan(xs, ys, num_images=2):
    return {'initGridLocationsX': xs, 'initGridLocationsY': ys,
            'NumImages': num_images}


def _bundle(img1, img2, seq_ids):
    return {'logicals_img1': np.asarray(img1, dtype=np.uint8),
            'logicals_img2': np.asarray(img2, dtype=np.uint8),
            'seq_ids': np.asarray(seq_ids, dtype=np.int64)}


def test_survival_vs_distance_synthetic():
    # Sites at x = 0, 10, 30 (y = 0). Two shots, one pair each:
    #   shot1: (0,0) -> (0,10)  dist 10, target=site1, survived (img2=1)
    #   shot2: (0,0) -> (0,30)  dist 30, target=site2, lost     (img2=0)
    scan = _scan([0, 10, 30], [0, 0, 0])
    bundle = _bundle(img1=[[1, 0, 0], [1, 0, 0]],
                     img2=[[0, 1, 0], [0, 0, 0]],
                     seq_ids=[1, 2])
    paths_info = {'paths_frame': 'knm_native', 'paths_per_shot': [
        {'seq_id': 1, 'init_xy': [[0, 0]], 'target_xy': [[0, 10]]},
        {'seq_id': 2, 'init_xy': [[0, 0]], 'target_xy': [[0, 30]]},
    ]}
    out = _survival_vs_distance(paths_info, bundle, scan, n_bins=2)
    assert out is not None and 'skipped_reason' not in out
    assert out['n_total_pairs'] == 2
    assert out['n_unmatched'] == 0
    assert out['n_pairs_per_bin'] == [1, 1]
    assert out['survival_mean'] == [1.0, 0.0]
    assert out['distance_units'] == 'knm_pixels'
    # Bin centres for edges linspace(10, 30, 3) = [10, 20, 30] -> [15, 25].
    assert out['centers'] == [15.0, 25.0]


def test_survival_vs_distance_no_paths():
    scan = _scan([0, 10], [0, 0])
    bundle = _bundle([[1, 0]], [[1, 0]], [1])
    assert _survival_vs_distance(
        {'paths_per_shot': None, 'paths_frame': None}, bundle, scan) is None
    assert _survival_vs_distance(None, bundle, scan) is None


def test_survival_vs_distance_lattice_mismatch():
    # Target coord nowhere near any lab site -> not computable.
    scan = _scan([0, 10, 30], [0, 0, 0])
    bundle = _bundle([[1, 0, 0]], [[0, 0, 0]], [1])
    paths_info = {'paths_frame': 'knm_native', 'paths_per_shot': [
        {'seq_id': 1, 'init_xy': [[0, 0]], 'target_xy': [[0, 1000]]},
    ]}
    out = _survival_vs_distance(paths_info, bundle, scan)
    assert out == {'skipped_reason': 'lattice_mismatch'}


def test_distance_units_propagates(monkeypatch):
    # With NO affine available, camera-pixel distances stay in camera px
    # (no knm conversion). Patch the scale lookup to None to isolate the
    # unit-propagation logic from whatever affine is committed on the box.
    import yb_analysis.analysis.run_analysis as RA
    monkeypatch.setattr(RA, '_affine_scale_for_scan', lambda sid: (None, None))
    scan = _scan([0, 10, 30], [0, 0, 0])
    bundle = _bundle([[1, 0, 0]], [[0, 1, 0]], [1])
    pps = [{'seq_id': 1, 'init_xy': [[0, 0]], 'target_xy': [[0, 10]]}]
    cam = _survival_vs_distance(
        {'paths_frame': 'camera_bitorder', 'paths_per_shot': pps},
        bundle, scan)
    knm = _survival_vs_distance(
        {'paths_frame': 'knm_native', 'paths_per_shot': pps}, bundle, scan)
    none = _survival_vs_distance(
        {'paths_frame': None, 'paths_per_shot': pps}, bundle, scan)
    assert cam['distance_units'] == 'camera_pixels'
    assert knm['distance_units'] == 'knm_pixels'
    assert none['distance_units'] == 'unknown'


def test_camera_pixels_converted_to_knm_when_affine_present(monkeypatch):
    # When an affine scale IS available, camera-pixel distances are divided
    # by it and reported as knm pixels.
    import yb_analysis.analysis.run_analysis as RA
    monkeypatch.setattr(RA, '_affine_scale_for_scan', lambda sid: (2.0, 'run'))
    scan = _scan([0, 10, 30], [0, 0, 0])
    bundle = _bundle([[1, 0, 0]], [[0, 1, 0]], [1])
    pps = [{'seq_id': 1, 'init_xy': [[0, 0]], 'target_xy': [[0, 10]]}]
    out = _survival_vs_distance(
        {'paths_frame': 'camera_bitorder', 'paths_per_shot': pps},
        bundle, scan, scan_id='20260101000000')
    assert out['distance_units'] == 'knm_pixels'
    assert out['cam_px_per_knm_px'] == 2.0
    # the single pair's 10-camera-px transit -> 5 knm px (min bin edge)
    assert out['bins'][0] == 5.0
