"""Tests for the img2 spot-shape GMM detector (detection/spot_shape_model.py).

Covers: dependency-free numpy inference matching the sklearn training model
bit-for-bit; the vectorised ``detect_frame`` producing intensities identical to
``detect_atom`` (so stored intensities / histograms are unchanged) plus the
model logicals + per-site posterior; and the edge-fallback contract.

The model artifact lives in ``spot_shape_ml/model/`` at the repo root. Tests
that need it skip cleanly when it isn't present (e.g. a standalone yb_analysis
checkout without the model folder).
"""
import os

import numpy as np
import pytest

from yb_analysis.detection import spot_shape_model as ssm
from yb_analysis.detection.detect_atom import detect_atom
from yb_analysis.detection.hist_init import make_mask


def _model_or_skip(variant='C'):
    ssm.clear_cache()
    m = ssm.load_model(variant)
    if m is None:
        pytest.skip('spot-shape model %s not available' % ssm.model_tag(variant))
    return m


def test_model_loads_and_shapes():
    m = _model_or_skip('C')
    assert m['tag'] == 'gmm_shape_model_C'
    assert m['box_size'] == 9
    d = m['n_pca']
    assert m['pca_components'].shape == (d, m['box_size'] ** 2)
    assert m['means'].shape[1] == d
    assert m['precisions_chol'].shape[1:] == (d, d)
    assert m['loaded_component'] in (0, 1)


def test_predict_patches_output_contract():
    m = _model_or_skip('C')
    rng = np.random.default_rng(0)
    box = m['box_size']
    patches = rng.normal(200, 5, (50, box, box))
    loaded, p = ssm.predict_patches(m, patches)
    assert loaded.shape == (50,) and loaded.dtype == bool
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))
    # bright (clearly-loaded) patches should read more loaded than flat ones
    bright = np.full((20, box, box), 200.0)
    bright[:, box // 2 - 1:box // 2 + 2, box // 2 - 1:box // 2 + 2] += 40
    flat = np.full((20, box, box), 200.0)
    lb, _ = ssm.predict_patches(m, bright)
    lf, _ = ssm.predict_patches(m, flat)
    assert lb.mean() >= lf.mean()


def test_numpy_inference_matches_sklearn():
    """The dependency-free numpy posterior must equal the sklearn training
    model to floating-point noise (so the live backend needs no sklearn)."""
    joblib = pytest.importorskip('joblib')
    pytest.importorskip('sklearn')
    m = _model_or_skip('C')
    jpath = os.path.join(ssm._model_dir(), 'gmm_shape_model_C.joblib')
    if not os.path.isfile(jpath):
        pytest.skip('joblib model not present')
    bundle = joblib.load(jpath)
    rng = np.random.default_rng(3)
    box = m['box_size']
    patches = rng.normal(201, 6, (4000, box, box))
    # sklearn reference
    Xs = bundle['scaler'].transform(patches.reshape(len(patches), -1))
    Z = bundle['pca'].transform(Xs) if bundle.get('pca') is not None else Xs
    ref_comp = bundle['gmm'].predict(Z)
    ref_p = bundle['gmm'].predict_proba(Z)[:, bundle['loaded_component']]
    ref_loaded = ref_comp == bundle['loaded_component']
    my_loaded, my_p = ssm.predict_patches(m, patches)
    assert np.array_equal(my_loaded, ref_loaded)
    assert np.max(np.abs(my_p - ref_p)) < 1e-9


def test_detect_frame_intensities_match_detect_atom():
    """detect_frame's intensities must equal detect_atom's masked sum for
    interior sites (so stored intensities / histograms / analysis are
    unchanged), while logicals come from the model."""
    m = _model_or_skip('C')
    box = m['box_size']
    rng = np.random.default_rng(1)
    H = W = 400
    frame = rng.normal(200, 3, (H, W))
    M = 120
    gy = rng.uniform(30, H - 30, M)
    gx = rng.uniform(30, W - 30, M)
    grid = np.column_stack([gy, gx])
    mask = make_mask(box, 2.0)
    _, inten_da = detect_atom(frame, grid, np.zeros(M), mask)
    res = ssm.detect_frame(m, frame, grid, mask)
    assert res is not None, 'fast path should engage for an interior grid'
    loaded, p, inten_df = res
    assert np.max(np.abs(inten_df - inten_da)) < 1e-9
    assert loaded.shape == (M,) and p.shape == (M,)
    assert np.all((p >= 0) & (p <= 1))


def test_detect_frame_edge_returns_none():
    """A site within the box half-width of the frame edge -> None (caller falls
    back to detect_atom + per-site patches)."""
    m = _model_or_skip('C')
    mask = make_mask(m['box_size'], 2.0)
    frame = np.full((60, 60), 200.0)
    grid = np.array([[1.0, 1.0], [30.0, 30.0]])   # first site hugs the corner
    assert ssm.detect_frame(m, frame, grid, mask) is None


def test_missing_variant_is_none():
    """A nonexistent variant loads to None (caller -> threshold fallback), and
    the negative result is cached."""
    ssm.clear_cache()
    assert ssm.load_model('ZZ_nonexistent') is None
    assert 'ZZ_nonexistent' in ssm._CACHE
    ssm.clear_cache()


def test_resolve_img2_model_only_for_distinct_patterns(monkeypatch):
    """The model is the img2 detector ONLY when img2 is a DISTINCT loading
    pattern from img1 (and the variant is enabled)."""
    _model_or_skip('C')
    import yb_analysis.acquisition.data_manager as dm
    from yb_analysis.acquisition.data_manager import DataManager
    monkeypatch.setattr(dm, 'IMG2_SHAPE_MODEL_VARIANT', 'C')

    def mk(p0, p2, n2=1068):
        d = DataManager.__new__(DataManager)
        d.is_two_array = True
        d.num_images_per_seq = 2
        d.num_sites_img2 = n2
        d._pattern_names = {0: p0, 1: p2}
        d._img2_model = None
        d._img2_logicals_source = None
        return d

    d = mk('47x47_uniform', '33x33_uniform')   # distinct -> model active
    d._resolve_img2_model()
    assert d._img2_model is not None
    assert d._img2_logicals_source == 'gmm_shape_model_C'

    d = mk('33x33_uniform', '33x33_uniform')   # same -> threshold detection
    d._resolve_img2_model()
    assert d._img2_model is None and d._img2_logicals_source is None

    monkeypatch.setattr(dm, 'IMG2_SHAPE_MODEL_VARIANT', '')   # disabled
    d = mk('47x47_uniform', '33x33_uniform')
    d._resolve_img2_model()
    assert d._img2_model is None and d._img2_logicals_source is None


def test_hdf5_certainties_roundtrip(tmp_path):
    """create_scan_file + append_block persist the certainties_img2 dataset and
    the logicals_img2_source provenance attribute."""
    h5py = pytest.importorskip('h5py')
    from yb_analysis.io.hdf5_store import create_scan_file, append_block
    p = str(tmp_path / 'd.h5')
    H = W = 8
    n1, n2, nseq = 3, 4, 5
    create_scan_file(p, {}, (H, W), n1, two_array=True, num_sites_img2=n2,
                     img2_logicals_source='gmm_shape_model_C')
    imgs = np.zeros((nseq * 2, H, W), dtype=np.int16)
    l1 = np.ones((nseq, n1), bool)
    i1 = np.ones((nseq, n1))
    l2 = np.ones((nseq, n2), bool)
    i2 = np.ones((nseq, n2))
    pr = np.random.default_rng(0).random((nseq, n2))
    append_block(p, imgs, l1, i1, np.arange(nseq), logicals_img2_block=l2,
                 intensities_img2_block=i2, proba_img2_block=pr)
    with h5py.File(p, 'r') as f:
        assert f.attrs['logicals_img2_source'] == 'gmm_shape_model_C'
        assert 'certainties_img2' in f
        assert f['certainties_img2'].shape == (nseq, n2)
        assert np.allclose(f['certainties_img2'][:], pr.astype('float32'), atol=1e-6)
        assert f['certainties_img2'].attrs['source'] == 'gmm_shape_model_C'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
