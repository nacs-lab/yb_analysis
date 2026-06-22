"""Spot-shape GMM classifier: loaded vs empty by patch SHAPE, not intensity.

Used for **img2** (post-protocol) detection when intensity thresholding fails
because nearly all sites are loaded -- the per-site intensity histogram goes
unimodal, so there is no clean threshold (the degenerate-fit guard then rejects
the refit). A spot-shape model still discriminates the two classes.

The model (trained offline in ``spot_shape_ml/``) is:
    9x9 patch -> StandardScaler -> PCA(n) -> GaussianMixture(2, 'full')
``loaded_component`` is the higher-intensity component; ``p_loaded`` is its
posterior. Variant ``C`` (PCA-5, curated training set) gives graded posteriors.

Inference here is **dependency-free numpy** straight from the saved ``.npz``
(no sklearn at runtime -> no pickle/version risk in the live backend). It is
validated bit-for-bit against the sklearn training model in
``tests/test_spot_shape_model.py``.

Model artifacts live in the ``spot_shape_ml/model/`` folder at the repo root
(moved out of ``tmp/``); override the base dir with ``$YB_SPOT_SHAPE_ML_DIR``.
"""

import os
import json
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE = {}   # variant -> model dict, or None when unavailable (negative cache)

_LOG2PI = float(np.log(2.0 * np.pi))


def _model_dir():
    """Base dir holding ``gmm_shape_model*.npz``. ``$YB_SPOT_SHAPE_ML_DIR`` may
    point at either the ``spot_shape_ml`` folder or its ``model`` subdir."""
    env = os.environ.get('YB_SPOT_SHAPE_ML_DIR')
    if env:
        cand = os.path.join(env, 'model')
        return cand if os.path.isdir(cand) else env
    # Repo root = two levels up from this file (<root>/yb_analysis/detection/).
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, 'spot_shape_ml', 'model')


def _npz_name(variant):
    v = (variant or 'A')
    return 'gmm_shape_model.npz' if v in ('A', '') else 'gmm_shape_model_%s.npz' % v


def model_tag(variant):
    """Provenance string stored in the data — always carries the variant letter
    (e.g. ``gmm_shape_model_A`` / ``_C``) so the data is unambiguous about which
    model produced the logicals, even though variant A's *file* is unsuffixed."""
    v = (variant or 'A')
    return 'gmm_shape_model_%s' % v


def load_model(variant='C'):
    """Load + cache the GMM shape model ``variant`` from its ``.npz``.

    Returns a dict of precomputed inference tensors (+ ``meta`` provenance and
    ``tag``), or ``None`` when the artifact is missing/unreadable (so callers
    cleanly fall back to threshold detection). Negative results are cached too,
    so a missing model costs one stat per process, not per shot."""
    key = variant or 'A'
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        m = None
        path = os.path.join(_model_dir(), _npz_name(variant))
        try:
            z = np.load(path, allow_pickle=True)
            pchol = np.ascontiguousarray(z['precisions_cholesky'], dtype=np.float64)
            if pchol.ndim != 3:
                raise ValueError('expected full-covariance precisions_cholesky '
                                 '(K,d,d); got shape %s' % (pchol.shape,))
            # PCA stage is OPTIONAL: PCA-reduced variants (B/C) carry
            # pca_components/pca_mean; the full-D variant (A) has none and the
            # GMM runs directly on the standardized patch vector.
            has_pca = ('pca_components' in z) and ('pca_mean' in z)
            m = {
                'variant': key,
                'tag': model_tag(variant),
                'scaler_mean': np.asarray(z['scaler_mean'], dtype=np.float64),
                'scaler_scale': np.asarray(z['scaler_scale'], dtype=np.float64),
                'pca_components': (np.asarray(z['pca_components'], dtype=np.float64)
                                   if has_pca else None),
                'pca_mean': (np.asarray(z['pca_mean'], dtype=np.float64)
                             if has_pca else None),
                'means': np.asarray(z['means'], dtype=np.float64),
                'precisions_chol': pchol,
                'log_weights': np.log(np.asarray(z['weights'], dtype=np.float64)),
                # precompute the per-component (M @ prec_chol) and log|prec_chol|
                'log_det': np.sum(
                    np.log(np.diagonal(pchol, axis1=1, axis2=2)), axis=1),
                'loaded_component': int(z['loaded_component']),
                'box_size': int(z['box_size']),
                'n_pca': int(z['n_pca']) if 'n_pca' in z else int(pchol.shape[1]),
                'path': path,
            }
            m['mean_dot_prec'] = np.stack(
                [m['means'][k] @ pchol[k] for k in range(m['means'].shape[0])])
            jpath = os.path.splitext(path)[0] + '.json'
            try:
                with open(jpath, 'r', encoding='utf-8') as fh:
                    m['meta'] = json.load(fh)
            except Exception:  # noqa: BLE001
                m['meta'] = {}
            logger.info('Loaded spot-shape model %s (box=%d, %s) from %s',
                        m['tag'], m['box_size'],
                        ('PCA=%d' % m['n_pca']) if m['pca_components'] is not None
                        else ('full-%dD' % m['n_pca']), path)
        except FileNotFoundError:
            logger.info('Spot-shape model %s not found at %s; img2 will use '
                        'threshold detection', model_tag(variant), path)
            m = None
        except Exception as e:  # noqa: BLE001
            logger.warning('Spot-shape model %s load failed (%s); img2 will use '
                           'threshold detection', model_tag(variant), e)
            m = None
        _CACHE[key] = m
        return m


def clear_cache():
    """Drop the cached models (tests / after replacing artifacts on disk)."""
    with _LOCK:
        _CACHE.clear()


def _features(model, patches):
    """(N, box, box) or (N, box*box) -> GMM feature matrix.

    Standardize per-pixel, then PCA-project for reduced variants (B/C). The
    full-D variant (A) has no PCA -> the standardized patch IS the feature."""
    P = np.asarray(patches, dtype=np.float64)
    P = P.reshape(P.shape[0], -1)
    Xz = (P - model['scaler_mean']) / model['scaler_scale']
    if model['pca_components'] is None:
        return Xz
    return (Xz - model['pca_mean']) @ model['pca_components'].T


def _posterior(model, Z):
    """sklearn-identical GaussianMixture('full') responsibilities.

    Returns (argmax_component (N,), p_loaded (N,))."""
    means = model['means']
    pchol = model['precisions_chol']
    mdp = model['mean_dot_prec']
    K, d = means.shape
    n = Z.shape[0]
    lp = np.empty((n, K), dtype=np.float64)
    for k in range(K):
        y = Z @ pchol[k] - mdp[k]                       # (N, d)
        lp[:, k] = np.einsum('ij,ij->i', y, y)
    wlp = -0.5 * (d * _LOG2PI + lp) + model['log_det'] + model['log_weights']
    mx = wlp.max(axis=1, keepdims=True)
    pr = np.exp(wlp - mx)
    pr /= pr.sum(axis=1, keepdims=True)
    return np.argmax(wlp, axis=1), pr[:, model['loaded_component']]


def predict_patches(model, patches):
    """(N, box, box) or (N, box*box) -> (loaded_bool (N,), p_loaded (N,))."""
    Z = _features(model, patches)
    comp, p = _posterior(model, Z)
    return (comp == model['loaded_component']), p


def detect_frame(model, frame, grid_yx, mask_mat):
    """Vectorised single-frame img2 detection.

    Returns ``(loaded (M,), p_loaded (M,), intensities (M,))`` where
    ``loaded``/``p_loaded`` come from the shape model and ``intensities`` is the
    SAME Gaussian-masked sum production thresholding integrates (so histograms /
    stored intensities / the threshold refit are unchanged). One vectorised
    patch gather serves both -> ~3x faster than the per-site ``detect_atom``
    loop.

    Returns ``None`` (so the caller falls back to ``detect_atom`` + per-site
    patches) when any site is within the box half-width of the frame edge -- the
    fast path is clamp-free/centred and would otherwise read out of bounds.
    Real tweezer sites sit well inside the crop, so this is the universal path.
    """
    frame = np.asarray(frame, dtype=np.float64)
    H, W = frame.shape
    mask = np.asarray(mask_mat, dtype=np.float64)
    box_m = int(model['box_size'])
    box_i = int(mask.shape[0])
    box = max(box_m, box_i)
    half = box // 2
    grid = np.asarray(grid_yx, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] == 0:
        return None
    y0 = np.round(grid[:, 0]).astype(np.intp)
    x0 = np.round(grid[:, 1]).astype(np.intp)
    if (y0.min() - half < 0 or y0.max() - half + box > H
            or x0.min() - half < 0 or x0.max() - half + box > W):
        return None
    M = y0.shape[0]
    dw = np.arange(box)
    yy = np.broadcast_to((y0 - half)[:, None, None] + dw[None, :, None],
                         (M, box, box))
    xx = np.broadcast_to((x0 - half)[:, None, None] + dw[None, None, :],
                         (M, box, box))
    patch = frame[yy, xx]                               # (M, box, box), centred
    off_i = (box - box_i) // 2
    pi = patch[:, off_i:off_i + box_i, off_i:off_i + box_i]
    intensities = np.tensordot(pi, mask, axes=([1, 2], [0, 1]))
    off_m = (box - box_m) // 2
    pm = patch[:, off_m:off_m + box_m, off_m:off_m + box_m]
    loaded, p = predict_patches(model, pm)
    return loaded, p, intensities
