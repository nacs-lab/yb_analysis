"""Shot-to-shot error family on the LIVE scan panel.

The live curve used to draw only the per-site binomial SEM, propagated across
sites -- an error that shrinks with the site count and is blind to the
shot-to-shot scatter (loading fluctuations, drift, one bad frame). The panel now
defaults to the per-shot family, matching the Analysis tab's
``per_shot_rate_stats``; the Scan card's "Shot-to-shot" switch turns it OFF for
the old array-average family.

Covers: the arrays compute_scan_curve publishes (1-D/2-D, survival + loading,
site-masked), their agreement with the offline per-shot stats, and the figure
layer honouring ?scan_err=.
"""
import numpy as np
import pytest

from yb_analysis.tests.conftest import fig_json
from yb_analysis.analysis.probabilities import per_shot_rate_stats
from yb_analysis.detection.scan_analysis import compute_scan_curve
from yb_analysis.plotting import dashboard as D


def _cubes(seed=0, nS=24, nP=5, nR=9):
    rng = np.random.default_rng(seed)
    l1 = rng.random((nS, nP, nR)) < 0.5
    l2 = l1 & (rng.random((nS, nP, nR)) < 0.9)
    return l1, l2


def _scan_logicals(l1, l2):
    nS, nP, nR = l1.shape
    logs, pidx = [], []
    sid = 1
    for p in range(nP):
        for r in range(nR):
            logs.append((sid, l1[:, p, r], l2[:, p, r]))
            pidx.append(p + 1)
            sid += 1
    return logs, np.array(pidx), np.arange(1, nP + 1, dtype=float)


def _d(sc, **kw):
    """Minimal _read_data dict the scan panel builders need."""
    d = {'scan_curve': sc, 'scan_name': 'test', 'scan_filename': 'run.h5',
         'scan_param_path': 'X', 'plot_scale': 1}
    d.update(kw)
    return d


# ---- what compute_scan_curve publishes ------------------------------------

def test_1d_publishes_both_error_families():
    l1, l2 = _cubes()
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    for k in ('y_sem', 'y_sem_pershot', 'y_std_pershot', 'n_shots_pershot'):
        assert k in sc, k
        assert len(sc[k]) == l1.shape[1]
    # SEM is the STD divided by sqrt(N shots) -- not a second, unrelated number.
    n = np.asarray(sc['n_shots_pershot'], float)
    assert np.allclose(sc['y_sem_pershot'],
                       np.asarray(sc['y_std_pershot']) / np.sqrt(n))
    # ...and it is a DIFFERENT number from the per-site family it replaces.
    assert not np.allclose(sc['y_sem_pershot'], sc['y_sem'])


def test_1d_survival_matches_offline_per_shot_stats():
    """The live per-shot SEM equals the Analysis tab's, on the same shots."""
    l1, l2 = _cubes(seed=3)
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    ps = per_shot_rate_stats(l1, l2)
    assert np.allclose(sc['y_sem_pershot'], ps['survival_sem_pershot'],
                       equal_nan=True)
    assert np.allclose(sc['y_std_pershot'], ps['survival_std_pershot'],
                       equal_nan=True)


def test_1d_loading_only_matches_offline_per_shot_stats():
    l1, l2 = _cubes(seed=4)
    logs, pidx, params = _scan_logicals(l1, l2)
    logs = [(s, a, None) for (s, a, _b) in logs]          # 1-image scan
    sc = compute_scan_curve(logs, pidx, params, 1)
    assert sc['mode'] == 'loading'
    ps = per_shot_rate_stats(l1)
    assert np.allclose(sc['y_sem_pershot'], ps['loading_sem_pershot'],
                       equal_nan=True)


def test_site_mask_restricts_the_per_shot_family_too():
    """Masking == the same computation on site-sliced logicals (as for y_mean)."""
    l1, l2 = _cubes(seed=5)
    m = np.zeros(l1.shape[0], bool)
    m[1::2] = True
    logs, pidx, params = _scan_logicals(l1, l2)
    masked = compute_scan_curve(logs, pidx, params, 2, site_mask=m)
    manual = compute_scan_curve([(s, a[m], b[m]) for (s, a, b) in logs],
                                pidx, params, 2)
    assert np.allclose(masked['y_sem_pershot'], manual['y_sem_pershot'],
                       equal_nan=True)
    full = compute_scan_curve(logs, pidx, params, 2)
    assert not np.allclose(masked['y_sem_pershot'], full['y_sem_pershot'],
                           equal_nan=True)


def test_single_shot_point_is_zero_not_nan():
    """One usable shot -> the spread is undefined; report 0, not NaN (a NaN
    error array blanks the Plotly trace)."""
    l1, l2 = _cubes(nP=2, nR=1)
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    assert np.all(np.asarray(sc['n_shots_pershot']) == 1)
    assert np.allclose(sc['y_sem_pershot'], 0.0)


def test_2d_publishes_per_shot_grids():
    l1, l2 = _cubes(nP=6, nR=5)
    logs, pidx, _ = _scan_logicals(l1, l2)
    dims = [{'name': 'a', 'values': np.arange(3.0), 'size': 3},
            {'name': 'b', 'values': np.arange(2.0), 'size': 2}]
    sc = compute_scan_curve(logs, pidx, None, 2, scan_dims=dims)
    assert sc['ndim'] == 2
    for k in ('sem', 'sem_pershot', 'std_pershot'):
        assert np.shape(sc[k]) == (2, 3), k


# ---- the figure layer honours the requested family ------------------------

def _err_y(fig):
    """The first trace's error_y, whichever representation the panel returns.

    _fig_scan_curve is still a go.Figure builder, and go.Figure's JSON form
    base64-packs numeric arrays ({'bdata', 'dtype'}) -- so read the plotly
    object when there is one and fall back to the dict for an Approach-A
    conversion later.
    """
    data = getattr(fig, 'data', None)
    if data:
        return data[0].error_y
    return fig_json(fig)['data'][0]['error_y']


def _err_array(fig):
    e = _err_y(fig)
    return np.asarray(e['array'] if isinstance(e, dict) else e.array, float)


@pytest.mark.parametrize('mode,key', [
    ('sem_pershot', 'y_sem_pershot'),     # switch ON  (default)
    ('sem_site',    'y_sem'),             # switch OFF (array average)
])
def test_fig_scan_curve_draws_the_requested_family(mode, key):
    l1, l2 = _cubes(seed=7)
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    fig = D._fig_scan_curve(_d(sc), scan_opts=(None, None, mode))
    assert np.allclose(_err_array(fig), np.asarray(sc[key], float))


def test_fig_scan_curve_defaults_to_per_shot():
    """No ?scan_err (and the legacy 2-tuple) -> the shot-to-shot family."""
    l1, l2 = _cubes(seed=8)
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    want = np.asarray(sc['y_sem_pershot'], float)
    for opts in (None, (None, None)):
        assert np.allclose(_err_array(D._fig_scan_curve(_d(sc), scan_opts=opts)),
                           want)


def test_the_two_families_are_the_only_ones_offered():
    """The switch has two sides; an unknown ?scan_err falls back to the default
    rather than drawing an arbitrary array."""
    assert D._SCAN_ERR_MODES == ('sem_pershot', 'sem_site')
    assert set(D._SCAN_ERR_LABEL) == set(D._SCAN_ERR_MODES)
    l1, l2 = _cubes(seed=9)
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    fig = D._fig_scan_curve(_d(sc), scan_opts=(None, None, 'bogus'))
    assert np.allclose(_err_array(fig), np.asarray(sc['y_sem_pershot'], float))


def test_fig_scan_curve_falls_back_when_per_shot_keys_are_absent():
    """An older snapshot has no *_pershot keys; the panel must still draw."""
    l1, l2 = _cubes(seed=10)
    sc = compute_scan_curve(*_scan_logicals(l1, l2), 2)
    legacy = {k: v for k, v in sc.items() if not k.endswith('_pershot')}
    fig = D._fig_scan_curve(_d(legacy), scan_opts=(None, None, 'sem_pershot'))
    assert np.allclose(_err_array(fig), np.asarray(sc['y_sem'], float))
