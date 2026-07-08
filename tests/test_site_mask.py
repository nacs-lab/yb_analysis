"""Tests for the global 'analyze only these sites' mask.

Covers the resolver, the site_mask kwarg on the probability functions (survival /
loss / loading / FP / per-shot), and the live compute_scan_curve path. Guards the
core invariants: site_mask=None is an exact no-op, and masking equals the same
computation on the site-sliced logicals.
"""
import numpy as np
import pytest

from yb_analysis.analysis import probabilities as P
from yb_analysis.analysis.site_mask import (
    resolve_site_mask, mask_site_resolved, effective_spec)
from yb_analysis.detection.scan_analysis import compute_scan_curve


def _cubes(seed=0, nS=20, nP=5, nR=8):
    rng = np.random.default_rng(seed)
    l1 = rng.random((nS, nP, nR)) < 0.5
    l2 = l1 & (rng.random((nS, nP, nR)) < 0.9)
    return l1, l2


def _mask(nS=20, keep=(1, 3, 5, 7, 9, 11)):
    m = np.zeros(nS, bool)
    m[list(keep)] = True
    return m


# ---- resolver ----

def test_resolve_none_is_none():
    assert resolve_site_mask(None, 20) is None


def test_resolve_bool_and_index_equivalent():
    m1 = resolve_site_mask(_mask(), 20)
    m2 = resolve_site_mask([1, 3, 5, 7, 9, 11], 20)
    assert np.array_equal(m1, m2)
    assert m1.sum() == 6


def test_resolve_bad_length_raises():
    with pytest.raises(ValueError):
        resolve_site_mask(np.ones(21, bool), 20)


def test_resolve_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_site_mask("not_a_registered_name", 20)


def test_resolve_index_out_of_range_raises():
    with pytest.raises(ValueError):
        resolve_site_mask([0, 99], 20)


# ---- effective_spec precedence (explicit > pattern > None; False forces full) ----

def test_effective_spec_explicit_wins():
    assert effective_spec("stable", pattern="whatever") == "stable"


def test_effective_spec_false_forces_full():
    # False -> None even if a pattern is given (opt out per call).
    assert effective_spec(False, pattern="33x33_feedback9") is None


def test_effective_spec_none_no_pattern_is_none():
    assert effective_spec(None, pattern=None) is None


def test_effective_spec_pattern_default(monkeypatch):
    # None + a pattern that configures a mask -> that pattern's spec.
    import yb_analysis.analysis.pattern_registry as pr
    monkeypatch.setattr(pr, "load_pattern_site_mask", lambda name: "stable"
                        if name == "P" else None)
    assert effective_spec(None, pattern="P") == "stable"
    assert effective_spec(None, pattern="Q") is None


# ---- masking primitive ----

def test_mask_site_resolved_nans_excluded_rows():
    arr = np.ones((20, 5))
    out = mask_site_resolved(arr, _mask())
    assert np.isnan(out[0]).all()      # excluded
    assert not np.isnan(out[1]).any()  # kept
    assert mask_site_resolved(arr, None) is arr  # no-op


# ---- probability functions: None is a no-op ----

@pytest.mark.parametrize("fn", [P.prob11, P.prob10])
def test_prob_none_is_noop(fn):
    l1, l2 = _cubes()
    a = fn(l1, l2)
    b = fn(l1, l2, site_mask=None)
    assert np.allclose(a[0], b[0], equal_nan=True)
    assert np.allclose(a[1], b[1], equal_nan=True)


def test_loading_none_is_noop():
    l1, _ = _cubes()
    a = P.loading_rate(l1)
    b = P.loading_rate(l1, site_mask=None)
    assert np.allclose(a[0], b[0], equal_nan=True)


# ---- probability functions: mask == manual over kept rows ----

def test_prob11_mask_matches_manual():
    l1, l2 = _cubes()
    m = _mask()
    msr, _ = P.prob11_site_resolved(l1, l2)
    manual = np.nanmean(msr[m], axis=0)
    got, _ = P.prob11(l1, l2, site_mask=m)
    assert np.allclose(got, manual, equal_nan=True)


def test_loading_mask_matches_manual():
    l1, _ = _cubes()
    m = _mask()
    lsr, _ = P.loading_rate_site_resolved(l1)
    manual = np.nanmean(lsr[m], axis=0)
    got, _ = P.loading_rate(l1, site_mask=m)
    assert np.allclose(got, manual, equal_nan=True)


def test_per_shot_mask_matches_site_slice():
    l1, l2 = _cubes()
    m = _mask()
    sliced = P.per_shot_rate_stats(l1[m], l2[m])
    masked = P.per_shot_rate_stats(l1, l2, site_mask=m)
    assert np.allclose(sliced["survival_mean"], masked["survival_mean"], equal_nan=True)
    assert np.allclose(sliced["loading_mean"], masked["loading_mean"], equal_nan=True)


def test_prob11_sem_uses_kept_count():
    # SEM = sqrt(sum sem_sr^2)/n_avg; masking must normalize by kept count, not
    # the full width, else the SEM is deflated.
    l1, l2 = _cubes()
    m = _mask()
    _, sem = P.prob11(l1, l2, site_mask=m)
    msr, ssr = P.prob11_site_resolved(l1, l2, site_mask=m)
    expect = np.sqrt(np.nansum(ssr ** 2, axis=0)) / m.sum()
    assert np.allclose(sem, expect, equal_nan=True)


# ---- live scan curve ----

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


# ---- live get_plot_data masking helpers ----

def test_live_mask_vec_and_list():
    from yb_analysis.acquisition.data_manager import _mask_vec, _mask_list
    m = np.array([1, 0, 1, 0], bool)
    v = np.arange(4, dtype=float)
    mv = _mask_vec(v, m)
    assert np.array_equal(np.isnan(mv), ~m)
    assert np.array_equal(v, np.arange(4))          # original untouched
    assert _mask_vec(v, None) is v                   # no-op
    assert _mask_vec(np.arange(5.0), m).shape == (5,)  # length mismatch no-op
    # bool logicals -> float NaN at excluded, 1.0 kept
    lg = np.array([1, 1, 0, 0], bool)
    ml = _mask_vec(lg, m)
    assert ml[0] == 1.0 and np.isnan(ml[1])
    lst = [{"a": i} for i in range(4)]
    out = _mask_list(lst, m)
    assert out[0] == {"a": 0} and out[1] is None and len(out) == 4
    assert _mask_list(lst, None) is lst


def test_apply_live_site_mask_covers_all_readouts():
    from yb_analysis.acquisition.data_manager import _apply_live_site_mask
    nS = 6
    m = np.array([1, 0, 1, 0, 1, 0], bool)
    d = {
        "cur_intensities": np.arange(nS, dtype=float),
        "logicals": np.ones(nS, bool),
        "thresholds": np.ones(nS),
        "infidelities": np.zeros(nS),
        "loading_rates": np.full(nS, 0.5),
        "live_hist_data": [{"counts": [1]} for _ in range(nS)],
        "live_gauss_fits": [{} for _ in range(nS)],
        "loaded_gauss_fits": [{} for _ in range(nS)],
        "cur_intensities2": None, "logicals2": None, "thresholds_img2": None,
        "infidelities_img2": None, "loading_rates_img2": None,
        "loaded_hist_data_img2": None, "loaded_gauss_fits_img2": None,
        "cur_intensities_mid": None, "logicals_mid": None,
    }
    _apply_live_site_mask(d, m, None)
    for k in ("cur_intensities", "thresholds", "infidelities", "loading_rates"):
        arr = np.asarray(d[k], float)
        assert np.array_equal(np.isnan(arr), ~m), k
    assert d["live_hist_data"][1] is None and d["live_hist_data"][0] == {"counts": [1]}
    assert d["live_gauss_fits"][3] is None


# ---- server figure builders: excluded (NaN-logical) sites render "excluded" ----

_PNG_URI = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def _masked_fig_dict(n):
    grid = np.column_stack([np.arange(n) * 10.0, np.arange(n) * 10.0])
    lg = np.zeros(n, dtype=float)
    lg[0::3] = 1.0        # loaded
    lg[1::5] = np.nan     # site-mask-excluded
    ci = np.full(n, 200.0); ci[0::3] = 210.0; ci[1::5] = np.nan
    thr = np.full(n, 205.0); thr[1::5] = np.nan
    return {
        '_img_data_uri': _PNG_URI, '_img_shape': [100, 100],
        '_img_vlo': 0, '_img_vhi': 255,
        'logicals': lg, 'grid_locations': grid, 'box_size': 9,
        'thresholds': thr, 'cur_intensities': ci, 'num_sites': n,
    }


@pytest.mark.parametrize("n,gl", [(60, 40), (20, 200)])  # (WebGL branch, shapes branch)
def test_fig_array_excluded_sites_are_gray(n, gl, monkeypatch):
    from yb_analysis.plotting import dashboard as D
    monkeypatch.setattr(D, "_GL_SITES", gl)
    fa = D._fig_array(_masked_fig_dict(n))
    trace_colors = [tr.line.color for tr in fa.data
                    if getattr(tr, 'line', None)
                    and getattr(tr.line, 'color', None)]
    shape_colors = [s['line']['color'] for s in (fa.layout.shapes or [])]
    all_colors = trace_colors + shape_colors
    assert '#555555' in all_colors, "excluded sites must be gray, not green/red"
    assert '#00ff88' in all_colors  # some loaded sites still green


def test_fig_intens_masked_no_nan_range(monkeypatch):
    from yb_analysis.plotting import dashboard as D
    fi = D._fig_intens(_masked_fig_dict(30))
    yr = fi.layout.yaxis.range
    assert yr is not None and all(np.isfinite(v) for v in yr)


def _masked_hist_dict(n=30):
    lg = np.zeros(n, dtype=float); lg[0::3] = 1.0; lg[1::5] = np.nan
    hist = [{"bin_centers": np.linspace(195, 210, 20), "counts": np.ones(20)}
            for _ in range(n)]
    fits = [{"params": [200, 1, 0.5, 205, 1, 0.5]} for _ in range(n)]
    for i in range(n):          # emulate _mask_list -> None at excluded sites
        if np.isnan(lg[i]):
            hist[i] = None; fits[i] = None
    thr = np.full(n, 202.0); thr[1::5] = np.nan
    inf = np.full(n, 0.01); inf[1::5] = np.nan
    rates = np.full(n, 0.5); rates[1::5] = np.nan
    return {"logicals": lg, "live_hist_data": hist, "live_gauss_fits": fits,
            "loaded_gauss_fits": fits, "thresholds": thr, "infidelities": inf,
            "loading_rates": rates, "n_accum_shots": 100,
            "hist_rep_sites": [0, 1, 6, 11]}   # sites 1, 11 excluded


def test_avg_hist_and_signatures_survive_masked_none_entries():
    from yb_analysis.plotting import dashboard as D
    d = _masked_hist_dict()
    D._fig_avghist(d)            # must not raise on None entries
    D._sig_avghist(d)
    D._sig_site(d, 1)            # excluded-site signature
    figs = D._figs_reps(d)
    assert len(figs) == 4        # incl. placeholders for excluded rep sites


def test_fig_site_excluded_marks_excluded():
    from yb_analysis.plotting import dashboard as D
    d = _masked_hist_dict()
    _, info = D._fig_site(d, 1)  # site 1 is excluded (None entry, NaN scalars)
    txt = " ".join(str(x) for x in info).lower()
    assert "excluded" in txt
    _, info0 = D._fig_site(d, 0)  # normal loaded site still populated
    assert len(info0) >= 3


# ---- loading-rate + infidelity maps: no 'nan' text, finite current ----

def test_fig_loading_and_infid_maps_no_nan_labels():
    from yb_analysis.plotting import dashboard as D
    n = 30
    grid = np.column_stack([np.arange(n) * 10.0, np.arange(n) * 10.0])
    rates = np.full(n, 0.5); rates[1::5] = np.nan
    inf = np.full(n, 0.01); inf[1::5] = np.nan
    fl = D._fig_loading({"grid_locations": grid, "loading_rates": rates})
    fi = D._fig_infid({"grid_locations": grid, "infidelities": inf})
    for fig in (fl, fi):
        txt = list(getattr(fig.data[0], "text", []) or [])
        assert not any("nan" in str(t).lower() for t in txt)


def test_fig_loading_live_current_is_finite_over_kept_sites():
    from yb_analysis.plotting import dashboard as D
    lg = np.zeros(30, dtype=float); lg[0::2] = 1.0; lg[1::5] = np.nan
    fig = D._fig_loading_live({"loading_history": np.full(20, 0.5), "logicals": lg})
    cur = [a.text for a in fig.layout.annotations if "Current" in str(a.text)]
    assert cur and "nan" not in str(cur[0]).lower()


# ---- pattern-registry site_mask round-trip ----

def test_pattern_registry_site_mask_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("YB_PATTERNS_DIR", str(tmp_path))
    import importlib
    import yb_analysis.analysis.pattern_registry as pr
    importlib.reload(pr)
    # seed a minimal record, then set/clear the mask spec
    pr.write_pattern({"name": "P", "n_sites": 20})
    assert pr.load_pattern_site_mask("P") is None
    pr.set_pattern_site_mask("P", "stable")
    assert pr.load_pattern_site_mask("P") == "stable"
    # other fields survive
    assert pr.get_pattern("P").get("n_sites") == 20
    pr.set_pattern_site_mask("P", None)
    assert pr.load_pattern_site_mask("P") is None
    importlib.reload(pr)  # restore default patterns dir for other tests


def test_compute_scan_curve_mask_matches_slice():
    l1, l2 = _cubes(seed=1)
    m = _mask()
    logs, pidx, params = _scan_logicals(l1, l2)
    logs_sliced = [(s, a[m], b[m]) for (s, a, b) in logs]
    full = compute_scan_curve(logs, pidx, params, 2)
    masked = compute_scan_curve(logs, pidx, params, 2, site_mask=m)
    manual = compute_scan_curve(logs_sliced, pidx, params, 2)
    assert np.allclose(masked["y_mean"], manual["y_mean"], equal_nan=True)
    assert not np.allclose(masked["y_mean"], full["y_mean"], equal_nan=True)
    # None is a no-op
    none = compute_scan_curve(logs, pidx, params, 2, site_mask=None)
    assert np.allclose(full["y_mean"], none["y_mean"], equal_nan=True)
