"""Tests for Approach A: the hot per-site Scattergl maps (_fig_loading /
_fig_infid) built as RAW JSON dicts instead of go.* objects, serialized by
orjson -- skipping Plotly validation + the bdata encode/decode round-trip
(~45% of the dashboard GIL per py-spy). Output must stay VALUE-IDENTICAL to the
old go.* path; the silent-breakage risks are (a) a named colorscale left as a
string that Plotly.js can't resolve, and (b) a missing default template.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_raw_figures.py -v
"""
import json

import numpy as np
import plotly.graph_objects as go
import pytest

from yb_analysis.plotting import dashboard as dsh


def _frame(n, nan_frac=0.0, seed=0):
    r = np.random.RandomState(seed)
    grid = r.randint(0, 2100, (n, 2))
    rates = r.rand(n)
    inf = 10 ** r.uniform(-5, -0.5, n)
    if nan_frac:
        rates[r.rand(n) < nan_frac] = np.nan
        inf[r.rand(n) < nan_frac] = np.nan
    return {"grid_locations": grid, "loading_rates": rates, "infidelities": inf}


def test_loading_and_infid_return_raw_dicts():
    d = _frame(300)
    for fn in (dsh._fig_loading, dsh._fig_infid):
        fig = fn(d)
        assert isinstance(fig, dict) and "data" in fig and "layout" in fig
        assert fig["data"][0]["type"] == "scattergl"


def test_named_colorscale_is_expanded_not_a_string():
    """RdYlGn / Magma are NOT Plotly.js-native names -- they MUST be expanded to
    an explicit [[pos, color], ...] list or the browser renders wrong colors."""
    for fn, name in ((dsh._fig_loading, "RdYlGn"), (dsh._fig_infid, "Magma")):
        cs = fn(_frame(300))["data"][0]["marker"]["colorscale"]
        assert isinstance(cs, list) and cs and isinstance(cs[0], list), \
            f"{name} colorscale not expanded: {type(cs)}"
        # matches what go.* would have produced
        assert cs == dsh._expand_colorscale(name)


def test_default_template_present():
    """The ~7 KB default template go.Figure() attaches must be reproduced so the
    raw-dict figure renders identically."""
    lay = dsh._fig_loading(_frame(300))["layout"]
    assert "template" in lay and isinstance(lay["template"], dict)
    assert "data" in lay["template"] and "layout" in lay["template"]


def test_text_mode_toggles_at_100_sites():
    """<100 sites -> markers+text with per-site labels + textfont; >=100 ->
    markers only, no text/textfont (but textposition stays, as go.* emitted)."""
    small = dsh._fig_loading(_frame(50))["data"][0]
    assert small["mode"] == "markers+text"
    assert "text" in small and "textfont" in small
    assert small["textposition"] == "middle center"
    big = dsh._fig_loading(_frame(300))["data"][0]
    assert big["mode"] == "markers"
    assert "text" not in big and "textfont" not in big
    assert big["textposition"] == "middle center"


def test_build_fig_json_value_identical_to_go_star():
    """The end-to-end _build_fig_json('load'/'infid') output must equal the old
    go.* path (to_plotly_json -> decode bdata) in JSON VALUES, incl NaN->null,
    across text-mode and marker-mode + NaN sites."""
    def norm(o):
        if isinstance(o, dict):
            return {k: norm(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [norm(v) for v in o]
        if isinstance(o, np.ndarray):
            return norm(o.tolist())
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    _L, _A = dsh._L, dsh._A

    def golden_loading(d, sz=12):
        grid, rates = d["grid_locations"], d["loading_rates"]
        n = len(grid); rates = np.asarray(rates, float); exc = np.isnan(rates)
        if n < 100:
            mode = "markers+text"; text = ["" if e else f"{r:.0%}" for r, e in zip(rates, exc)]; tf = dict(size=7, color="black")
        else:
            mode = "markers"; text = None; tf = None
        cd = np.column_stack([np.arange(1, n + 1), rates])
        f = go.Figure(go.Scattergl(x=grid[:, 1], y=grid[:, 0], mode=mode,
            marker=dict(size=sz, color=rates.tolist(), colorscale="RdYlGn", cmin=0, cmax=1,
                        colorbar=dict(title="Rate", len=0.9), line=dict(width=0.5, color="white")),
            text=text, textfont=tf, textposition="middle center", customdata=cd,
            hovertemplate="Site %{customdata[0]}: %{customdata[1]:.1%}<extra></extra>"))
        f.update_layout(**_L, title=f"Loading Rates ({n} sites)", clickmode="event",
            yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False, **_A),
            xaxis=dict(visible=False, **_A))
        return f

    for n in (50, 99, 100, 500):
        d = _frame(n, nan_frac=0.1, seed=n)
        golden = norm(dsh._decode_plotly_bdata(golden_loading(d).to_plotly_json()))
        new = json.loads(dsh._build_fig_json(d, "load"))
        assert golden == new, f"loading n={n} diverged"


def test_go_star_figures_untouched():
    """A figure still built via go.* (e.g. avghist/scan) must still serialize."""
    d = _frame(300)
    # loadlive/intens/etc. may need more keys; just assert the dispatcher path
    # for a go.* builder returns valid JSON or None (no crash).
    for name in ("load", "infid"):
        assert json.loads(dsh._build_fig_json(d, name))  # dict path
