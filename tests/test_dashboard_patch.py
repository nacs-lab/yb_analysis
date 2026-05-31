"""Tests for the dashboard's partial-update (Patch) machinery.

The dashboard freeze fix replaced "return ~16 whole figures every tick" with
"build the full figure, emit a Dash Patch carrying only what changed". These
tests pin the differ (`_diff_into`) and the emit lifecycle (`_emit`) so a
future edit can't silently turn a minimal patch back into a full re-render or,
worse, emit a patch that corrupts the client figure.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_dashboard_patch.py -v
"""
import numpy as np
import pytest
import plotly.graph_objects as go
from dash import Patch, no_update

from yb_analysis.plotting import dashboard as dsh


@pytest.fixture(autouse=True)
def _clear_fig_cache():
    """Each test starts with an empty per-panel figure cache."""
    dsh._last_figs.clear()
    yield
    dsh._last_figs.clear()


def _assign_locations(patch):
    """Tuples of every Assign op location recorded in a Patch."""
    return [tuple(op['location']) for op in patch._operations
            if op['operation'] == 'Assign']


# ---- _vals_equal -----------------------------------------------------------

def test_vals_equal_numpy_and_nan():
    assert dsh._vals_equal(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
    assert not dsh._vals_equal(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert not dsh._vals_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_vals_equal_lists_and_scalars():
    assert dsh._vals_equal([1, 2, 3], [1, 2, 3])
    assert not dsh._vals_equal([1, 2], [1, 2, 3])
    assert dsh._vals_equal('#fff', '#fff')
    assert dsh._vals_equal(np.int64(5), 5)
    assert dsh._vals_equal(float('nan'), float('nan'))


# ---- _diff_into ------------------------------------------------------------

def test_identical_figures_no_change():
    f = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3])).to_plotly_json()
    patch = Patch()
    assert dsh._diff_into(patch, f, f) is False
    assert patch._operations == []


def test_single_trace_y_change_is_minimal():
    f1 = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3])).to_plotly_json()
    f2 = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 4])).to_plotly_json()
    patch = Patch()
    assert dsh._diff_into(patch, f1, f2) is True
    locs = _assign_locations(patch)
    # Only y changed; x is identical so it must NOT be reassigned.
    assert ('data', 0, 'y') in locs
    assert ('data', 0, 'x') not in locs


def test_shape_color_change_is_granular():
    f1 = go.Figure()
    f1.update_layout(shapes=[dict(type='rect', x0=0, x1=1, y0=0, y1=1,
                                  line=dict(color='#ffffff', width=2))])
    f2 = go.Figure()
    f2.update_layout(shapes=[dict(type='rect', x0=0, x1=1, y0=0, y1=1,
                                  line=dict(color='#000000', width=2))])
    patch = Patch()
    assert dsh._diff_into(patch, f1.to_plotly_json(), f2.to_plotly_json()) is True
    # Recolor must touch only the one shape's line color, not rebuild shapes.
    assert ('layout', 'shapes', 0, 'line', 'color') in _assign_locations(patch)
    assert ('layout', 'shapes') not in _assign_locations(patch)


def test_trace_count_change_replaces_data_list():
    f1 = go.Figure([go.Scatter(y=[1, 2])]).to_plotly_json()
    f2 = go.Figure([go.Scatter(y=[1, 2]), go.Scatter(y=[3, 4])]).to_plotly_json()
    patch = Patch()
    assert dsh._diff_into(patch, f1, f2) is True
    # A length change replaces the whole data list (safe + correct).
    assert ('data',) in _assign_locations(patch)


def test_trace_type_flip_is_structural():
    f1 = go.Figure(go.Scatter(y=[1, 2, 3])).to_plotly_json()
    f2 = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]])).to_plotly_json()
    with pytest.raises(dsh._Structural):
        dsh._diff_into(Patch(), f1, f2)


def test_dropped_key_is_structural():
    old = {'data': [], 'layout': {'title': 'x', 'foo': 1}}
    new = {'data': [], 'layout': {'title': 'x'}}
    with pytest.raises(dsh._Structural):
        dsh._diff_into(Patch(), old, new)


def test_nan_heatmap_unchanged_no_patch():
    z = [[1.0, np.nan], [np.nan, 0.5]]
    f1 = go.Figure(go.Heatmap(z=z)).to_plotly_json()
    f2 = go.Figure(go.Heatmap(z=z)).to_plotly_json()
    patch = Patch()
    # NaN cells must compare equal so an idle scan curve doesn't churn.
    assert dsh._diff_into(patch, f1, f2) is False


# ---- _emit lifecycle -------------------------------------------------------

def test_emit_first_call_is_full_and_caches():
    f = go.Figure(go.Scatter(y=[1, 2, 3]))
    out = dsh._emit('p', f, force_full=False)
    assert out is f                       # no prior cache -> full figure
    assert 'p' in dsh._last_figs


def test_emit_unchanged_returns_no_update():
    dsh._emit('p', go.Figure(go.Scatter(y=[1, 2, 3])), force_full=False)
    out = dsh._emit('p', go.Figure(go.Scatter(y=[1, 2, 3])), force_full=False)
    assert out is no_update


def test_emit_changed_returns_patch():
    dsh._emit('p', go.Figure(go.Scatter(y=[1, 2, 3])), force_full=False)
    out = dsh._emit('p', go.Figure(go.Scatter(y=[1, 2, 9])), force_full=False)
    assert isinstance(out, Patch)
    assert ('data', 0, 'y') in _assign_locations(out)


def test_emit_force_full_returns_full_even_when_cached():
    dsh._emit('p', go.Figure(go.Scatter(y=[1, 2, 3])), force_full=False)
    f3 = go.Figure(go.Scatter(y=[1, 2, 9]))
    out = dsh._emit('p', f3, force_full=True)   # keyframe / page-load resync
    assert out is f3


def test_emit_structural_change_returns_full_figure():
    dsh._emit('p', go.Figure(go.Scatter(y=[1, 2, 3])), force_full=False)
    f_heat = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]]))
    out = dsh._emit('p', f_heat, force_full=False)
    assert out is f_heat                  # type flip -> whole figure


def test_force_full_schedule():
    assert dsh._force_full(0) is True                      # initial page load
    assert dsh._force_full(dsh._KEYFRAME_EVERY) is True    # keyframe
    assert dsh._force_full(1) is False
    assert dsh._force_full(dsh._KEYFRAME_EVERY - 1) is False


# ---- downsample toggle (browser -> main reverse channel) -------------------

def test_control_file_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(dsh, '_CONTROL_FILE', str(tmp_path / 'ctrl.pkl'))
    assert dsh._read_control() is None                 # never written -> None
    dsh._write_control({'downsample': False})
    assert dsh._read_control() == {'downsample': False}
    dsh._write_control({'downsample': True})
    assert dsh._read_control() == {'downsample': True}


def test_downsample_toggle_changes_payload():
    rng = np.random.default_rng(0)
    img = rng.normal(200, 8, (2304, 4096)).astype(np.int16)
    full_uri = dsh._img_to_data_uri(img, max_dim=None)[0]    # toggle OFF
    ds_uri = dsh._img_to_data_uri(img, max_dim=1400)[0]      # toggle ON
    # Downsampling a full-sensor frame must shrink the data URI dramatically.
    assert len(ds_uri) < len(full_uri) / 5


def test_downsample_preserves_intensity_mapping():
    rng = np.random.default_rng(1)
    img = rng.normal(200, 8, (2304, 4096)).astype(np.int16)
    _, vlo_a, vhi_a = dsh._img_to_data_uri(img, max_dim=None)
    _, vlo_b, vhi_b = dsh._img_to_data_uri(img, max_dim=1400)
    # vlo/vhi come from the full image (pre-downsample) -> identical colorbar.
    assert (vlo_a, vhi_a) == (vlo_b, vhi_b)


# ---- large-array build gating + WebGL --------------------------------------

def test_h_detects_array_change():
    a = np.arange(1000, dtype=float)
    assert dsh._h(a) == dsh._h(a.copy())          # same content -> same hash
    b = a.copy(); b[500] = -1
    assert dsh._h(a) != dsh._h(b)                  # one element differs
    assert dsh._h(None) is None


def test_gated_skips_when_signature_unchanged():
    dsh._last_sig.clear()
    calls = {'n': 0}
    def build():
        calls['n'] += 1
        return go.Figure(go.Scatter(y=[1, 2, 3]))
    sig = ('load', 12, False, dsh._h(np.zeros(3270)))
    # First call builds; identical signature on the next tick skips the build.
    assert dsh._gated('load', sig, build, force_full=False) is not dsh._SKIP
    assert dsh._gated('load', sig, build, force_full=False) is dsh._SKIP
    assert calls['n'] == 1
    # force_full always rebuilds (keyframe / page-load resync).
    assert dsh._gated('load', sig, build, force_full=True) is not dsh._SKIP
    assert calls['n'] == 2


def test_emit_skip_returns_no_update():
    assert dsh._emit('load', dsh._SKIP, force_full=False) is no_update


def test_large_array_uses_webgl_scattergl():
    rng = np.random.default_rng(0)
    n = 3270
    grid = np.column_stack([rng.integers(0, 2000, n), rng.integers(0, 2000, n)]).astype(float)
    rates = rng.random(n)
    inf = rng.random(n) * 0.01
    d = {'grid_locations': grid, 'loading_rates': rates, 'infidelities': inf}
    load_types = {tr['type'] for tr in dsh._fig_loading(d).to_plotly_json()['data']}
    infid_types = {tr['type'] for tr in dsh._fig_infid(d).to_plotly_json()['data']}
    assert load_types == {'scattergl'}
    assert infid_types == {'scattergl'}


def test_large_array_has_no_per_site_shapes():
    # The array panel must NOT emit thousands of layout shapes for a big array
    # (that was the freeze); it uses a single WebGL occupancy trace instead.
    rng = np.random.default_rng(0)
    n = 3270
    grid = np.column_stack([rng.integers(0, 2000, n), rng.integers(0, 2000, n)]).astype(float)
    uri, vlo, vhi = dsh._img_to_data_uri(rng.normal(200, 8, (200, 200)).astype(np.int16))
    d = {'_img_data_uri': uri, '_img_shape': (200, 200), '_img_vlo': vlo,
         '_img_vhi': vhi, 'grid_locations': grid,
         'logicals': (rng.random(n) > 0.5), 'box_size': 11}
    fig = dsh._fig_array(d).to_plotly_json()
    assert not fig['layout'].get('shapes')          # no per-site SVG rects
    assert any(tr['type'] == 'scattergl' for tr in fig['data'])
