"""3-D scan cube + coupled-1-D scan-curve support (live dashboard).

Covers:
  * ``compute_scan_curve`` producing an ndim=3 data cube (no silent dim-2
    truncation, correct column-major decomposition).
  * ``extract_scan_dims`` collecting EVERY coupled parameter per dimension.
  * ``_fig_scan_3d`` baking one frame per slice + a slider, honoring the
    slice-dim choice; ``_fig_scan_curve`` re-labeling/re-sorting a coupled
    1-D curve by the chosen x-param.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  compute_scan_curve -> 3-D cube
# ---------------------------------------------------------------------------

def _cube_dims(s0, s1, s2, names=('d0', 'd1', 'd2')):
    return [
        {'name': names[0], 'values': np.arange(s0) * 1.0, 'size': s0},
        {'name': names[1], 'values': np.arange(s1) * 1.0, 'size': s1},
        {'name': names[2], 'values': np.arange(s2) * 1.0, 'size': s2},
    ]


def test_compute_scan_curve_3d_no_truncation():
    """A 3-D scan must fill the whole cube; the old 2-D path dropped every
    shot whose dim-2 index >= 1 (p >= s0*s1). Column-major decomposition:
    i0 = p%s0, i1 = (p//s0)%s1, i2 = p//(s0*s1)."""
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    s0, s1, s2 = 2, 2, 3
    dims = _cube_dims(s0, s1, s2)
    n_total = s0 * s1 * s2
    n_sites = 4
    # One rep per flat param index; survival = flat index / n_total so every
    # cell is distinguishable and the reshape order is checkable.
    sl = []
    for p in range(n_total):
        frac = (p + 1) / (n_total + 1)
        # img1 all loaded; img2 loaded on ~frac of sites (deterministic).
        k = int(round(frac * n_sites))
        l1 = np.ones(n_sites, bool)
        l2 = np.zeros(n_sites, bool)
        l2[:k] = True
        sl.append((p + 1, l1, l2))          # seq_id 1-indexed
    pidx = np.arange(1, n_total + 1)         # seq_id -> flat param index

    out = compute_scan_curve(sl, pidx, None, 2, scan_dims=dims)
    assert out is not None and out['ndim'] == 3
    cube = out['cube']
    assert cube.shape == (s2, s1, s0)        # (dim2, dim1, dim0)
    reps = out['n_reps']
    # EVERY cell got exactly one rep -> nothing dropped.
    assert reps.shape == (s2, s1, s0)
    assert int(reps.sum()) == n_total
    assert np.all(reps == 1)
    # Spot-check the column-major mapping: flat p -> cube[i2, i1, i0].
    for p in range(n_total):
        i0 = p % s0
        i1 = (p // s0) % s1
        i2 = p // (s0 * s1)
        frac = (p + 1) / (n_total + 1)
        expected = int(round(frac * n_sites)) / n_sites
        assert abs(cube[i2, i1, i0] - expected) < 1e-9


def test_compute_scan_curve_4d_folds_into_slice_axis():
    """>3 dims: extra dims fold into the slice axis so no shot is dropped."""
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    sizes = (2, 2, 2, 2)
    dims = [{'name': f'd{i}', 'values': np.arange(s) * 1.0, 'size': s}
            for i, s in enumerate(sizes)]
    n_total = int(np.prod(sizes))
    sl = [(p + 1, np.ones(3, bool), np.ones(3, bool)) for p in range(n_total)]
    pidx = np.arange(1, n_total + 1)
    out = compute_scan_curve(sl, pidx, None, 2, scan_dims=dims)
    assert out['ndim'] == 3
    # slice axis size = product of dims 2..N = 2*2 = 4
    assert out['cube'].shape == (4, 2, 2)
    assert int(out['n_reps'].sum()) == n_total


# ---------------------------------------------------------------------------
#  extract_scan_dims -> coupled params
# ---------------------------------------------------------------------------

def test_extract_scan_dims_collects_coupled_params():
    """A coupled 1-D sweep sets several params along the one axis; every one
    must appear in dim['coupled'] (the first is still name/values)."""
    from yb_analysis.detection.scan_analysis import extract_scan_dims
    cfg = {"ScanGroup": {"base": {"vars": {
        "params": {"Beam": {"Freq": [10, 20, 30], "Amp": [1, 2, 3]}},
        "size": 3,
    }}}}
    dims = extract_scan_dims(cfg)
    assert dims is not None and len(dims) == 1
    coupled = dims[0].get('coupled')
    assert coupled is not None and len(coupled) == 2
    names = {c['name'] for c in coupled}
    assert names == {"Beam.Freq", "Beam.Amp"}
    # primary == first coupled entry
    assert dims[0]['name'] == coupled[0]['name']


def test_extract_scan_dims_single_param_no_coupled_list():
    """A plain 1-D scan has a length-1 coupled list (the primary only)."""
    from yb_analysis.detection.scan_analysis import extract_scan_dims
    cfg = {"ScanGroup": {"base": {"vars": {
        "params": {"Beam": {"Freq": [10, 20, 30]}}, "size": 3,
    }}}}
    dims = extract_scan_dims(cfg)
    assert len(dims[0]['coupled']) == 1


def test_compute_scan_curve_1d_exposes_coupled():
    """The 1-D curve carries a 'coupled' list (sorted with the curve) only
    when the dim really sweeps > 1 param."""
    from yb_analysis.detection.scan_analysis import compute_scan_curve
    dims = [{'name': 'Freq', 'values': np.array([20.0, 10.0]), 'size': 2,
             'coupled': [{'name': 'Freq', 'values': np.array([20.0, 10.0])},
                         {'name': 'Amp', 'values': np.array([2.0, 1.0])}]}]
    sl = [(1, np.ones(2, bool), np.ones(2, bool)),
          (2, np.ones(2, bool), np.ones(2, bool))]
    pidx = np.array([1, 2])
    sp = np.array([20.0, 10.0])
    out = compute_scan_curve(sl, pidx, sp, 2, scan_dims=dims)
    assert out['ndim'] if 'ndim' in out else True   # 1-D has no ndim key
    coupled = out.get('coupled')
    assert coupled is not None and len(coupled) == 2
    # Sorted by scan_params ascending (10, 20): Amp follows Freq's reorder.
    assert np.allclose(out['scan_x'], [10.0, 20.0])
    amp = [c for c in coupled if c['name'] == 'Amp'][0]['values']
    assert np.allclose(amp, [1.0, 2.0])


# ---------------------------------------------------------------------------
#  Figure builders
# ---------------------------------------------------------------------------

def _cube_sc(s0, s1, s2):
    """A minimal ndim=3 scan_curve dict for the figure builder."""
    rng = np.arange(s0 * s1 * s2, dtype=float).reshape(s2, s1, s0)
    return {
        'mode': 'survival', 'ndim': 3,
        'cube': rng / rng.max(),
        'sem': np.full((s2, s1, s0), 0.01),
        'n_reps': np.ones((s2, s1, s0), int),
        'dims': _cube_dims(s0, s1, s2, names=('Freq', 'Amp', 'Depth')),
        'current': [{'x_idx': 0, 'y_idx': 0, 'z_idx': 1}],
    }


def test_fig_scan_3d_frames_and_slider():
    """_fig_scan_3d bakes one frame per slice + a slider; default slice = the
    slice holding the current-scan cell."""
    from yb_analysis.plotting import dashboard as D
    s0, s1, s2 = 2, 3, 4
    d = {'scan_curve': _cube_sc(s0, s1, s2), 'scan_name': 'X', 'plot_scale': 1}
    fig = D._fig_scan_curve(d)            # routes to _fig_scan_3d for ndim>=3
    js = fig.to_plotly_json()
    # default slice axis = dim2 (Depth, s2) -> one frame per Depth value.
    assert len(js['frames']) == s2
    assert 'sliders' in js['layout'] and len(js['layout']['sliders']) == 1
    slider = js['layout']['sliders'][0]
    assert len(slider['steps']) == s2
    assert slider['active'] == 1          # current cell z_idx=1


def test_fig_scan_3d_slice_dim_choice():
    """slice_dim picks which axis the slider walks; frame count == that dim."""
    from yb_analysis.plotting import dashboard as D
    s0, s1, s2 = 2, 3, 4
    d = {'scan_curve': _cube_sc(s0, s1, s2), 'scan_name': 'X', 'plot_scale': 1}
    # slice on dim0 (size 2) -> 2 frames, heatmap over (dim1, dim2)
    fig = D._fig_scan_curve(d, scan_opts=(0, None))
    js = fig.to_plotly_json()
    assert len(js['frames']) == s0
    # slice on dim1 (size 3) -> 3 frames
    fig = D._fig_scan_curve(d, scan_opts=(1, None))
    assert len(fig.to_plotly_json()['frames']) == s1


def test_fig_scan_3d_slice_values_are_correct():
    """The slice + transpose must preserve cell values: for a slice on dim0,
    frame k's heatmap[y_idx=dim1, x_idx=dim2] == cube[dim2, dim1, dim0=k].
    Guards the axis bookkeeping in _fig_scan_3d._slice."""
    from yb_analysis.plotting import dashboard as D
    s0, s1, s2 = 2, 3, 4
    sc = _cube_sc(s0, s1, s2)
    cube = sc['cube']                    # (s2, s1, s0)
    d = {'scan_curve': sc, 'scan_name': 'X', 'plot_scale': 1}
    # Slice on dim0 -> plane dims are (dim1, dim2): x=dim1(s1), y=dim2(s2).
    fig = D._fig_scan_curve(d, scan_opts=(0, None))
    frames = fig.frames
    assert len(frames) == s0
    for k in range(s0):                  # k = dim0 index (the slider position)
        z = np.asarray(frames[k].data[0].z, dtype=float)   # shape (ny, nx)
        # plane_dims=[1,2] -> x_dim=dim1 (nx=s1), y_dim=dim2 (ny=s2)
        assert z.shape == (s2, s1)
        for i1 in range(s1):             # dim1 along x
            for i2 in range(s2):         # dim2 along y
                assert abs(z[i2, i1] - cube[i2, i1, k]) < 1e-9


def test_fig_scan_curve_coupled_x_choice():
    """A coupled 1-D curve relabels + re-sorts by the chosen coupled param."""
    from yb_analysis.plotting import dashboard as D
    sc = {
        'mode': 'survival',
        'scan_x': np.array([10.0, 20.0]),
        'y_mean': np.array([0.9, 0.8]),
        'y_sem': np.array([0.01, 0.01]),
        'n_reps': np.array([5, 5]),
        'coupled': [
            {'name': 'Freq', 'values': np.array([10.0, 20.0])},
            {'name': 'Amp', 'values': np.array([2.0, 1.0])},   # decreasing
        ],
    }
    d = {'scan_curve': sc, 'scan_name': 'X', 'plot_scale': 1,
         'scan_param_path': 'Freq'}
    # x = Amp (index 1): curve must re-sort ascending by Amp (1,2).
    fig = D._fig_scan_curve(d, scan_opts=(None, 1))
    # Read from the go.Figure trace directly (to_plotly_json bdata-encodes x).
    xtitle = fig.layout.xaxis.title.text
    assert 'Amp' in xtitle
    xs = np.asarray(fig.data[0].x, dtype=float)
    assert np.all(np.diff(xs) >= 0)       # ascending along Amp
    # The hover for each point names the OTHER coupled param (Freq).
    assert any('Freq' in t for t in fig.data[0].hovertext)
