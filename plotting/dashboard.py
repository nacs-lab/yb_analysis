"""Yb Tweezer Dashboard — Plotly Dash, single-callback architecture.

Layout:
  Row 1: [Tweezer Array]  [Atom Intensities]
  Row 2: [Loading 2D] [Infidelities 2D] [Grid Shift] [Avg Histogram]
  Row 3: [4 rep site histograms — stable between refits]
  Row 4: [Interactive site selector + histogram]
  /debug endpoint for state inspection
"""

import base64
import io
import json
import os
import pickle
import time
import multiprocessing
import tempfile
import logging
import traceback
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from dash import Dash, html, dcc, Input, Output, State, no_update

logger = logging.getLogger(__name__)

# Theme
BG = '#0a0a16'
PANEL = '#0d1220'
TEXT = '#e0e0e0'
GRID = '#1a1a30'
_L = dict(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, size=10),
          margin=dict(l=40, r=15, t=35, b=30), uirevision='live')
_A = dict(gridcolor=GRID, zerolinecolor=GRID)

# Shared data file path
_DATA_FILE = os.path.join(tempfile.gettempdir(), 'yb_dash_data.pkl')


class DashboardRenderer:
    """Runs the Dash web server in a **separate process** to avoid GIL
    starvation from the heavy image-processing thread.

    Data is shared via a pickle file: the main process writes it,
    the Dash process reads it on each callback tick.
    """

    def __init__(self, port=8050):
        self._port = port
        self._proc = None

    def start(self):
        if self._proc is None or not self._proc.is_alive():
            self._proc = multiprocessing.Process(
                target=_dash_main, args=(self._port, _DATA_FILE), daemon=True)
            self._proc.start()
            logger.info('Dashboard process started (pid=%d) at http://localhost:%d',
                        self._proc.pid, self._port)

    def update(self, data):
        """Write plot data to the shared file.

        Downsample large images to keep pickle small (~1MB vs 32MB).
        Uses double-buffer strategy to avoid Windows file locking.
        """
        d = dict(data)
        # Pre-encode image to JPEG data URI in the main process so the
        # Dash callback (separate process) doesn't pay the ~130ms PIL cost.
        img = d.get('cur_image')
        if img is not None:
            uri, vlo, vhi = _img_to_data_uri(np.asarray(img, dtype=np.int16))
            d['_img_data_uri'] = uri
            d['_img_shape'] = img.shape
            d['_img_vlo'] = vlo
            d['_img_vhi'] = vhi
            d.pop('cur_image', None)  # don't pickle the raw image (18MB)
        # Write to alternating files to avoid read/write conflicts on Windows
        idx = getattr(self, '_write_idx', 0)
        target = _DATA_FILE + f'.{idx}'
        with open(target, 'wb') as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Update pointer (tiny file, fast write)
        with open(_DATA_FILE, 'w') as f:
            f.write(str(idx))
        self._write_idx = 1 - idx  # toggle 0 ↔ 1
        self.start()

    def close(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1)
            logger.info('Dashboard process stopped')
        for p in (_DATA_FILE, _DATA_FILE + '.0', _DATA_FILE + '.1'):
            try:
                os.remove(p)
            except OSError:
                pass


def _read_data():
    """Read plot data from the shared pickle file (called in Dash process).

    Uses pointer file to find which buffer to read (avoids Windows lock conflicts).
    """
    try:
        with open(_DATA_FILE, 'r') as f:
            idx = f.read().strip()
        with open(_DATA_FILE + f'.{idx}', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, ValueError, pickle.UnpicklingError, OSError):
        return None


def _dash_main(port, data_file):
    """Entry point for the Dash subprocess."""
    # Reconfigure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [dash] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    app = _build_app()
    app.run(port=port, debug=False, use_reloader=False)


def _build_app():
    app = Dash(__name__, title='Yb Tweezer Dashboard')

    # Force crisp pixel rendering on zoomed images. Plotly recreates SVG
    # elements on each update, so CSS alone doesn't stick. A MutationObserver
    # re-applies the style whenever a new <image> element appears.
    app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
new MutationObserver(function(mutations) {
    document.querySelectorAll('image').forEach(function(el) {
        if (el.style.imageRendering !== 'pixelated') {
            el.style.imageRendering = 'pixelated';
        }
    });
    /* Bind plotly_click on load/infid graphs → update site dropdown.
       Re-binds after each DOM mutation so it survives figure refreshes. */
    ['load', 'infid'].forEach(function(gid) {
        var el = document.getElementById(gid);
        if (el && el.classList.contains('js-plotly-plot') && !el._ybClick) {
            el._ybClick = true;
            el.on('plotly_click', function(evtData) {
                if (!evtData || !evtData.points || !evtData.points.length) return;
                var pt = evtData.points[0];
                var site = (pt.customdata != null) ? pt.customdata
                         : (pt.pointIndex != null) ? pt.pointIndex + 1 : null;
                if (site == null) return;
                /* Dash stores component props on the React fiber. We can update
                   the dropdown by finding its Dash component and calling setProps. */
                var dd = document.getElementById('site-dd');
                if (!dd) return;
                var key = Object.keys(dd).find(function(k) {
                    return k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$');
                });
                if (key) {
                    var fiber = dd[key];
                    /* Walk up to find the Dash component with setProps */
                    var node = fiber;
                    for (var i = 0; i < 30 && node; i++) {
                        if (node.memoizedProps && typeof node.memoizedProps.setProps === 'function') {
                            node.memoizedProps.setProps({value: site});
                            break;
                        }
                        node = node.return;
                    }
                }
            });
        }
    });
}).observe(document.body, {childList: true, subtree: true});
</script></body></html>'''

    app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh',
        'fontFamily': '"Segoe UI", sans-serif', 'color': TEXT, 'padding': '10px'}, children=[
        html.H1('Yb Tweezer Dashboard', style={'textAlign': 'center', 'color': '#e94560',
            'margin': '5px 0 10px 0', 'fontSize': '24px'}),
        # Row 1+2: Tweezer Array (spans both rows) | right column stacked
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}, children=[
            # Left: Tweezer Array (tall, spanning 2 rows)
            _graph('array', 670),
            # Right: stacked panels
            html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
                _graph('intens', 330),
                html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                    _graph('shift', 330), _graph('scan', 330),
                ]),
            ]),
        ]),
        # Row 3: Avg Histogram + Rep site histograms
        _row([_graph('avghist', 240)] + [_graph(f'rep{i}', 240) for i in range(4)]),
        # Row 4: Loading Rates | Infidelities | Site selector (equal thirds)
        _row([
            _graph('load', 280),
            _graph('infid', 280),
            html.Div(style={'flex': '1', 'minWidth': '0', 'display': 'flex', 'gap': '8px'}, children=[
                # Left: dropdown + parameters
                html.Div(style={'width': '140px', 'flexShrink': '0'}, children=[
                    html.Label('Site:', style={'fontSize': '12px'}),
                    dcc.Dropdown(id='site-dd', options=[], value=1, clearable=False,
                                 style={'backgroundColor': '#2b2b4a', 'color': '#222', 'marginBottom': '8px'}),
                    html.Div(id='site-info', style={'fontSize': '11px', 'color': '#bbb',
                        'lineHeight': '1.6'}),
                ]),
                # Right: histogram
                _graph('site', 270),
            ]),
        ]),
        # Debug
        html.Details([
            html.Summary('Debug Info', style={'cursor': 'pointer', 'color': '#888', 'fontSize': '11px'}),
            html.Pre(id='debug-pre', style={'fontSize': '10px', 'color': '#aaa',
                'maxHeight': '300px', 'overflow': 'auto', 'whiteSpace': 'pre-wrap'}),
        ], style={'marginTop': '10px'}),
        dcc.Interval(id='tick', interval=3000, n_intervals=0),
    ])

    # --- Single callback for all panels ---
    outputs = ([Output('array', 'figure'), Output('intens', 'figure'),
                 Output('load', 'figure'), Output('infid', 'figure'),
                 Output('shift', 'figure'), Output('scan', 'figure'),
                 Output('avghist', 'figure')]
               + [Output(f'rep{i}', 'figure') for i in range(4)]
               + [Output('site-dd', 'options'), Output('debug-pre', 'children')])

    @app.callback(outputs, Input('tick', 'n_intervals'))
    def refresh(_n):
        d = _read_data()
        debug_lines = []

        if d is None:
            debug_lines.append('No data yet (pickle file not found)')
            empty = [_waiting(t) for t in ['Tweezer Array', 'Intensities',
                     'Loading', 'Infidelities', 'Grid Shift', 'Scan Curve', 'Avg Histogram']]
            return empty + [_waiting('Site Hist')]*4 + [[], '\n'.join(debug_lines)]

        try:
            has_img = d.get('_img_data_uri') is not None
            n = d.get('num_sites', 0)
            v = d.get('hist_version', 0)
            n_acc = d.get('n_accum_shots', 0)

            figs = [
                _fig_array(d) if has_img else _waiting('Tweezer Array'),
                _fig_intens(d),
                _fig_loading(d),
                _fig_infid(d),
                _fig_shift(d),
                _fig_scan_curve(d),
                _fig_avghist(d),
            ]

            reps = _figs_reps(d)
            opts = [{'label': f'Site {i+1}', 'value': i+1} for i in range(n)]

            lh = d.get('live_hist_data')
            lf = d.get('live_gauss_fits')
            ldf = d.get('loaded_gauss_fits')
            debug_lines.append(f'sites={n} accum={n_acc} hist_v={v}')
            debug_lines.append(f'live_hist: {"list["+str(len(lh))+"]" if isinstance(lh, list) else type(lh).__name__}')
            debug_lines.append(f'live_fits: {"list["+str(len(lf))+"]" if isinstance(lf, list) else type(lf).__name__}')
            debug_lines.append(f'loaded_fits: {"list["+str(len(ldf))+"]" if isinstance(ldf, list) else type(ldf).__name__}')
            debug_lines.append(f'img={has_img} rep_sites={d.get("hist_rep_sites")}')

            return figs + reps + [opts, '\n'.join(debug_lines)]

        except Exception:
            tb = traceback.format_exc()
            logging.error('Dashboard render error:\n%s', tb)
            return [no_update] * 13

    @app.callback([Output('site', 'figure'), Output('site-info', 'children')],
                  [Input('site-dd', 'value'), Input('tick', 'n_intervals')])
    def site_hist(val, _n):
        d = _read_data()
        if d is None or val is None:
            return _waiting('Site Histogram'), ''
        return _fig_site(d, int(val) - 1)

    # Click on loading-rate or infidelity 2D plot → select site in dropdown
    # is handled entirely in JavaScript (index_string) via plotly_click +
    # React fiber setProps. This bypasses Dash's callback system which can
    # lose clickData when the refresh callback replaces figures every 3s.

    return app


# ---- Helpers ----

def _row(children):
    return html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}, children=children)

def _graph(id, h):
    # Set initial "waiting" figure so Plotly has a uirevision baseline.
    # Without this, Plotly may not re-render when the callback first returns
    # a figure with uirevision='live' (no prior value to compare against).
    return dcc.Graph(id=id, figure=_waiting(''), style={'flex': '1', 'height': f'{h}px'},
                     config={'displayModeBar': False})

def _waiting(title):
    fig = go.Figure()
    fig.add_annotation(text='Waiting for data...', x=0.5, y=0.5, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=14, color='#666'))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, size=10),
                      margin=dict(l=40, r=15, t=35, b=30), uirevision='waiting',
                      title=title)
    return fig


# ---- Figure builders ----

def _img_to_data_uri(img):
    """Convert int16 image to a lossless PNG data URI for Plotly background.

    Uses cv2 with PNG compression=0 (no compression, just framing) for speed.
    83ms for 4000x2300 — fully lossless, no artifacts.
    """
    import cv2
    vlo, vhi = float(np.percentile(img, 2)), float(np.percentile(img, 98))
    gray = np.clip((img.astype(np.float32) - vlo) / max(vhi - vlo, 1) * 255, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.png', gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    b64 = base64.b64encode(enc.tobytes()).decode()
    return f'data:image/png;base64,{b64}', vlo, vhi


def _fig_array(d):
    data_uri = d.get('_img_data_uri')
    shape = d.get('_img_shape')
    if data_uri is None or shape is None:
        return _waiting('Tweezer Array')
    H, W = shape
    fig = go.Figure()
    fig.add_layout_image(
        source=data_uri, xref='x', yref='y',
        x=0, y=0, sizex=W, sizey=H,
        sizing='stretch', layer='below',
    )
    # Colorbar via invisible scatter + autorange anchor at image corners
    vlo = d.get('_img_vlo', 0)
    vhi = d.get('_img_vhi', 255)
    fig.add_trace(go.Scatter(
        x=[0, W, 0, W], y=[0, 0, H, H], mode='markers',
        marker=dict(size=0.1, opacity=0, color=[vlo, vhi, vlo, vhi],
                    colorscale='gray', cmin=vlo, cmax=vhi, showscale=True,
                    colorbar=dict(title='Counts', len=0.9)),
        hoverinfo='skip', showlegend=False))

    # Site markers as lightweight scatter overlay
    grid = d.get('grid_locations')
    logicals = d.get('logicals')
    box = d.get('box_size', 11)
    n = len(grid) if grid is not None else 0
    if grid is not None:
        half = box / 2
        # Batch layout shapes — rectangles in data coordinates that scale with zoom
        shapes = []
        for i in range(n):
            y0, x0 = grid[i]
            c = '#00ff88' if (logicals is not None and i < n and logicals[i]) else '#ff4444'
            shapes.append(dict(type='rect', x0=x0-half, y0=y0-half, x1=x0+half, y1=y0+half,
                                line=dict(color=c, width=2)))
        fig.update_layout(shapes=shapes)
        if n <= 200:
            # Text labels only for small arrays
            fig.add_trace(go.Scatter(
                x=grid[:, 1], y=grid[:, 0] - half - 3, mode='text',
                text=[str(i+1) for i in range(n)],
                textfont=dict(color='#ffdd44', size=7),
                hoverinfo='skip', showlegend=False))

    fig.update_layout(**_L, title='Tweezer Array',
                      xaxis=dict(range=[0, W], showgrid=False, zeroline=False, **_A),
                      yaxis=dict(range=[H, 0], scaleanchor='x', scaleratio=1,
                                 showgrid=False, zeroline=False, **_A))
    return fig


def _fig_intens(d):
    t = d.get('thresholds')
    if t is None or len(t) == 0:
        return _waiting('Intensities')
    n = len(t)
    sites = list(range(1, n+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sites, y=t.tolist(), mode='markers', name='Threshold',
                              marker=dict(size=10, color='#777', symbol='circle', line=dict(width=1, color='#999'))))
    ymin, ymax = float(t.min()), float(t.max())
    ci = d.get('cur_intensities')
    if ci is not None:
        logicals = d.get('logicals')
        colors = ['#0c6' if (logicals is not None and i < len(logicals) and logicals[i]) else '#e44' for i in range(n)]
        fig.add_trace(go.Scatter(x=sites, y=ci.tolist(), mode='markers', name='Current',
                                  marker=dict(size=12, color=colors, symbol='circle', line=dict(width=1, color='white'))))
        ymin = min(ymin, float(ci.min()))
        ymax = max(ymax, float(ci.max()))
    # Mean lines for loaded / empty sites + distance annotation
    if ci is not None and logicals is not None:
        mask = np.array(logicals[:n], dtype=bool) if len(logicals) >= n else np.zeros(n, dtype=bool)
        if mask.any():
            mu_loaded = float(ci[mask].mean())
            fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=mu_loaded, y1=mu_loaded,
                          line=dict(color='#0c6', width=1.5, dash='dash'))
            fig.add_annotation(text=f'Loaded: {mu_loaded:.1f}', xref='paper', y=mu_loaded,
                               x=1.0, showarrow=False, xanchor='left',
                               font=dict(color='#0c6', size=10))
        else:
            mu_loaded = None
        if (~mask).any():
            mu_empty = float(ci[~mask].mean())
            fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=mu_empty, y1=mu_empty,
                          line=dict(color='#e44', width=1.5, dash='dash'))
            fig.add_annotation(text=f'Empty: {mu_empty:.1f}', xref='paper', y=mu_empty,
                               x=1.0, showarrow=False, xanchor='left',
                               font=dict(color='#e44', size=10))
        else:
            mu_empty = None
        if mu_loaded is not None and mu_empty is not None:
            delta = mu_loaded - mu_empty
            fig.add_annotation(text=f'Δ = {delta:.2f}', xref='paper', yref='paper',
                               x=0.5, y=1.0, showarrow=False,
                               font=dict(size=12, color='#ffdd44', family='monospace'),
                               bgcolor='rgba(20,20,40,0.8)')

    pad = max((ymax - ymin) * 0.2, 1)
    fig.update_layout(**_L, title='Atom Intensities', xaxis=dict(title='Site', dtick=max(1, n//20), **_A),
                      yaxis=dict(title='Intensity', range=[ymin-pad, ymax+pad], **_A),
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.3)'))
    return fig


def _fig_loading(d):
    grid, rates = d.get('grid_locations'), d.get('loading_rates')
    if grid is None or rates is None or len(grid) == 0:
        return _waiting('Loading Rates')
    n = len(grid)
    if n < 100:
        sz = 20
        mode = 'markers+text'
        text = [f'{r:.0%}' for r in rates]
        tfont = dict(size=7, color='black')
    else:
        sz = max(4, min(12, 800 // int(np.sqrt(n))))
        mode = 'markers'
        text = None
        tfont = None
    fig = go.Figure(go.Scatter(
        x=grid[:,1], y=grid[:,0], mode=mode,
        marker=dict(size=sz, color=rates.tolist(), colorscale='RdYlGn', cmin=0, cmax=1,
                    colorbar=dict(title='Rate', len=0.9), line=dict(width=0.5, color='white')),
        text=text, textfont=tfont, textposition='middle center',
        customdata=[i+1 for i in range(n)],
        hoverinfo='text', hovertext=[f'Site {i+1}: {r:.1%}' for i, r in enumerate(rates)]))
    fig.update_layout(**_L, title=f'Loading Rates ({n} sites)', clickmode='event',
                      yaxis=dict(autorange='reversed', scaleanchor='x', scaleratio=1,
                                 visible=False, **_A),
                      xaxis=dict(visible=False, **_A))
    return fig


def _fig_infid(d):
    grid, inf = d.get('grid_locations'), d.get('infidelities')
    if grid is None or inf is None or len(grid) == 0:
        return _waiting('Infidelities')
    n = len(grid)
    log_inf = np.log10(np.clip(inf, 1e-6, 1.0))
    if n < 100:
        sz = 20
        mode = 'markers+text'
        text = [f'{v:.0e}' for v in inf]
        tfont = dict(size=6, color='white')
    else:
        sz = max(4, min(12, 800 // int(np.sqrt(n))))
        mode = 'markers'
        text = None
        tfont = None
    fig = go.Figure(go.Scatter(
        x=grid[:,1], y=grid[:,0], mode=mode,
        marker=dict(size=sz, color=log_inf.tolist(), colorscale='Magma_r', cmin=-4, cmax=-0.3,
                    colorbar=dict(title='log10', len=0.9), line=dict(width=0.5, color='white')),
        text=text, textfont=tfont, textposition='middle center',
        customdata=[i+1 for i in range(n)],
        hoverinfo='text', hovertext=[f'Site {i+1}: {v:.2e}' for i, v in enumerate(inf)]))
    fig.update_layout(**_L, title=f'Discrimination Infidelities ({n} sites)', clickmode='event',
                      yaxis=dict(autorange='reversed', scaleanchor='x', scaleratio=1,
                                 visible=False, **_A),
                      xaxis=dict(visible=False, **_A))
    return fig


def _fig_shift(d):
    hm = d.get('grid_shift_heatmap')
    if hm is None:
        return _waiting('Grid Shift')
    R = (hm.shape[0]-1)//2
    fig = go.Figure(go.Heatmap(z=hm, x0=-R, dx=1, y0=-R, dy=1, colorscale='Viridis',
                                showscale=True, colorbar=dict(len=0.9)))
    hist = d.get('grid_shift_history', [])
    title = 'Grid Shift Heatmap'
    if hist:
        dy, dx = hist[-1]
        fig.add_trace(go.Scatter(x=[dx], y=[dy], mode='markers',
                                 marker=dict(symbol='cross', size=14, color='red', line=dict(width=2)),
                                 showlegend=False))
        title = f'Grid Shift (dy={dy}, dx={dx})'
    fig.update_layout(**_L, title=title, xaxis=dict(title='dx', **_A),
                      yaxis=dict(title='dy', autorange='reversed', **_A))
    return fig


def _fig_scan_curve(d):
    sc = d.get('scan_curve')
    if sc is None or sc.get('mode') == 'undefined':
        return _waiting('Scan Curve')

    # --- 2-D heatmap ---
    if sc.get('ndim', 1) >= 2:
        return _fig_scan_2d(d, sc)

    # --- 1-D scatter with error bars ---
    x = sc['scan_x']
    y = sc['y_mean']
    err = sc['y_sem']
    n_reps = sc['n_reps']
    mode = sc['mode']
    mask = n_reps > 0
    if not np.any(mask):
        return _waiting('Scan Curve')
    x, y, err, n_reps = x[mask], y[mask], err[mask], n_reps[mask]

    scale = d.get('plot_scale', 1)
    if scale and scale != 0 and scale != 1:
        x_disp = x * scale
    else:
        x_disp = x

    scan_name = d.get('scan_name', 'Scan')
    x_label = d.get('scan_param_path') or scan_name
    y_label = 'Survival' if mode == 'survival' else 'Loading Rate'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_disp, y=y, error_y=dict(type='data', array=err, visible=True, thickness=1.5),
        mode='markers', marker=dict(size=6, color='#44aaff'),
        hoverinfo='text', hovertext=[f'{x_label}={xi:.4g}, {y_label}={yi:.3f}+/-{ei:.3f} (n={ni})'
                                      for xi, yi, ei, ni in zip(x_disp, y, err, n_reps)]))
    fig.update_layout(**_L, title=f'{scan_name} ({int(n_reps.mean())} reps/pt)',
                      xaxis=dict(title=x_label, **_A),
                      yaxis=dict(title=y_label, range=[-0.05, 1.05], **_A))
    return fig


def _fig_scan_2d(d, sc):
    """Render a 2-D scan as a survival/loading heatmap."""
    heatmap = sc.get('heatmap')
    n_reps = sc.get('n_reps')
    if heatmap is None:
        return _waiting('Scan 2D')

    # Mask cells with no data
    mask = (n_reps is not None) and np.any(n_reps > 0)
    if not mask:
        return _waiting('Scan 2D')

    x_vals = sc['x_values']
    y_vals = sc['y_values']
    x_name = sc.get('x_name', 'dim0')
    y_name = sc.get('y_name', 'dim1')
    mode = sc.get('mode', 'survival')
    scan_name = d.get('scan_name', 'Scan')

    scale = d.get('plot_scale', 1)
    if scale and scale != 0 and scale != 1:
        x_disp = x_vals * scale
    else:
        x_disp = x_vals

    z = np.where(n_reps > 0, heatmap, np.nan)
    y_label = 'Survival' if mode == 'survival' else 'Loading'
    avg_reps = int(n_reps[n_reps > 0].mean()) if np.any(n_reps > 0) else 0

    fig = go.Figure(go.Heatmap(
        z=z, x=x_disp, y=y_vals,
        colorscale='Viridis', zmin=0, zmax=1,
        colorbar=dict(title=y_label, len=0.9),
        hovertemplate=f'{x_name}=%{{x:.4g}}<br>{y_name}=%{{y:.4g}}<br>{y_label}=%{{z:.3f}}<extra></extra>',
    ))
    fig.update_layout(**_L, title=f'{scan_name} ({avg_reps} reps/pt)',
                      xaxis=dict(title=x_name, **_A),
                      yaxis=dict(title=y_name, **_A))
    return fig


def _fig_avghist(d):
    fig = go.Figure()
    has_live_f = d.get('live_gauss_fits') is not None
    # Show loaded fit only when no live fit
    if not has_live_f:
        _add_avg_fit_curve(fig, d.get('loaded_gauss_fits'), '#888', 'Loaded fit', faint=True)
    # Live fit curve (replaces loaded when available)
    _add_avg_fit_curve(fig, d.get('live_gauss_fits'), '#44aaff', 'Live fit', faint=False)
    # Live bars
    _add_avg_bars(fig, d.get('live_hist_data'), d.get('n_accum_shots', 0))
    fig.update_layout(**_L, title='Avg Histogram', barmode='overlay',
                      xaxis=dict(title='Intensity', **_A), yaxis=dict(title='Density', **_A),
                      legend=dict(x=0.5, y=0.99, bgcolor='rgba(0,0,0,0.3)', font=dict(size=8)))
    return fig


def _add_avg_fit_curve(fig, fits, color, name, faint=False):
    if not fits or not isinstance(fits, list):
        return
    valid = [g['params'] for g in fits if isinstance(g, dict) and g.get('params') is not None]
    if not valid:
        return
    # Vectorized: stack all params into (N,6) array, compute all curves at once
    P = np.array(valid)  # (N, 6): mu1, sig1, w1, mu2, sig2, w2
    xmin = float((P[:, 0] - 4*P[:, 1]).min())
    xmax = float((P[:, 3] + 4*P[:, 4]).max())
    xf = np.linspace(xmin, xmax, 200)
    # Broadcast: xf(200,) vs P(N,6) → (N,200) for each Gaussian
    dx1 = (xf[None, :] - P[:, 0:1]) / P[:, 1:2]  # (N, 200)
    dx2 = (xf[None, :] - P[:, 3:4]) / P[:, 4:5]
    g1 = P[:, 2:3] / (P[:, 1:2] * np.sqrt(2*np.pi)) * np.exp(-0.5 * dx1**2)
    g2 = P[:, 5:6] / (P[:, 4:5] * np.sqrt(2*np.pi)) * np.exp(-0.5 * dx2**2)
    avg = (g1 + g2).mean(axis=0)
    op = 0.3 if faint else 0.8
    fig.add_trace(go.Scatter(x=xf, y=avg, mode='lines',
                              line=dict(color=color, width=1.5, dash='dot' if faint else 'solid'),
                              fill='tozeroy', fillcolor=f'rgba(136,136,136,{0.05 if faint else 0.1})',
                              name=name, opacity=op))


def _add_avg_bars(fig, hist_data, n_shots):
    if not hist_data or not isinstance(hist_data, list) or len(hist_data) == 0:
        return
    # Common x-axis across all sites, then interpolate each site's density
    all_c = np.concatenate([h['bin_centers'] for h in hist_data])
    centers = np.linspace(all_c.min(), all_c.max(), 50)
    avg = np.zeros(50)
    for h in hist_data:
        avg += np.interp(centers, h['bin_centers'], h['counts'], left=0, right=0)
    avg /= len(hist_data)
    bw = (centers[-1] - centers[0]) / (len(centers) - 1) * 0.85
    fig.add_trace(go.Bar(x=centers, y=avg, marker_color='#4488cc', opacity=0.8,
                         width=bw, name=f'Live ({n_shots})'))


# ---- Rep site histograms ----

def _figs_reps(d):
    sites = d.get('hist_rep_sites')
    if not sites:
        return [_waiting('Site Hist')] * 4
    labels = ['Best', 'Worst', 'Random', 'Random']
    figs = []
    for k in range(4):
        if k < len(sites):
            figs.append(_build_hist(d, sites[k], f'{labels[k]}: Site {sites[k]+1}'))
        else:
            figs.append(_waiting('Site Hist'))
    return figs


# ---- Single-site histogram (shared builder) ----

def _fig_site(d, idx):
    fig = _build_hist(d, idx, f'Site {idx+1} Histogram')
    info = []
    t = d.get('thresholds')
    if t is not None and idx < len(t):
        info.append(html.Div(f'Threshold: {t[idx]:.2f}'))
    fits = d.get('live_gauss_fits') or d.get('loaded_gauss_fits')
    if fits and isinstance(fits, list) and idx < len(fits):
        p = fits[idx].get('params') if isinstance(fits[idx], dict) else None
        if p is not None:
            info.extend([html.Div(f'mu_empty: {p[0]:.2f}'), html.Div(f'mu_atom: {p[3]:.2f}'),
                         html.Div(f'sig_empty: {p[1]:.2f}'), html.Div(f'sig_atom: {p[4]:.2f}')])
    inf = d.get('infidelities')
    if inf is not None and idx < len(inf):
        v = float(inf[idx])
        c = '#4c4' if v < 0.01 else '#cc4' if v < 0.05 else '#c44'
        info.append(html.Div(html.Span(f'Infidelity: {v:.2e}', style={'color': c, 'fontWeight': 'bold'})))
    rates = d.get('loading_rates')
    if rates is not None and idx < len(rates):
        info.append(html.Div(f'Loading: {rates[idx]:.1%}'))
    info.append(html.Div(f'Shots: {d.get("n_accum_shots", 0)}', style={'color': '#888'}))
    return fig, info


def _build_hist(d, idx, title):
    """Build site histogram: loaded fit (background) + live bars + live fit (foreground)."""
    fig = go.Figure()
    loaded_fits = d.get('loaded_gauss_fits')
    live_hist = d.get('live_hist_data')
    live_fits = d.get('live_gauss_fits')
    thresholds = d.get('thresholds')
    inf = d.get('infidelities')

    has_live = live_hist is not None and isinstance(live_hist, list) and idx < len(live_hist)
    has_loaded_f = loaded_fits is not None and isinstance(loaded_fits, list) and idx < len(loaded_fits)
    has_live_f = live_fits is not None and isinstance(live_fits, list) and idx < len(live_fits)

    if not has_live and not has_loaded_f and not has_live_f:
        return _waiting(title)

    # Determine x range from histogram data (not fit tails)
    xmin, xmax = 195, 210  # fallback
    if has_live:
        bc = live_hist[idx]['bin_centers']
        xmin, xmax = float(bc.min()), float(bc.max())
        pad = (xmax - xmin) * 0.05
        xmin -= pad
        xmax += pad
    elif has_loaded_f:
        p = loaded_fits[idx].get('params') if isinstance(loaded_fits[idx], dict) else None
        if p is not None:
            xmin, xmax = p[0] - 5*p[1], p[3] + 5*p[4]

    # Layer 1: Loaded fit curves (faint background) — only when no live fit
    if has_loaded_f and not has_live_f:
        p = loaded_fits[idx].get('params') if isinstance(loaded_fits[idx], dict) else None
        if p is not None:
            xf = np.linspace(xmin, xmax, 200)
            y1 = p[2]*norm.pdf(xf, p[0], p[1])
            y2 = p[5]*norm.pdf(xf, p[3], p[4])
            fig.add_trace(go.Scatter(x=xf, y=y1, mode='lines', line=dict(color='#44cc44', width=1.5, dash='dot'),
                                     fill='tozeroy', fillcolor='rgba(68,204,68,0.08)', name='Empty (loaded)', opacity=0.5))
            fig.add_trace(go.Scatter(x=xf, y=y2, mode='lines', line=dict(color='#cc44cc', width=1.5, dash='dot'),
                                     fill='tozeroy', fillcolor='rgba(204,68,204,0.08)', name='Atom (loaded)', opacity=0.5))

    # Layer 2: Live histogram bars
    if has_live:
        h = live_hist[idx]
        bw = np.diff(h['bin_centers']).mean() * 0.85 if len(h['bin_centers']) > 1 else 1
        fig.add_trace(go.Bar(x=h['bin_centers'], y=h['counts'], marker_color='#5588bb',
                             opacity=0.7, width=bw, name='Live'))

    # Layer 3: Live fit curves (solid, on top)
    if has_live_f:
        p = live_fits[idx].get('params') if isinstance(live_fits[idx], dict) else None
        if p is not None:
            xf = np.linspace(xmin, xmax, 200)
            y1 = p[2]*norm.pdf(xf, p[0], p[1])
            y2 = p[5]*norm.pdf(xf, p[3], p[4])
            fig.add_trace(go.Scatter(x=xf, y=y1, mode='lines', line=dict(color='#44cc44', width=2),
                                     name='Empty (live)'))
            fig.add_trace(go.Scatter(x=xf, y=y2, mode='lines', line=dict(color='#cc44cc', width=2),
                                     name='Atom (live)'))
            fig.add_trace(go.Scatter(x=xf, y=y1+y2, mode='lines', line=dict(color='white', width=1.5, dash='dot'),
                                     name='Sum'))

    # Threshold line
    if thresholds is not None and idx < len(thresholds):
        fig.add_vline(x=float(thresholds[idx]), line=dict(color='#ff4444', width=2, dash='dash'))

    # Infidelity badge (above legend, top-right corner)
    if inf is not None and idx < len(inf):
        v = float(inf[idx])
        c = '#4c4' if v < 0.01 else '#cc4' if v < 0.05 else '#c44'
        fig.add_annotation(text=f'Infid: {v:.1e}', xref='paper', yref='paper',
                           x=0.99, y=1.0, xanchor='right', yanchor='top',
                           showarrow=False, font=dict(size=10, color=c, family='monospace'),
                           bgcolor='rgba(20,20,40,0.8)', bordercolor=c)

    fig.update_layout(**_L, title=title, xaxis=dict(title='Intensity', **_A),
                      yaxis=dict(title='Density', **_A),
                      legend=dict(x=0.99, y=0.88, xanchor='right', yanchor='top',
                                  bgcolor='rgba(0,0,0,0.3)', font=dict(size=7)),
                      barmode='overlay')
    return fig
