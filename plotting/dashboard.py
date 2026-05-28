"""Yb Tweezer Dashboard — Plotly Dash, single-callback architecture.

Layout:
  Row 1: [Tweezer Array]  [Atom Intensities]
  Row 2: [Atom Intensities (wide)]  [Loading Rate (live)]
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
        img2 = d.get('cur_image2')
        if img2 is not None:
            uri2, vlo2, vhi2 = _img_to_data_uri(np.asarray(img2, dtype=np.int16))
            d['_img2_data_uri'] = uri2
            d['_img2_shape'] = img2.shape
            d['_img2_vlo'] = vlo2
            d['_img2_vhi'] = vhi2
            d.pop('cur_image2', None)
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

    def _terminate_proc(self, tag=''):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1)
            logger.info('Dashboard process stopped%s', f' ({tag})' if tag else '')
        self._proc = None

    def close(self):
        self._terminate_proc()
        for p in (_DATA_FILE, _DATA_FILE + '.0', _DATA_FILE + '.1'):
            try:
                os.remove(p)
            except OSError:
                pass

    def restart(self):
        """Kill and immediately respawn the Dash subprocess.

        Keeps the shared data files in place so the new subprocess renders
        the current frame straight away (no 'waiting for data' gap).
        """
        self._terminate_proc(tag='restart')
        # Brief pause to let the OS release port 8050 before the new
        # subprocess tries to bind it. Without this the child can die
        # silently with WinError 10048.
        time.sleep(0.5)
        self.start()


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
<style>
/* Sleek iOS-style toggle switch (used for the colorbar autoscale control). */
input.yb-switch {
    -webkit-appearance: none; appearance: none; margin: 0 6px 0 0;
    position: relative; width: 30px; height: 16px; flex: none;
    background: #3a3a52; border-radius: 8px; cursor: pointer;
    transition: background .15s ease; outline: none;
}
input.yb-switch::before {
    content: ''; position: absolute; top: 2px; left: 2px;
    width: 12px; height: 12px; border-radius: 50%;
    background: #d8d8e0; transition: transform .15s ease;
}
input.yb-switch:checked { background: #2a7fff; }
input.yb-switch:checked::before { transform: translateX(14px); background: #fff; }
</style>
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
        # Row 1: image1 | image2 | scan curve — three equal-width 670px panels
        # (per-shot live data: gets the most vertical real estate)
        _row([
            _graph('array', 670),
            _graph('array2', 670),
            # Scan panel; colorbar-scale toggle overlaid INSIDE the panel (top-right).
            html.Div(style={'flex': '1', 'minWidth': '0', 'position': 'relative'}, children=[
                dcc.Graph(id='scan', figure=_waiting(''), style={'height': '670px'},
                          config={'displayModeBar': False}),
                # Toggle switch: on → autoscale colorbar to data; off → fixed 0–1.
                html.Div(style={'position': 'absolute', 'top': '9px', 'right': '70px',
                                'zIndex': '5', 'display': 'flex', 'alignItems': 'center'},
                    children=[
                        dcc.Checklist(id='cbar-scale',
                            options=[{'label': 'Autoscale', 'value': 'auto'}],
                            value=[], inline=True, inputClassName='yb-switch',
                            style={'fontSize': '11px', 'color': '#ffffff'},
                            labelStyle={'display': 'flex', 'alignItems': 'center',
                                        'cursor': 'pointer', 'margin': '0', 'color': '#ffffff'}),
                    ]),
            ]),
        ]),
        # Row 2: Atom Intensities (wide) + live Loading Rate panel
        # (per-shot data; intens gets 3x the width since it scales with #sites)
        _row([_graph('intens', 320, flex=3), _graph('loadlive', 320, flex=1)]),
        # Row 3: Avg Histogram + Rep site histograms (refit every 200 shots)
        _row([_graph('avghist', 240)] + [_graph(f'rep{i}', 240) for i in range(4)]),
        # Row 4: Loading | Infidelities | Site selector + Site Hist | Grid Shift
        # Tall enough for the 2-D site maps to read as roughly square on large arrays.
        _row([
            _graph('load', 600),
            _graph('infid', 600),
            html.Div(style={'flex': '1', 'minWidth': '0', 'display': 'flex', 'gap': '8px'}, children=[
                # Left: dropdown + parameters
                html.Div(style={'width': '140px', 'flexShrink': '0'}, children=[
                    html.Label('Site:', style={'fontSize': '12px'}),
                    dcc.Dropdown(id='site-dd', options=[], value=1, clearable=False,
                                 style={'backgroundColor': '#2b2b4a', 'color': '#222', 'marginBottom': '8px'}),
                    html.Div(id='site-info', style={'fontSize': '11px', 'color': '#bbb',
                        'lineHeight': '1.6'}),
                    # Slider controls marker size for the site-resolved
                    # scatter plots (load / infid) so they read well at
                    # any array spacing.
                    html.Div(style={'marginTop': '14px', 'paddingTop': '10px',
                        'borderTop': '1px solid #333'}, children=[
                        html.Label('Marker size:', style={'fontSize': '11px', 'color': '#bbb'}),
                        dcc.Slider(id='marker-size', min=2, max=40, step=1, value=12,
                            marks={2: {'label': '2', 'style': {'fontSize': '9px', 'color': '#888'}},
                                   20: {'label': '20', 'style': {'fontSize': '9px', 'color': '#888'}},
                                   40: {'label': '40', 'style': {'fontSize': '9px', 'color': '#888'}}},
                            tooltip={'placement': 'top', 'always_visible': False}),
                    ]),
                ]),
                # Right: histogram
                _graph('site', 590),
            ]),
            _graph('shift', 600),
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
    outputs = ([Output('array', 'figure'), Output('array2', 'figure'),
                 Output('intens', 'figure'), Output('loadlive', 'figure'),
                 Output('load', 'figure'), Output('infid', 'figure'),
                 Output('shift', 'figure'), Output('scan', 'figure'),
                 Output('avghist', 'figure')]
               + [Output(f'rep{i}', 'figure') for i in range(4)]
               + [Output('site-dd', 'options'), Output('debug-pre', 'children')])

    @app.callback(outputs, [Input('tick', 'n_intervals'),
                            Input('marker-size', 'value'),
                            Input('cbar-scale', 'value')])
    def refresh(_n, marker_size, cbar_toggle):
        # Guard against the slider returning None during first render.
        marker_size = int(marker_size) if marker_size else 12
        # Checklist returns a list; 'auto' present → autoscale, else fixed 0–1.
        cbar_scale = 'auto' if (cbar_toggle and 'auto' in cbar_toggle) else '01'
        d = _read_data()
        debug_lines = []

        if d is None:
            debug_lines.append('No data yet (pickle file not found)')
            empty = [_waiting(t) for t in ['Tweezer Array (img 1)', 'Tweezer Array (img 2)',
                     'Intensities', 'Loading Rate', 'Loading', 'Infidelities', 'Grid Shift',
                     'Scan Curve', 'Avg Histogram']]
            return empty + [_waiting('Site Hist')]*4 + [[], '\n'.join(debug_lines)]

        try:
            has_img = d.get('_img_data_uri') is not None
            has_img2 = d.get('_img2_data_uri') is not None
            n = d.get('num_sites', 0)
            v = d.get('hist_version', 0)
            n_acc = d.get('n_accum_shots', 0)

            num_images = int(d.get('num_images', 1) or 1)
            img2_no_data_msg = ('No image 2 (NumImages = 1)' if num_images < 2
                                else 'Waiting for data...')
            img2_grid_key = ('grid_locations_img2' if d.get('is_two_array')
                             else 'grid_locations')
            # Dummy mode: live panels (image, intensities, loading-rate trace)
            # still reflect the current frame, but cumulative panels carry
            # over stale values from the last real scan. Blank those out and
            # label them so the user isn't misled by frozen data.
            is_dummy = bool(d.get('_dummy_mode'))
            dummy_msg = 'Dummy mode'

            def _stale(title, builder):
                return _waiting(title, dummy_msg) if is_dummy else builder()

            figs = [
                _fig_array(d) if has_img else _waiting('Tweezer Array (img 1)'),
                _fig_array(d, img_key='_img2_data_uri', shape_key='_img2_shape',
                           vlo_key='_img2_vlo', vhi_key='_img2_vhi',
                           logicals_key='logicals2', grid_key=img2_grid_key,
                           title='Tweezer Array (img 2)')
                    if has_img2 else _waiting('Tweezer Array (img 2)', img2_no_data_msg),
                _fig_intens(d),
                _fig_loading_live(d),
                _stale('Loading Rates', lambda: _fig_loading(d, marker_size=marker_size)),
                _stale('Infidelities', lambda: _fig_infid(d, marker_size=marker_size)),
                _stale('Grid Shift', lambda: _fig_shift(d)),
                _stale('Scan Curve', lambda: _fig_scan_curve(d, cbar_scale=cbar_scale)),
                _stale('Avg Histogram', lambda: _fig_avghist(d)),
            ]

            reps = ([_waiting('Site Hist', dummy_msg)] * 4
                    if is_dummy else _figs_reps(d))
            opts = [{'label': f'Site {i+1}', 'value': i+1} for i in range(n)]

            lh = d.get('live_hist_data')
            lf = d.get('live_gauss_fits')
            ldf = d.get('loaded_gauss_fits')
            debug_lines.append(f'sites={n} accum={n_acc} hist_v={v}')
            debug_lines.append(f'live_hist: {"list["+str(len(lh))+"]" if isinstance(lh, list) else type(lh).__name__}')
            debug_lines.append(f'live_fits: {"list["+str(len(lf))+"]" if isinstance(lf, list) else type(lf).__name__}')
            debug_lines.append(f'loaded_fits: {"list["+str(len(ldf))+"]" if isinstance(ldf, list) else type(ldf).__name__}')
            debug_lines.append(f'img={has_img} img2={has_img2} rep_sites={d.get("hist_rep_sites")}')

            return figs + reps + [opts, '\n'.join(debug_lines)]

        except Exception:
            tb = traceback.format_exc()
            logging.error('Dashboard render error:\n%s', tb)
            return [no_update] * 15

    @app.callback([Output('site', 'figure'), Output('site-info', 'children')],
                  [Input('site-dd', 'value'), Input('tick', 'n_intervals')])
    def site_hist(val, _n):
        d = _read_data()
        if d is None or val is None:
            return _waiting('Site Histogram'), ''
        if d.get('_dummy_mode'):
            return _waiting('Site Histogram', 'Dummy mode'), ''
        return _fig_site(d, int(val) - 1)

    # Click on loading-rate or infidelity 2D plot → select site in dropdown
    # is handled entirely in JavaScript (index_string) via plotly_click +
    # React fiber setProps. This bypasses Dash's callback system which can
    # lose clickData when the refresh callback replaces figures every 3s.

    return app


# ---- Helpers ----

def _row(children):
    return html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}, children=children)

def _graph(id, h, flex=1):
    # Set initial "waiting" figure so Plotly has a uirevision baseline.
    # Without this, Plotly may not re-render when the callback first returns
    # a figure with uirevision='live' (no prior value to compare against).
    return dcc.Graph(id=id, figure=_waiting(''),
                     style={'flex': f'{flex}', 'minWidth': '0', 'height': f'{h}px'},
                     config={'displayModeBar': False})

def _waiting(title, message='Waiting for data...'):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref='paper', yref='paper',
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


def _fig_array(d, img_key='_img_data_uri', shape_key='_img_shape',
               vlo_key='_img_vlo', vhi_key='_img_vhi',
               logicals_key='logicals', grid_key='grid_locations',
               title='Tweezer Array (img 1)'):
    data_uri = d.get(img_key)
    shape = d.get(shape_key)
    if data_uri is None or shape is None:
        return _waiting(title)
    H, W = shape
    fig = go.Figure()
    fig.add_layout_image(
        source=data_uri, xref='x', yref='y',
        x=0, y=0, sizex=W, sizey=H,
        sizing='stretch', layer='below',
    )
    # Colorbar via invisible scatter + autorange anchor at image corners
    vlo = d.get(vlo_key, 0)
    vhi = d.get(vhi_key, 255)
    fig.add_trace(go.Scatter(
        x=[0, W, 0, W], y=[0, 0, H, H], mode='markers',
        marker=dict(size=0.1, opacity=0, color=[vlo, vhi, vlo, vhi],
                    colorscale='gray', cmin=vlo, cmax=vhi, showscale=True,
                    colorbar=dict(title='Counts', len=0.9)),
        hoverinfo='skip', showlegend=False))

    # Site markers as lightweight scatter overlay
    grid = d.get(grid_key)
    logicals = d.get(logicals_key)
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

    fig.update_layout(**_L, title=title,
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
    # Marker size shrinks as the array grows so dots don't overlap on dense
    # arrays but stay readable for small ones (~13px @ n<=140, ~8px @ n=225).
    cur_size = float(np.clip(1800.0 / n, 6, 13))
    thr_size = max(4.0, cur_size - 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sites, y=t.tolist(), mode='markers', name='Threshold',
                              marker=dict(size=thr_size, color='#777', symbol='circle', line=dict(width=1, color='#999'))))
    ymin, ymax = float(t.min()), float(t.max())
    ci = d.get('cur_intensities')
    if ci is not None:
        logicals = d.get('logicals')
        colors = ['#0c6' if (logicals is not None and i < len(logicals) and logicals[i]) else '#e44' for i in range(n)]
        fig.add_trace(go.Scatter(x=sites, y=ci.tolist(), mode='markers', name='Current',
                                  marker=dict(size=cur_size, color=colors, symbol='circle', line=dict(width=1, color='white'))))
        ymin = min(ymin, float(ci.min()))
        ymax = max(ymax, float(ci.max()))
    # Mean line + 68% (±1σ) band for loaded / empty sites + distance annotation
    if ci is not None and logicals is not None:
        mask = np.array(logicals[:n], dtype=bool) if len(logicals) >= n else np.zeros(n, dtype=bool)

        def _band(values, color, fill, label, yanchor):
            mu = float(values.mean())
            sd = float(values.std())
            # ±1σ band ≈ central 68% of a normal distribution
            fig.add_shape(type='rect', xref='paper', x0=0, x1=1, y0=mu-sd, y1=mu+sd,
                          fillcolor=fill, line=dict(width=0), layer='below')
            fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=mu, y1=mu,
                          line=dict(color=color, width=1.5, dash='dash'))
            fig.add_annotation(text=f'{label}: {mu:.1f} ± {sd:.1f}', xref='paper', y=mu,
                               x=0.99, showarrow=False, xanchor='right', yanchor=yanchor,
                               font=dict(color=color, size=10), bgcolor='rgba(20,20,40,0.6)')
            return mu, sd

        if mask.any():
            mu_loaded, sd_loaded = _band(ci[mask], '#0c6', 'rgba(0,204,102,0.12)', 'Loaded', 'bottom')
            ymin = min(ymin, mu_loaded - sd_loaded)
            ymax = max(ymax, mu_loaded + sd_loaded)
        else:
            mu_loaded = None
        if (~mask).any():
            mu_empty, sd_empty = _band(ci[~mask], '#e44', 'rgba(238,68,68,0.12)', 'Empty', 'top')
            ymin = min(ymin, mu_empty - sd_empty)
            ymax = max(ymax, mu_empty + sd_empty)
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


def _fig_loading_live(d):
    hist = d.get('loading_history')
    if hist is None or len(hist) == 0:
        return _waiting('Loading Rate')
    hist = np.asarray(hist, dtype=float)
    logicals = d.get('logicals')
    cur = float(np.asarray(logicals).mean()) if logicals is not None and len(logicals) > 0 else None
    # Average over the displayed history window (always populated, unlike
    # loading_rates which only refreshes every UPDATE_LOADING_INTERVAL shots).
    avg = float(hist.mean())

    n = len(hist)
    x = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hist.tolist(), mode='lines+markers',
                              line=dict(color='#0c6', width=1.5),
                              marker=dict(size=4, color='#0c6'),
                              name='Per-shot', hoverinfo='y'))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=avg, y1=avg,
                  line=dict(color='#ffdd44', width=1.5, dash='dash'))
    fig.add_annotation(text=f'Avg: {avg:.1%}', xref='paper', y=avg,
                       x=0.99, showarrow=False, xanchor='right', yanchor='bottom',
                       font=dict(color='#ffdd44', size=10))
    if cur is not None:
        fig.add_annotation(text=f'Current: {cur:.1%}', xref='paper', yref='paper',
                           x=0.5, y=1.0, showarrow=False,
                           font=dict(size=18, color='#0c6', family='monospace'),
                           bgcolor='rgba(20,20,40,0.8)')
    fig.update_layout(**_L, title='Loading Rate (last 100)',
                      xaxis=dict(title='Shot # (oldest → latest)', **_A),
                      yaxis=dict(title='Fraction loaded', autorange=True,
                                 tickformat='.0%', **_A),
                      showlegend=False)
    return fig


def _fig_loading(d, marker_size=12):
    grid, rates = d.get('grid_locations'), d.get('loading_rates')
    if grid is None or rates is None or len(grid) == 0:
        return _waiting('Loading Rates')
    n = len(grid)
    sz = marker_size
    if n < 100:
        mode = 'markers+text'
        text = [f'{r:.0%}' for r in rates]
        tfont = dict(size=7, color='black')
    else:
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


def _fig_infid(d, marker_size=12):
    grid, inf = d.get('grid_locations'), d.get('infidelities')
    if grid is None or inf is None or len(grid) == 0:
        return _waiting('Infidelities')
    n = len(grid)
    log_inf = np.log10(np.clip(inf, 1e-6, 1.0))
    sz = marker_size
    if n < 100:
        mode = 'markers+text'
        text = [f'{v:.0e}' for v in inf]
        tfont = dict(size=6, color='white')
    else:
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


def _fig_scan_curve(d, cbar_scale='01'):
    sc = d.get('scan_curve')
    if sc is None or sc.get('mode') == 'undefined':
        return _waiting('Scan Curve')

    # --- 2-D heatmap ---
    if sc.get('ndim', 1) >= 2:
        return _fig_scan_2d(d, sc, cbar_scale=cbar_scale)

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
    if mode == 'survival':
        y_label = 'Survival'
    elif mode == 'rearrangement':
        y_label = 'Rearrangement Success (mean of logic2)'
    else:
        y_label = 'Loading Rate'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_disp, y=y, error_y=dict(type='data', array=err, visible=True, thickness=1.5),
        mode='markers', marker=dict(size=6, color='#44aaff'),
        hoverinfo='text', hovertext=[f'{x_label}={xi:.4g}, {y_label}={yi:.3f}+/-{ei:.3f} (n={ni})'
                                      for xi, yi, ei, ni in zip(x_disp, y, err, n_reps)]))
    title_text = _scan_title(f'{scan_name} ({int(n_reps.mean())} reps/pt)',
                             d.get('scan_filename'))
    fig.update_layout(**_L, title=title_text,
                      xaxis=dict(title=x_label, **_A),
                      yaxis=dict(title=y_label, range=[-0.05, 1.05], **_A))
    return fig


def _scan_title(main, fname):
    """Scan-panel title with the run/folder name shown inline and clearly
    (bright, larger) next to the main title rather than as a faint subtitle."""
    if not fname:
        return main
    run = fname[:-3] if fname.endswith('.h5') else fname  # strip .h5 → run folder
    return f'{main}  <span style="font-size:14px;color:#7cc4ff">— {run}</span>'


def _fmt_tick(v):
    """Compact axis label: SI suffix for big magnitudes, else general format."""
    av = abs(float(v))
    if av >= 1e9:
        return f'{v/1e9:g}G'
    if av >= 1e6:
        return f'{v/1e6:g}M'
    if av >= 1e3:
        return f'{v/1e3:g}k'
    return f'{v:g}'


def _tickset(vals):
    """Tick indices + labels for an equal-step (index) axis, thinned to ~12
    ticks max so labels don't crowd on long scans."""
    n = len(vals)
    step = max(1, int(np.ceil(n / 12)))
    idx = list(range(0, n, step))
    return idx, [_fmt_tick(vals[i]) for i in idx]


def _fig_scan_2d(d, sc, cbar_scale='01'):
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

    # Colorbar z-range: 'auto' lets Plotly autoscale to the data, '01' pins 0–1.
    if cbar_scale == 'auto':
        zmin = zmax = None
    else:
        zmin, zmax = 0, 1

    # Plot against equal-step indices so EVERY cell is the same size even when
    # the scan values are unevenly spaced; the real values are restored as tick
    # labels and ride in customdata for the hover (along with reps + error).
    nx, ny = len(x_disp), len(y_vals)
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    xv = np.asarray(x_disp, dtype=float)
    yv = np.asarray(y_vals, dtype=float)
    Xv = np.broadcast_to(xv.reshape(1, nx), (ny, nx))   # actual x per cell
    Yv = np.broadcast_to(yv.reshape(ny, 1), (ny, nx))   # actual y per cell

    sem = sc.get('sem')
    if sem is not None:
        # customdata: [x_val, y_val, reps, error] per cell
        customdata = np.dstack([Xv, Yv, n_reps, sem])
        hovertemplate = (f'{x_name}=%{{customdata[0]:.4g}}<br>'
                         f'{y_name}=%{{customdata[1]:.4g}}<br>'
                         f'{y_label}=%{{z:.3f}} ± %{{customdata[3]:.3f}}<br>'
                         f'reps=%{{customdata[2]:d}}<extra></extra>')
    else:
        customdata = np.dstack([Xv, Yv, n_reps])
        hovertemplate = (f'{x_name}=%{{customdata[0]:.4g}}<br>'
                         f'{y_name}=%{{customdata[1]:.4g}}<br>'
                         f'{y_label}=%{{z:.3f}}<br>reps=%{{customdata[2]:d}}<extra></extra>')

    fig = go.Figure(go.Heatmap(
        z=z, x=x_idx, y=y_idx,
        colorscale='Viridis', zmin=zmin, zmax=zmax,
        colorbar=dict(title=y_label, len=0.9),
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))

    # Red box around every cell updated in the latest batch (the cells
    # currently being scanned). On the index grid every cell is unit-sized.
    cur = sc.get('current') or []
    if isinstance(cur, dict):       # backward-compat: old single-cell format
        cur = [cur]
    for cell in cur:
        xi, yi = cell.get('x_idx'), cell.get('y_idx')
        if (xi is None or yi is None
                or not (0 <= xi < nx) or not (0 <= yi < ny)):
            continue
        fig.add_shape(type='rect', xref='x', yref='y',
                      x0=xi-0.5, x1=xi+0.5, y0=yi-0.5, y1=yi+0.5,
                      line=dict(color='#ff0000', width=3), fillcolor='rgba(0,0,0,0)',
                      layer='above')

    xtv, xtt = _tickset(xv)
    ytv, ytt = _tickset(yv)
    title_text = _scan_title(f'{scan_name} ({avg_reps} reps/pt)', d.get('scan_filename'))
    fig.update_layout(**_L, title=title_text,
                      xaxis=dict(title=x_name, tickmode='array',
                                 tickvals=xtv, ticktext=xtt, **_A),
                      yaxis=dict(title=y_name, tickmode='array',
                                 tickvals=ytv, ticktext=ytt, **_A))
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
