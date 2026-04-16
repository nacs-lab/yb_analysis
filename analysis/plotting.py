"""Plotting functions for scan analysis.

Port of MATLAB's plotScan_and_fit_lorentzian_peak.m, plotSiteData_anyGeometry.m, etc.
Interactive plots via plotly; static fallback via matplotlib.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import plotly.graph_objects as go

# ---- MATLAB-like default style ----
_MATLAB_BLUE = '#0072BD'
_MATLAB_RED = '#D95319'
_MATLAB_YELLOW = '#EDB120'
_MATLAB_PURPLE = '#7E2F8E'
_MATLAB_GREEN = '#77AC30'
_MATLAB_COLORS = [_MATLAB_BLUE, _MATLAB_RED, _MATLAB_YELLOW, _MATLAB_PURPLE, _MATLAB_GREEN]

_FONT_FAMILY = 'STIX Two Text, STIXGeneral, Times New Roman, Georgia, serif'
_LEGEND_STYLE = dict(bordercolor='black', borderwidth=1, font=dict(size=10),
                     bgcolor='rgba(255,255,255,0.85)')

# Common plotly layout template
_PLOTLY_LAYOUT = dict(
    font=dict(family=_FONT_FAMILY, size=14),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=False, ticks='inside', mirror=True,
               showline=True, linewidth=1, linecolor='black'),
    yaxis=dict(showgrid=False, ticks='inside', mirror=True,
               showline=True, linewidth=1, linecolor='black'),
    hovermode='closest',
    width=780, height=500,
    margin=dict(l=70, r=30, t=60, b=60),
)


def apply_matlab_style():
    """Apply MATLAB-like rcParams globally for matplotlib fallback plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.prop_cycle': plt.cycler('color', _MATLAB_COLORS),
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'legend.fontsize': 10,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'figure.facecolor': 'white',
        'figure.dpi': 120,
    })


import re

# SI prefixes for auto-scaling
_SI_PREFIXES_LARGE = [
    (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
]
_SI_PREFIXES_SMALL = [
    (1e-3, 'm'), (1e-6, 'μ'), (1e-9, 'n'), (1e-12, 'p'),
]

# Patterns that identify frequency vs time axes
_FREQ_PATTERNS = re.compile(r'(?i)(freq|hz|detuning)')
_TIME_PATTERNS = re.compile(r'(?i)(time|delay|duration|pulse.?width|gap)')


def _auto_scale_axis(x, xlabel):
    """Pick the best SI prefix for the x-axis data and rewrite the label.

    Parameters
    ----------
    x : ndarray — raw x values
    xlabel : str — e.g. '616 EOM Frequency (Hz)' or 'Time (s)'

    Returns
    -------
    x_scaled : ndarray
    xlabel_new : str — label with updated unit, e.g. '616 EOM Frequency (MHz)'
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x, xlabel

    x_abs_max = np.nanmax(np.abs(x))
    if x_abs_max == 0:
        return x, xlabel

    is_freq = bool(_FREQ_PATTERNS.search(xlabel))
    is_time = bool(_TIME_PATTERNS.search(xlabel))

    if is_freq:
        # Frequency: scale large values down (Hz → kHz → MHz → GHz)
        base_unit = 'Hz'
        prefixes = _SI_PREFIXES_LARGE
    elif is_time:
        # Time: scale small values up (s → ms → μs → ns)
        base_unit = 's'
        prefixes = _SI_PREFIXES_SMALL
    else:
        # Unknown axis type — try both directions
        if x_abs_max >= 1e3:
            base_unit = _extract_unit(xlabel) or ''
            prefixes = _SI_PREFIXES_LARGE
        elif x_abs_max < 1e-1:
            base_unit = _extract_unit(xlabel) or ''
            prefixes = _SI_PREFIXES_SMALL
        else:
            return x, xlabel

    # Find the best prefix
    for threshold, prefix in prefixes:
        if x_abs_max >= threshold * 0.999:
            scale = 1.0 / threshold
            new_unit = f'{prefix}{base_unit}'
            xlabel_new = _replace_unit_in_label(xlabel, base_unit, new_unit)
            return x * scale, xlabel_new

    return x, xlabel


def _extract_unit(label):
    """Extract the unit from a label like 'Frequency (Hz)' → 'Hz'."""
    m = re.search(r'\(([^)]+)\)\s*$', label)
    return m.group(1).strip() if m else None


def _replace_unit_in_label(label, old_unit, new_unit):
    """Replace the unit in parentheses: 'Freq (Hz)' → 'Freq (MHz)'."""
    # Try replacing inside parentheses first
    pattern = re.compile(r'\(' + re.escape(old_unit) + r'\)')
    if pattern.search(label):
        return pattern.sub(f'({new_unit})', label)
    # If no parenthesized unit found, try bare unit at end
    if label.rstrip().endswith(old_unit):
        return label[:label.rfind(old_unit)] + new_unit
    # Fallback: append
    return f'{label} ({new_unit})'


def _short_data_path(data_path):
    """Extract a short label like '20260404\\161445' from a data path."""
    parts = os.path.normpath(data_path).replace('/', '\\').rstrip('\\').split('\\')
    # Find the date directory (YYYYMMDD) and take date + time
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            # Next part is usually data_YYYYMMDD_HHMMSS — extract just HHMMSS
            if i + 1 < len(parts):
                subdir = parts[i + 1]
                # Extract time from 'data_20260404_161445' → '161445'
                time_parts = subdir.split('_')
                time_str = time_parts[-1] if len(time_parts) >= 3 else subdir
                return f'{p}\\{time_str}'
            return p
    return '\\'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _build_fit_label(fit_result):
    """Build a legend label string from a fit result dict."""
    if 'fit_label' in fit_result:
        return fit_result['fit_label']
    if 'center' in fit_result and 'width' in fit_result:
        y0, A, x0, w = fit_result['params']
        model_name = getattr(fit_result.get('model'), '__name__', '')
        sign = '-' if 'dip' in model_name else '+'
        formula = f'y = y₀ {sign} A/(1 + (x - x₀)²/(w/2)²)'
        param_str = f'y₀={y0:.3g}, A={A:.3g}, x₀={x0:.7g}, w={w:.3g}'
        return f'{formula};<br>{param_str}'
    elif 'tau' in fit_result:
        a, tau, c = fit_result['params']
        return f'y = a·exp(-x/τ) + c;<br>a={a:.3g}, τ={tau:.3g}, c={c:.3g}'
    else:
        return f'Fit (R²={fit_result["r_squared"]:.3f})'


# ---- Interactive (plotly) plots ----

def plot_scan_interactive(scan_params, y_mean, y_sem=None, fit_result=None,
                           xlabel='Scan Parameter', ylabel='Probability',
                           title=None, scale=1.0, data_path=None):
    """Interactive 1D scan plot with hover, zoom, and pan (plotly).

    Parameters
    ----------
    scan_params : ndarray (nParams,)
    y_mean, y_sem : ndarray (nParams,)
    fit_result : dict from fitting.fit_lorentzian/fit_exponential, optional
    xlabel, ylabel, title : str
    scale : float — multiply x values for display
    data_path : str, optional — shown in legend

    Returns
    -------
    plotly Figure
    """
    x_raw = np.asarray(scan_params) * scale
    # Auto-scale x-axis units (e.g. Hz → MHz, s → μs)
    x, xlabel = _auto_scale_axis(x_raw, xlabel)

    fig = go.Figure()

    # Data trace — open circles with error bars
    data_label = _short_data_path(data_path) if data_path else 'Data'
    error_y = dict(type='data', array=y_sem, visible=True,
                   color=_MATLAB_BLUE, thickness=1.3) if y_sem is not None else None
    fig.add_trace(go.Scatter(
        x=x, y=y_mean, error_y=error_y,
        mode='markers', name=data_label,
        marker=dict(size=9, color='white',
                    line=dict(color=_MATLAB_BLUE, width=2)),
        hovertemplate=(
            'x = %{x:.5g}<br>'
            'y = %{y:.4f}<br>'
            + ('err = %{error_y.array:.4f}' if y_sem is not None else '')
            + '<extra></extra>'
        ),
    ))

    # Fit trace — apply same scale factor
    if fit_result is not None:
        x_fit_raw = fit_result['x_fit'] * scale
        scale_factor = x[1] / x_raw[1] if len(x) > 1 and x_raw[1] != 0 else 1.0
        x_fit = x_fit_raw * scale_factor
        fit_label = _build_fit_label(fit_result)
        fig.add_trace(go.Scatter(
            x=x_fit, y=fit_result['y_fit'],
            mode='lines', name=fit_label,
            line=dict(color=_MATLAB_RED, width=2.5),
            hovertemplate='x = %{x:.5g}<br>y_fit = %{y:.4f}<extra></extra>',
        ))
        if 'center' in fit_result:
            fig.add_vline(x=fit_result['center'] * scale * scale_factor,
                          line_dash='dash', line_color='gray', opacity=0.5)

    # Auto-place legend away from data
    legend_pos = _auto_legend_pos(y_mean, fit_result)
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>' if title else '', font=dict(size=16)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        **_PLOTLY_LAYOUT,
        legend=dict(**_LEGEND_STYLE, **legend_pos),
    )
    return fig


def _auto_legend_pos(y_mean, fit_result=None):
    """Pick legend corner that avoids the data.

    Checks which corner of the plot has the most 'empty space'
    by looking at where the data mean sits relative to the y-range.
    """
    y = np.asarray(y_mean)
    y_mid = (np.nanmax(y) + np.nanmin(y)) / 2

    # Check if data is mostly in top or bottom half
    top_heavy = np.nanmean(y) > y_mid

    # Check if interesting features (peak/dip) are on left or right
    if fit_result and 'center' in fit_result:
        x_range = np.nanmax(y) - np.nanmin(y)  # not used for x
        # center is in fit_result, check relative position
        center = fit_result['center']
        x_min, x_max = np.nanmin(y), np.nanmax(y)  # dummy
    # For left/right, check where the extreme values are
    extreme_idx = np.nanargmax(y) if top_heavy else np.nanargmin(y)
    left_heavy = extreme_idx < len(y) / 2

    # Place legend in the opposite corner
    if top_heavy and left_heavy:
        return dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom')
    elif top_heavy and not left_heavy:
        return dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
    elif not top_heavy and left_heavy:
        return dict(x=0.98, y=0.98, xanchor='right', yanchor='top')
    else:
        return dict(x=0.02, y=0.98, xanchor='left', yanchor='top')


def plot_rabi_interactive(x, y, yerr, x_fit, y_fit, fit_label,
                           xlabel='Time (s)', ylabel='Survival Rate',
                           title='Microwave Rabi Oscillations', data_path=None):
    """Interactive Rabi / damped-sine plot with auto-scaled x-axis."""
    # Auto-scale x-axis
    x_s, xlabel = _auto_scale_axis(np.asarray(x), xlabel)
    scale_factor = x_s[1] / x[1] if len(x) > 1 and x[1] != 0 else 1.0
    x_fit_s = np.asarray(x_fit) * scale_factor

    fig = go.Figure()
    data_label = _short_data_path(data_path) if data_path else 'Data'
    fig.add_trace(go.Scatter(
        x=x_s, y=y,
        error_y=dict(type='data', array=yerr, visible=True,
                     color=_MATLAB_BLUE, thickness=1.3),
        mode='markers', name=data_label,
        marker=dict(size=9, color='white',
                    line=dict(color=_MATLAB_BLUE, width=2)),
        hovertemplate='x = %{x:.5g}<br>y = %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_fit_s, y=y_fit, mode='lines', name=fit_label,
        line=dict(color=_MATLAB_RED, width=2.5),
        hovertemplate='x = %{x:.5g}<br>y_fit = %{y:.4f}<extra></extra>',
    ))
    legend_pos = _auto_legend_pos(y)
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=16)),
        xaxis_title=xlabel, yaxis_title=ylabel,
        **_PLOTLY_LAYOUT,
        legend=dict(**_LEGEND_STYLE, **legend_pos),
    )
    return fig


def heatmap_2d_interactive(scan_params, values, xlabel='Param 1', ylabel='Param 2',
                            title='', cmap='Viridis'):
    """Interactive 2D heatmap for two-parameter scans (plotly)."""
    v1 = np.unique(scan_params[:, 0])
    v2 = np.unique(scan_params[:, 1])
    n1, n2 = len(v1), len(v2)

    grid = np.full((n2, n1), np.nan)
    for i in range(len(scan_params)):
        i1 = np.argmin(np.abs(v1 - scan_params[i, 0]))
        i2 = np.argmin(np.abs(v2 - scan_params[i, 1]))
        grid[i2, i1] = values[i]

    fig = go.Figure(go.Heatmap(
        z=grid, x=v1, y=v2, colorscale=cmap,
        hovertemplate=f'{xlabel}=%{{x:.5g}}<br>{ylabel}=%{{y:.5g}}<br>value=%{{z:.4f}}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=16)),
        xaxis_title=xlabel, yaxis_title=ylabel,
        **_PLOTLY_LAYOUT,
    )
    return fig


def save_static(fig_plotly, path, width=800, height=500, scale=2):
    """Save a plotly figure to a static image file.

    Tries kaleido first, falls back to orca, raises if neither available.
    """
    fig_plotly.write_image(path, width=width, height=height, scale=scale)


# ---- Static (matplotlib) plots ----

def plot_scan(scan_params, y_mean, y_sem=None, fit_result=None,
              xlabel='Scan Parameter', ylabel='Probability', title=None,
              scale=1.0, ax=None, data_path=None, **kwargs):
    """Plot 1D scan data with error bars and optional fit curve (MATLAB style).

    Parameters
    ----------
    scan_params : ndarray (nParams,)
    y_mean, y_sem : ndarray (nParams,)
    fit_result : dict from fitting.fit_lorentzian/fit_exponential, optional
    xlabel, ylabel, title : str
    scale : float — multiply x values for display (e.g., 1e6 for MHz)
    ax : matplotlib Axes, optional
    data_path : str, optional — path to scan data directory (shown in legend)

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    x_raw = scan_params * scale
    # Auto-scale x-axis units
    x, xlabel = _auto_scale_axis(x_raw, xlabel)

    # Data: open circles with error bars (MATLAB style)
    data_label = _short_data_path(data_path) if data_path else 'Data'
    ax.errorbar(x, y_mean, yerr=y_sem, fmt='o', markersize=7,
                markerfacecolor='none', markeredgecolor=_MATLAB_BLUE,
                markeredgewidth=1.5, color=_MATLAB_BLUE,
                capsize=4, capthick=1.2, linewidth=1.2, elinewidth=1.2,
                label=data_label, **kwargs)

    if fit_result is not None:
        x_fit_raw = fit_result['x_fit'] * scale
        sf = x[1] / x_raw[1] if len(x) > 1 and x_raw[1] != 0 else 1.0
        x_fit = x_fit_raw * sf

        # Build fit label with formula and parameters
        fit_label = _build_fit_label(fit_result).replace('<br>', '\n')

        ax.plot(x_fit, fit_result['y_fit'], '-', linewidth=2.0,
                color=_MATLAB_RED, label=fit_label)

        # Center line for Lorentzian
        if 'center' in fit_result:
            ax.axvline(fit_result['center'] * scale * sf, color='gray',
                       linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize=9, fancybox=False, edgecolor='black')
    ax.tick_params(direction='in', top=True, right=True)
    return fig, ax


def plot_scan_site_resolved(scan_params, prob_sr, sem_sr=None, fits=None,
                             sites_per_page=20, scale=1.0, xlabel='Param',
                             ylabel='Prob', title_prefix=''):
    """Multi-page subplot grid of per-site probability vs scan parameter.

    Parameters
    ----------
    scan_params : ndarray (nParams,)
    prob_sr : ndarray (nSites, nParams)
    sem_sr : ndarray (nSites, nParams), optional
    fits : list of fit dicts (one per site), optional
    sites_per_page : int

    Returns
    -------
    list of (fig, axes) tuples
    """
    n_sites = prob_sr.shape[0]
    x = scan_params * scale
    pages = []

    for page_start in range(0, n_sites, sites_per_page):
        page_end = min(page_start + sites_per_page, n_sites)
        n = page_end - page_start
        ncols = 5
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), squeeze=False)
        fig.suptitle(f'{title_prefix} Sites {page_start+1}-{page_end}', fontsize=14)

        for i in range(n):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            s = page_start + i
            err = sem_sr[s] if sem_sr is not None else None
            ax.errorbar(x, prob_sr[s], yerr=err, fmt='o', markersize=3, capsize=2)

            if fits is not None and fits[s] is not None:
                ax.plot(fits[s]['x_fit'] * scale, fits[s]['y_fit'], '-', color='C1', linewidth=1)

            ax.set_title(f'Site {s+1}', fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=7)

        # Hide unused subplots
        for i in range(n, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].set_visible(False)

        fig.tight_layout()
        pages.append((fig, axes))

    return pages


def plot_site_data(data, grid_locations, data_sem=None, site_idx=None,
                   title='', cmap='RdYlGn', clim=None, marker_size=80):
    """3-panel spatial visualization: 1D line, 2D scatter, histogram.

    Port of MATLAB's plotSiteData_anyGeometry.m.

    Parameters
    ----------
    data : ndarray (nSites,) — values per site
    grid_locations : ndarray (nSites, 2) — [y, x] per site
    data_sem : ndarray, optional
    site_idx : ndarray, optional — subset of site indices
    title : str
    cmap : str — colormap name
    clim : (vmin, vmax), optional

    Returns
    -------
    fig, (ax_line, ax_spatial, ax_hist)
    """
    if site_idx is not None:
        data = data[site_idx]
        grid_locations = grid_locations[site_idx]
        if data_sem is not None:
            data_sem = data_sem[site_idx]

    n = len(data)
    valid = np.isfinite(data)

    fig, (ax_line, ax_spatial, ax_hist) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14)

    # Panel 1: 1D line plot
    sites = np.arange(1, n + 1)
    ax_line.errorbar(sites[valid], data[valid],
                     yerr=data_sem[valid] if data_sem is not None else None,
                     fmt='o-', markersize=4, capsize=2)
    ax_line.set_xlabel('Site')
    ax_line.set_ylabel('Value')
    ax_line.grid(True, alpha=0.3)

    # Panel 2: 2D spatial scatter
    if clim is None:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    else:
        vmin, vmax = clim
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = ax_spatial.scatter(grid_locations[valid, 1], grid_locations[valid, 0],
                            c=data[valid], cmap=cmap, norm=norm,
                            s=marker_size, edgecolors='white', linewidths=0.5)
    ax_spatial.set_aspect('equal')
    ax_spatial.invert_yaxis()
    ax_spatial.set_xlabel('X (px)')
    ax_spatial.set_ylabel('Y (px)')
    plt.colorbar(sc, ax=ax_spatial)

    # Panel 3: histogram + Gaussian fit
    valid_data = data[valid]
    ax_hist.hist(valid_data, bins=min(30, max(5, n // 3)), edgecolor='black', alpha=0.7)
    if len(valid_data) > 3:
        mu, std = valid_data.mean(), valid_data.std()
        ax_hist.axvline(mu, color='red', linestyle='--', label=f'μ={mu:.3g}')
        ax_hist.set_title(f'μ={mu:.3g}, σ={std:.3g}')
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Count')
    ax_hist.legend()

    fig.tight_layout()
    return fig, (ax_line, ax_spatial, ax_hist)


def heatmap_2d_scan(scan_params, values, xlabel='Param 1', ylabel='Param 2',
                     title='', cmap='viridis', clim=None):
    """2D heatmap for two-parameter scan.

    Parameters
    ----------
    scan_params : ndarray (nParams, 2) — [v1, v2] pairs
    values : ndarray (nParams,)

    Returns
    -------
    fig, ax
    """
    v1 = np.unique(scan_params[:, 0])
    v2 = np.unique(scan_params[:, 1])
    n1, n2 = len(v1), len(v2)

    grid = np.full((n2, n1), np.nan)
    for i in range(len(scan_params)):
        i1 = np.argmin(np.abs(v1 - scan_params[i, 0]))
        i2 = np.argmin(np.abs(v2 - scan_params[i, 1]))
        grid[i2, i1] = values[i]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   extent=[v1.min(), v1.max(), v2.min(), v2.max()],
                   cmap=cmap, vmin=clim[0] if clim else None,
                   vmax=clim[1] if clim else None)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax
