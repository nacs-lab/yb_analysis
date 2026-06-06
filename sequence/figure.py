"""Build the Plotly figure for the dashboard's Sequence tab.

Mirrors SeqPlotter's plot: per-channel ``value`` vs ``time (ms)`` lines+markers,
with channels whose values reach the frequency range routed to a secondary
y-axis. ``pulse_id`` is carried as ``customdata`` so a clicked point can later be
traced to its source (backtrace, a fast-follow).

Kept separate from ``dashboard.py`` so it is unit-testable without Flask. Numpy
arrays are converted to plain Python lists here so the JSON is directly
Plotly.js-friendly (no typed-array ``bdata`` round-trip needed).
"""

import colorsys

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Card palette (matches dashboard.css --bg-card / --bg-input / --text).
_PLOT_BG = "#0b0e13"
_FONT = "#d8dee9"
_HIGHLIGHT = "#f0c000"
_GRID = "rgba(255,255,255,0.08)"       # subtle gridlines on the dark bg
_AXIS_LINE = "rgba(255,255,255,0.22)"  # axis / zero lines + rangeslider border


def _distinct_colors(n, sat=0.72, light=0.62):
    """``n`` bright, well-separated trace colors for the near-black plot bg.

    Plotly's default palette washes out and repeats once a sequence has ~20
    channels. Golden-angle hue stepping spreads consecutive hues as far apart
    as possible (so neighbouring legend entries never look alike), and the
    fixed high saturation + lightness keeps every colour clearly visible on the
    dark background -- no pale/low-contrast entries.
    """
    golden = 0.6180339887498949   # golden-ratio conjugate hue step
    h = 0.137                     # start offset (away from pure red)
    out = []
    for _ in range(max(int(n), 1)):
        r, g, b = colorsys.hls_to_rgb(h % 1.0, light, sat)
        out.append("#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255)))
        h += golden
    return out


def build_sequence_figure(seq, channel_names, title=None):
    """Return a ``go.Figure`` for ``channel_names`` of one :class:`Sequence`.

    Unknown channel names are skipped. An empty selection yields an
    axes-only figure (the channel picker drives what's shown).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 0 is the click-selection highlight (mirrors SeqPlotter); the JS
    # moves its single point on click.
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="markers", name="Selected",
                   marker=dict(opacity=0.75, color=_HIGHLIGHT, size=12),
                   showlegend=False),
        secondary_y=False,
    )

    # Stable per-channel colour: keyed to the channel's index in the sequence's
    # FULL channel list, so a given channel keeps its colour regardless of which
    # others are currently selected.
    all_names = list(seq.channel_names) if seq is not None else []
    palette = _distinct_colors(max(len(all_names), 1))
    color_of = {nm: palette[i % len(palette)] for i, nm in enumerate(all_names)}

    for name in (channel_names or []):
        try:
            ch = seq.channel(name)
        except KeyError:
            continue
        color = color_of.get(name, palette[0])
        fig.add_trace(
            go.Scatter(
                x=ch.t_ms.tolist(),
                y=ch.v.tolist(),
                customdata=[int(p) for p in ch.pid],  # pulse_id (for backtrace)
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=5),
                hovertemplate="%{x:.6g} ms<br>%{y:.6g}<extra>" + name + "</extra>",
            ),
            secondary_y=bool(ch.is_frequency),
        )

    fig.update_layout(
        title=title if title is not None else (seq.name if seq else ""),
        xaxis=dict(
            title="Time (ms)",
            gridcolor=_GRID, zerolinecolor=_AXIS_LINE, linecolor=_AXIS_LINE,
            # Dark range-slider so the bottom selector matches the card theme
            # (Plotly's default is a light-grey strip with a white background).
            rangeslider=dict(visible=True, bgcolor=_PLOT_BG,
                             bordercolor=_AXIS_LINE, thickness=0.12),
        ),
        legend_title="Channel",
        font=dict(color=_FONT, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_PLOT_BG,
        margin=dict(l=60, r=60, t=40, b=40),
        uirevision="seq",  # keep zoom/pan across channel toggles
    )
    # Subtle grid on the primary axes only; the secondary (frequency) axis keeps
    # its ticks but no gridlines, so the two y-grids don't overlap into clutter.
    fig.update_yaxes(title_text="Value", secondary_y=False,
                     gridcolor=_GRID, zerolinecolor=_AXIS_LINE, linecolor=_AXIS_LINE)
    fig.update_yaxes(title_text="Frequency (Hz)", secondary_y=True,
                     showgrid=False, zerolinecolor=_AXIS_LINE, linecolor=_AXIS_LINE)
    return fig
