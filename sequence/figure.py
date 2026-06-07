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


_VERTEX_SIZE = 5          # marker size at real segment endpoints (the "magnet" dots)
_HOVER_TARGET_POINTS = 400  # ~this many hover points spread across the full time span


def _densify_for_hover(t, v, pid, target_points=_HOVER_TARGET_POINTS, max_points=4000):
    """Insert invisible in-between points so hover/datatip works ALONG flat lines.

    A constant-value pulse is drawn as just two endpoints, so Plotly's
    closest-point hover only fires near those two dots -- the middle of the line
    is "dead". We subdivide each intra-pulse segment (both endpoints share a
    ``pid``) finely enough that the gap between hover points never exceeds
    ``span / target_points``. The inserted points are linearly interpolated
    (correct time + value for the datatip) and carry the segment's ``pid`` so the
    SVG hover-overlay still highlights the whole pulse. Real vertices keep the
    visible marker (size ``_VERTEX_SIZE``); inserted points get size 0 -- so the
    "magnet" still snaps the cursor to meaningful segment endpoints while the
    flat line in between is fully hoverable.

    Returns ``(t, v, pid, sizes)`` as plain lists. Segments that cross a pulse
    boundary (differing pids, e.g. a vertical jump) are left as-is.
    """
    n = len(t)
    if n < 2:
        return list(t), list(v), list(pid), [_VERTEX_SIZE] * n
    span = float(t[-1]) - float(t[0])
    gap = (span / target_points) if span > 0 else 0.0
    ot, ov, op, os = [], [], [], []
    for i in range(n - 1):
        t0, t1 = float(t[i]), float(t[i + 1])
        v0, v1 = float(v[i]), float(v[i + 1])
        p0, p1 = int(pid[i]), int(pid[i + 1])
        ot.append(t0); ov.append(v0); op.append(p0); os.append(_VERTEX_SIZE)
        seg = t1 - t0
        if gap > 0 and p0 == p1 and seg > gap and len(ot) < max_points:
            k = min(int(seg / gap), max_points - len(ot))
            for j in range(1, k):
                f = j / k
                ot.append(t0 + f * seg); ov.append(v0 + f * (v1 - v0))
                op.append(p0); os.append(0)
    ot.append(float(t[-1])); ov.append(float(v[-1]))
    op.append(int(pid[-1])); os.append(_VERTEX_SIZE)
    return ot, ov, op, os


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
    # moves its single point on click. Scattergl to match the channel traces --
    # one rendering layer, so trace order alone determines z-order (no SVG/GL mix).
    fig.add_trace(
        go.Scattergl(x=[], y=[], mode="markers", name="Selected",
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
        vlist = [float(x) for x in ch.v]
        # Kill float64 noise on a CONSTANT channel. A channel that never changes
        # still carries ~1e-7-level FP jitter between samples; viewed alone, Plotly
        # auto-ranges the axis to that jitter band, so a flat line renders as wild
        # spikes (e.g. a constant 252.07 MHz EOM freq). If the whole channel is
        # constant to within 1e-9 (relative), pin every sample to the mean so it
        # draws as a true flat line. Real variation (> 1e-9 relative) is untouched.
        if vlist:
            vmin, vmax = min(vlist), max(vlist)
            ref = max(abs(vmin), abs(vmax), 1.0)
            if (vmax - vmin) <= ref * 1e-9:
                vlist = [(vmin + vmax) / 2.0] * len(vlist)
        # Densify so hover/datatip works along flat (constant-value) lines, not
        # just at the two endpoints. Vertices keep a visible marker; the inserted
        # in-between points are invisible (size 0) but fully hoverable.
        dt, dv, dpid, dsize = _densify_for_hover(
            ch.t_ms.tolist(), vlist, [int(p) for p in ch.pid])
        # WebGL (Scattergl), not SVG: each channel carries its real samples plus up
        # to ~400 densified hover points, so selecting many channels at once is
        # thousands of points -- SVG melts the browser, GL stays smooth. The only
        # reason this had to be SVG (Scattergl doesn't render inside a Plotly
        # rangeslider) is gone now that the rangeslider is removed (see below).
        fig.add_trace(
            go.Scattergl(
                x=dt,
                y=dv,
                customdata=dpid,  # pulse_id (for backtrace + hover/region overlay)
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                # No marker border -- the default leaves a pale ring around each
                # visible vertex dot, which reads as clutter on the dark bg.
                marker=dict(color=color, size=dsize, line=dict(width=0, color=color)),
                hovertemplate="%{x:.6g} ms<br>%{y:.6g}<extra>" + name + "</extra>",
            ),
            secondary_y=bool(ch.is_frequency),
        )

    fig.update_layout(
        # No on-plot title -- the scan name is shown in the Sequence-source card /
        # Scans picker; on the plot it just ate space in the top-left corner.
        title=dict(text=""),
        xaxis=dict(
            title="Time (ms)",
            gridcolor=_GRID, zerolinecolor=_AXIS_LINE, linecolor=_AXIS_LINE,
            # No range-slider: the compressed mini-plot strip at the bottom isn't
            # wanted (it just squeezes the main plot). Use box/zoom + autoscale.
            rangeslider=dict(visible=False),
        ),
        # Channel legend: COLLAPSED by default (the JS reveals it on hover and
        # pins it on click) + a smaller font so it doesn't crowd the plot when
        # shown. Anchored TOP-RIGHT inside the plot (over a translucent panel) so
        # it sits where the reveal/pin button is. uirevision keeps the JS-driven
        # showlegend state across react.
        showlegend=False,
        legend=dict(title=dict(text="Channel", font=dict(size=10)),
                    font=dict(size=10), itemsizing="constant",
                    x=0.99, xanchor="right", y=0.68, yanchor="top",
                    bgcolor="rgba(11,14,19,0.72)",
                    bordercolor="rgba(255,255,255,0.18)", borderwidth=1),
        font=dict(color=_FONT, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_PLOT_BG,
        margin=dict(l=60, r=60, t=18, b=40),   # tight top: no title anymore
        hovermode="closest",  # snap the datatip to the nearest point (the "magnet")
        uirevision="seq",  # keep zoom/pan + legend show/hide across channel toggles
    )
    # Subtle grid on the primary axes only; the secondary (frequency) axis keeps
    # its ticks but no gridlines, so the two y-grids don't overlap into clutter.
    fig.update_yaxes(title_text="Value", secondary_y=False,
                     gridcolor=_GRID, zerolinecolor=_AXIS_LINE, linecolor=_AXIS_LINE)
    fig.update_yaxes(title_text="Frequency (Hz)", secondary_y=True,
                     showgrid=False, zerolinecolor=_AXIS_LINE, linecolor=_AXIS_LINE)
    return fig
