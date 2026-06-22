"""The merged Sequence tab is present in the dashboard HTML/JS/CSS.

These guard the wiring (markup ids the JS expects, JS handlers, CSS classes)
without a browser. Run in the yb_analysis env::

    python -m pytest yb_analysis/tests/test_sequence_ui.py -v
"""

import os

import pytest

_PLOTTING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "yb_analysis", "plotting")


@pytest.fixture
def html_client():
    from yb_analysis.plotting import dashboard as dash_mod
    from flask import Flask
    app = Flask("seq_ui_test")
    dash_mod._register_main_html_routes(app)
    app.testing = True
    return app.test_client()


def test_index_renders_sequence_tab(html_client):
    r = html_client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    for marker in [
        # The standalone Sequence tab was removed; the sequence view is now a
        # sub-mode of the Analysis tab (Data/Sequence toggle). #tab-sequence
        # still exists as the (hidden) source container the cards are reparented
        # out of at load (foldSequenceIntoAnalysis).
        'id="analysis-mode-toggle"', 'data-mode="sequence"',
        'id="analysis-sequence-pane"',
        'id="tab-sequence"', 'id="plot-sequence"',
        'id="seq-folder"', 'id="seq-load-btn"', 'id="seq-point-select"',
        'id="seq-seq-select"', 'id="seq-params"',
        # Custom channel picker (replaced the native <select multiple>).
        'id="seq-chn-list"', 'id="seq-chn-clear"', 'id="seq-chn-search"',
        'data-card-id="sequence-source"', 'data-card-id="sequence-plot"',
        'data-card-id="sequence-params"', 'id="seq-autosave"',
        # Two floating-picker hosts: the SHARED "Scans" picker docks LEFT
        # (#floating-seqscan-host; dashboard.js reparents the unified
        # #analysis-runs-card into it -- there's no separate seqscan-card
        # anymore), Channels docks RIGHT.
        'id="floating-seqscan-host"',
        'id="floating-sequence-host"', 'id="sequence-chn-card"',
        # Params search + config/modified/scanned show-hide toggles.
        'id="seq-param-search"', 'id="seq-filter-config"',
        'id="seq-filter-modified"', 'id="seq-filter-scanned"',
    ]:
        assert marker in html, "missing in rendered HTML: " + marker


def test_dashboard_js_wires_sequence():
    js = open(os.path.join(_PLOTTING, "static", "dashboard.js"),
              encoding="utf-8").read()
    for marker in [
        'function foldSequenceIntoAnalysis',   # sequence cards -> Analysis pane
        'function setAnalysisSubMode',          # Data/Sequence sub-toggle
        'function initSequenceTab',
        'function loadSequence',
        'function seqRenderPlot',
        'function seqParamTree',
        '/api/sequence/list',
        '/api/sequence/figure',
        '/api/sequence/params',
        'initSequenceTab();',         # bootstrap call
        'if (tab === "sequence") return loadSequence();',
        '/api/sequence/dump_toggle',  # the auto-dump toggle
        'seq-autosave',
        'floating-seqscan-host',      # the left-docked Scans host (toggled per tab)
        # Three-state picker + reconstruct trigger (§12.4).
        'function seqReconstruct',
        '/api/sequence/reconstruct',
        'has_snapshot',
        'has_descriptor',       # Reconstructable requires a descriptor too (no dead button)
        'seq-row-working',      # the in-flight reconstruct row class (literal in JS)
        # Req 1/2/3: last-30 scans, params filter tree, param<->channel xref.
        'SEQ_SCANS_LIMIT',
        'function seqRenderParamTree',
        'function seqOnParamChannels',
        'function seqFocusPoint',        # point-click -> segment-specific params + formula
        'function seqSelectParam',       # param-click -> promote channels + emphasize regions
        'function seqSetFocus',          # the selection focus region (top of params panel)
        'seqMaybeBuildXref',             # background build/upgrade of xref.json
        'function seqForceRebuildXref',  # the "Rebuild ⟳" button handler
        'seq-rebuild-xref-btn',
        'function seqWirePlotHover',     # hover -> thick-line pulse highlight (2c)
        'time_regions',                  # wait/timing param -> shaded time bands (point 3)
        '/api/sequence/xref',
        '/api/sequence/build_xref',
        '/api/sequence/backtrace',        # click point -> source file:line panel
        'function seqShowBacktrace',
        'seq-notice-error',               # surfaced xref-build failure banner
    ]:
        assert marker in js, "missing JS wiring: " + marker


def test_dashboard_css_has_sequence_styles():
    css = open(os.path.join(_PLOTTING, "static", "dashboard.css"),
               encoding="utf-8").read()
    for marker in ["#plot-sequence", ".seq-tree",
                   ".seq-modified", ".seq-config", ".seq-scanned-badge",
                   "#floating-seqscan-host",          # left-dock for the Scans picker
                   ".seq-row-reconstructable",        # three-state picker
                   ".seq-row-unrecoverable",
                   ".seq-param-filters",              # params category toggles
                   ".seq-leaf-xref-hit",              # channel->param highlight
                   ".seq-focus",                      # selection focus region
                   ".seq-chip",                       # focus param/channel chips
                   ".seq-chip-val"]:                  # lifted parameter value
        assert marker in css, "missing CSS: " + marker
