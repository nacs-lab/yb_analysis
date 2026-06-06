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
        'id="tab-btn-sequence"', 'data-tab="sequence"',
        'id="tab-sequence"', 'id="plot-sequence"',
        'id="seq-folder"', 'id="seq-load-btn"', 'id="seq-point-select"',
        'id="seq-seq-select"', 'id="seq-chn-select"', 'id="seq-params"',
        'data-card-id="sequence-source"', 'data-card-id="sequence-plot"',
        'data-card-id="sequence-params"', 'id="seq-autosave"',
    ]:
        assert marker in html, "missing in rendered HTML: " + marker


def test_dashboard_js_wires_sequence():
    js = open(os.path.join(_PLOTTING, "static", "dashboard.js"),
              encoding="utf-8").read()
    for marker in [
        '"sequence"',                 # in TABS
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
    ]:
        assert marker in js, "missing JS wiring: " + marker


def test_dashboard_css_has_sequence_styles():
    css = open(os.path.join(_PLOTTING, "static", "dashboard.css"),
               encoding="utf-8").read()
    for marker in ["#tab-sequence .plot-container", ".seq-tree",
                   ".seq-modified", ".seq-config", ".seq-scanned-badge"]:
        assert marker in css, "missing CSS: " + marker
