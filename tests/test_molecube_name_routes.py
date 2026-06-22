"""/api/molecube/{dds,ttl}/name: route -> daemon name-store round-trip (OFFLINE).

End-to-end through the real Flask routes and the real MolecubeClient, but against the
in-process mock daemon (mock_molecube_server). NO live FPGA daemon is contacted and the
real channel names are never touched -- the mock seeds its own fake names.

Verifies a rename POST writes through to the daemon's own name store
(set_{dds,ttl}_names), merges (does not clobber sibling channels), clears on an empty
name, sanitizes input, validates the body, and honours the write gate.

    python -m pytest yb_analysis/tests/test_molecube_name_routes.py -v
"""

import pytest

from yb_analysis.plotting import dashboard as dash_mod
from yb_analysis.control.mock_molecube_server import MockMolecubeServer


@pytest.fixture
def client(monkeypatch):
    from flask import Flask
    from yb_analysis import config as yb_cfg

    url = "tcp://127.0.0.1:7796"
    srv = MockMolecubeServer(url).start()
    # Point the dashboard's client at the mock; reset the cached singleton (keyed by URL).
    monkeypatch.setattr(yb_cfg, "MOLECUBE_URL", url)
    monkeypatch.setattr(yb_cfg, "MOLECUBE_TIMEOUT_MS", 1500)
    monkeypatch.setattr(yb_cfg, "MOLECUBE_MAX_TTL_CHN", 31)
    monkeypatch.setattr(dash_mod, "_molecube_client_box", {})
    # Reads + writes open (defaults, but pin them so env from a prior test can't leak).
    monkeypatch.setenv("YB_MOLECUBE_READS", "1")
    monkeypatch.setenv("YB_MOLECUBE_WRITES", "1")
    monkeypatch.setenv("YB_MOLECUBE_TTL_READS", "1")

    app = Flask("molecube_name_test")
    dash_mod._register_molecube_routes(app)
    app.testing = True
    try:
        yield app.test_client()
    finally:
        srv.stop()


def _names(cl, kind):
    snap = cl.get("/api/molecube/snapshot").get_json()
    return {r["chn"]: r["name"] for r in snap[kind]}


def test_dds_name_route_writes_through(client):
    r = client.post("/api/molecube/dds/name", json={"chn": 2, "name": "SLM_x"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert _names(client, "dds")[2] == "SLM_x"


def test_ttl_name_route_writes_through(client):
    r = client.post("/api/molecube/ttl/name", json={"chn": 10, "name": "Tweak"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert _names(client, "ttl")[10] == "Tweak"


def test_name_route_merges_not_clobbers(client):
    # chn 7 is "EOM616" in the mock seed; renaming chn 0 must leave 7 alone.
    client.post("/api/molecube/dds/name", json={"chn": 0, "name": "AAA"})
    names = _names(client, "dds")
    assert names[0] == "AAA" and names[7] == "EOM616"


def test_name_route_empty_clears(client):
    # whitespace-only trims to "" -> reads back as no name (row still present, blank).
    client.post("/api/molecube/dds/name", json={"chn": 7, "name": "  "})
    assert _names(client, "dds").get(7, "") == ""


def test_name_route_strips_nul_and_caps_length(client):
    client.post("/api/molecube/dds/name",
                json={"chn": 1, "name": "a\x00b" + "x" * 100})
    nm = _names(client, "dds")[1]
    assert "\x00" not in nm and len(nm) <= 63 and nm.startswith("ab")


def test_name_route_bad_request(client):
    assert client.post("/api/molecube/dds/name", json={"name": "x"}).status_code == 400
    assert client.post("/api/molecube/ttl/name", json={"chn": 3}).status_code == 400


def test_name_route_write_gate_closed(client, monkeypatch):
    monkeypatch.setenv("YB_MOLECUBE_WRITES", "0")
    r = client.post("/api/molecube/dds/name", json={"chn": 2, "name": "nope"})
    assert r.status_code == 403
    assert r.get_json()["ok"] is False
