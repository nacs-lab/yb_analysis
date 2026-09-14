"""/api/molecube/snapshot: the shared, id-gated snapshot cache (OFFLINE, mock daemon).

The channels page polls this route every second (labctrl-node model). The route must
cost the daemon only the two cheap ``state_id``/``name_id`` requests per poll and run
the full DDS/TTL read (``MolecubeClient.snapshot``) only when:

* nothing is cached yet, or ``?force=1`` (the Refresh button),
* the daemon's change counter / name id moved (idle edit, rename, sequence start/stop),
* a sequence is RUNNING (state_id sign bit) and the cache is older than
  ``MOLECUBE_RUN_REFRESH_S`` (unless ``YB_MOLECUBE_LIVE_RUNNING=0``),
* the cache is older than ``MOLECUBE_SNAPSHOT_MAX_AGE_S``,
* a write route ran (the write wrapper invalidates the cache).

Runs against the in-process mock daemon -- no FPGA is contacted.

    python -m pytest yb_analysis/tests/test_molecube_snapshot_cache.py -v
"""

import pytest

from yb_analysis.plotting import dashboard as dash_mod
from yb_analysis.control.mock_molecube_server import MockMolecubeServer
from yb_analysis.control.molecube_client import MolecubeClient

RUN_BIT = 1 << 63


@pytest.fixture
def env(monkeypatch):
    from flask import Flask
    from yb_analysis import config as yb_cfg

    url = "tcp://127.0.0.1:7797"
    srv = MockMolecubeServer(url).start()
    monkeypatch.setattr(yb_cfg, "MOLECUBE_URL", url)
    monkeypatch.setattr(yb_cfg, "MOLECUBE_TIMEOUT_MS", 1500)
    monkeypatch.setattr(yb_cfg, "MOLECUBE_MAX_TTL_CHN", 31)
    monkeypatch.setattr(yb_cfg, "MOLECUBE_RUN_REFRESH_S", 1.0, raising=False)
    monkeypatch.setattr(yb_cfg, "MOLECUBE_SNAPSHOT_MAX_AGE_S", 60.0, raising=False)
    monkeypatch.setattr(dash_mod, "_molecube_client_box", {})
    monkeypatch.setattr(dash_mod, "_MC_SNAP", {"data": None, "ts": 0.0, "key": None})
    monkeypatch.setenv("YB_MOLECUBE_READS", "1")
    monkeypatch.setenv("YB_MOLECUBE_WRITES", "1")
    monkeypatch.setenv("YB_MOLECUBE_TTL_READS", "1")
    monkeypatch.delenv("YB_MOLECUBE_LIVE_RUNNING", raising=False)

    # Count FULL daemon reads (the expensive ~10-request snapshot), not the id polls.
    calls = {"n": 0}
    real = MolecubeClient.snapshot

    def counting(self, *a, **kw):
        calls["n"] += 1
        return real(self, *a, **kw)
    monkeypatch.setattr(MolecubeClient, "snapshot", counting)

    app = Flask("molecube_snapshot_cache_test")
    dash_mod._register_molecube_routes(app)
    app.testing = True
    try:
        yield app.test_client(), srv, calls, yb_cfg, monkeypatch
    finally:
        srv.stop()


def _get(cl, q=""):
    r = cl.get("/api/molecube/snapshot" + q)
    assert r.status_code == 200
    return r.get_json()


def test_first_read_then_cached_without_daemon_reads(env):
    cl, srv, calls, _cfg, _mp = env
    a = _get(cl)
    assert a["connected"] is True and a["cached"] is False and a["refresh"] == "first"
    assert a["running"] is False and a["state_cnt"] == srv._state_id
    assert calls["n"] == 1
    for _ in range(3):
        b = _get(cl)
        assert b["cached"] is True and b["refresh"] is None
        assert b["dds"] == a["dds"] and b["ttl"] == a["ttl"]
    assert calls["n"] == 1                       # id polls only, no full re-read


def test_state_change_triggers_reread(env):
    cl, srv, calls, _cfg, _mp = env
    _get(cl)
    srv._bump_state()                            # idle-time change on the device
    b = _get(cl)
    assert b["cached"] is False and b["refresh"] == "changed"
    assert calls["n"] == 2


def test_write_route_invalidates_cache_and_new_value_shows(env):
    cl, srv, calls, _cfg, _mp = env
    _get(cl)
    r = cl.post("/api/molecube/dds/set", json={"chn": 2, "type": "amp", "value": 0.25})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    b = _get(cl)
    assert b["cached"] is False
    amp = {row["chn"]: row["amp"] for row in b["dds"]}[2]
    assert abs(amp - 0.25) < 1e-3
    assert calls["n"] >= 2


def test_running_bit_refreshes_at_bounded_rate(env):
    cl, srv, calls, cfg, mp = env
    _get(cl)
    srv._state_id |= RUN_BIT                     # daemon: "a sequence is running"
    mp.setattr(cfg, "MOLECUBE_RUN_REFRESH_S", 0.0, raising=False)
    b = _get(cl)
    assert b["running"] is True
    assert b["state_cnt"] == (srv._state_id & (RUN_BIT - 1))
    # the run bit alone is a 'changed' key? No: the key uses state_cnt (bit stripped),
    # so the first poll after the bit appears refreshes for 'running', not 'changed'.
    assert b["refresh"] in ("running", "changed")
    n = calls["n"]
    c = _get(cl)                                 # refresh window 0 s -> re-read every poll
    assert c["cached"] is False and c["refresh"] == "running"
    assert calls["n"] == n + 1
    # A long refresh window -> served from cache while running.
    mp.setattr(cfg, "MOLECUBE_RUN_REFRESH_S", 1e6, raising=False)
    d = _get(cl)
    assert d["cached"] is True and d["running"] is True
    assert calls["n"] == n + 1


def test_live_running_gate_disables_mid_run_reads(env):
    cl, srv, calls, cfg, mp = env
    _get(cl)
    srv._state_id |= RUN_BIT
    mp.setattr(cfg, "MOLECUBE_RUN_REFRESH_S", 0.0, raising=False)
    mp.setenv("YB_MOLECUBE_LIVE_RUNNING", "0")
    n = calls["n"]
    b = _get(cl)
    assert b["running"] is True and b["live_running"] is False
    assert b["cached"] is True                   # no mid-run daemon read
    assert calls["n"] == n


def test_force_bypasses_cache(env):
    cl, srv, calls, _cfg, _mp = env
    _get(cl)
    b = _get(cl, "?force=1")
    assert b["cached"] is False and b["refresh"] == "force"
    assert calls["n"] == 2


def test_stale_cache_rereads(env):
    cl, srv, calls, cfg, mp = env
    _get(cl)
    mp.setattr(cfg, "MOLECUBE_SNAPSHOT_MAX_AGE_S", 0.0, raising=False)
    b = _get(cl)
    assert b["cached"] is False and b["refresh"] == "stale"
    assert calls["n"] == 2


def test_unreachable_daemon_reports_disconnected(env):
    cl, srv, calls, _cfg, _mp = env
    srv.stop()
    b = _get(cl)
    assert b["connected"] is False and "state_id" in b["errors"]
    assert calls["n"] == 0
