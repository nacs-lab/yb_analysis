"""Tests for port_utils -- the kill-and-wait handoff (no hardware, no network).

The behavior under test is the *wait* added to fix the "Orca not responding to
the controller" wedge: a force-killed backend releases the single DCAM camera
handle only as the process is torn down, so a replacement must not be spawned
until the old process is actually gone. ``_wait_pids_gone`` is that gate.
"""

from yb_analysis.acquisition import port_utils


def test_wait_pids_gone_returns_true_when_all_die(monkeypatch):
    # pid 111 dies after 2 polls, pid 222 after 1 -- the wait must succeed.
    alive = {111: 2, 222: 1}

    def fake_alive(pid):
        if alive.get(pid, 0) > 0:
            alive[pid] -= 1
            return True
        return False

    monkeypatch.setattr(port_utils, "_pid_alive", fake_alive)
    monkeypatch.setattr(port_utils.time, "sleep", lambda *_: None)  # no real wait
    assert port_utils._wait_pids_gone([111, 222], timeout=5.0) is True


def test_wait_pids_gone_times_out_when_pid_survives(monkeypatch):
    # An immortal pid must make the wait return False (and not hang) once the
    # deadline passes -- the caller logs a warning and proceeds.
    monkeypatch.setattr(port_utils, "_pid_alive", lambda pid: True)
    # Fake a clock that jumps past the deadline after the first sleep.
    t = {"now": 1000.0}
    monkeypatch.setattr(port_utils.time, "monotonic", lambda: t["now"])

    def fake_sleep(_):
        t["now"] += 10.0   # advance well past any small timeout

    monkeypatch.setattr(port_utils.time, "sleep", fake_sleep)
    assert port_utils._wait_pids_gone([999], timeout=1.0) is False


def test_wait_pids_gone_empty_is_trivially_true(monkeypatch):
    # No pids -> nothing to wait for.
    monkeypatch.setattr(port_utils, "_pid_alive",
                        lambda pid: (_ for _ in ()).throw(AssertionError("unused")))
    assert port_utils._wait_pids_gone([], timeout=5.0) is True


def _patch_both_platforms(monkeypatch, pids):
    """Patch the OS-specific bits for BOTH branches so kill_port works on the
    running platform without monkeypatching os.name (which breaks pathlib)."""
    monkeypatch.setattr(port_utils, "_stale_pids_windows", lambda port: list(pids))
    monkeypatch.setattr(port_utils, "_stale_pids_posix", lambda port: list(pids))
    monkeypatch.setattr(port_utils.subprocess, "call", lambda *a, **k: 0)  # win taskkill ok
    monkeypatch.setattr(port_utils.os, "kill", lambda pid, sig: None)       # posix kill ok


def test_kill_port_waits_for_killed_pids(monkeypatch):
    # kill_port(wait=True) must call the wait on exactly the pids it killed.
    _patch_both_platforms(monkeypatch, [4242])
    waited = {}

    def fake_wait(pids, timeout):
        waited["pids"] = list(pids)
        return True

    monkeypatch.setattr(port_utils, "_wait_pids_gone", fake_wait)
    n = port_utils.kill_port(1408, wait=True, wait_timeout=3.0)
    assert n == 1
    assert waited["pids"] == [4242]


def test_kill_port_no_wait_skips_the_gate(monkeypatch):
    _patch_both_platforms(monkeypatch, [7])

    def boom(*a, **k):
        raise AssertionError("_wait_pids_gone must not run when wait=False")

    monkeypatch.setattr(port_utils, "_wait_pids_gone", boom)
    assert port_utils.kill_port(1408, wait=False) == 1
