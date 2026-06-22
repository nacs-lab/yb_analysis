"""Tests for PyctrlLauncher.start() -- the clean-handoff settle (no hardware).

Regression guard for the "Orca not responding to the controller" wedge: the
launcher must clear the port AND let the old backend's DCAM handle release
BEFORE it spawns the replacement, or the new backend's camera open races the
release and (being a GIL-holding DCAM call) hangs the whole backend.
"""

from yb_analysis.acquisition import pyctrl_launcher
from yb_analysis.acquisition.pyctrl_launcher import PyctrlLauncher


class _FakeProc:
    def poll(self):
        return None     # always "alive" so start() proceeds to the ping check


def _wire(monkeypatch, order):
    """Patch the launcher's externals to record call order, no real I/O."""
    monkeypatch.setattr(pyctrl_launcher, "kill_port",
                        lambda port, **k: order.append(("kill_port", port)))
    monkeypatch.setattr(pyctrl_launcher.time, "sleep",
                        lambda s: order.append(("sleep", s)))
    monkeypatch.setattr(pyctrl_launcher.subprocess, "Popen",
                        lambda *a, **k: order.append(("popen",)) or _FakeProc())
    monkeypatch.setattr(pyctrl_launcher, "_ping", lambda url, **k: True)


def test_start_settles_between_port_clear_and_spawn(monkeypatch):
    monkeypatch.setenv("YB_BACKEND_SPAWN_SETTLE_S", "0.25")
    order = []
    _wire(monkeypatch, order)
    PyctrlLauncher("py", "mod", "tcp://127.0.0.1:1408").start()
    kinds = [o[0] for o in order]
    # kill_port must come first, the settle sleep must come before popen.
    assert kinds.index("kill_port") < kinds.index("sleep") < kinds.index("popen")
    assert ("sleep", 0.25) in order


def test_start_settle_disabled_skips_sleep(monkeypatch):
    monkeypatch.setenv("YB_BACKEND_SPAWN_SETTLE_S", "0")
    order = []
    _wire(monkeypatch, order)
    PyctrlLauncher("py", "mod", "tcp://127.0.0.1:1408").start()
    kinds = [o[0] for o in order]
    assert "sleep" not in kinds                          # settle=0 -> no delay
    assert kinds.index("kill_port") < kinds.index("popen")


def test_start_settle_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("YB_BACKEND_SPAWN_SETTLE_S", "notanumber")
    order = []
    _wire(monkeypatch, order)
    PyctrlLauncher("py", "mod", "tcp://127.0.0.1:1408").start()
    # Falls back to the 1.5 s default rather than crashing.
    assert ("sleep", 1.5) in order
