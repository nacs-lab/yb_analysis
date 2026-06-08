"""Phase-5 abort latency/UX hint (must-fix #6).

Abort is per-SEQUENCE: the in-flight FPGA shot can't be interrupted, so a stop lands at the next
sequence boundary (~0.1-1 s). The control panel surfaces ``_ABORT_HINT`` from the moment Abort is
pressed until the scan actually stops, so the operator isn't told "Running" right after pressing
Abort. This tests the pure status-masking logic (``_apply_abort_hint``) without a Tk display by
constructing the panel via ``__new__`` (no GUI init).

Run in the yb_analysis conda env:
    python -m pytest yb_analysis/tests/test_abort_latency_hint.py -v
"""

import pytest

from yb_analysis.gui.control_panel import ControlPanel, _ABORT_HINT, _STATUS_COLORS


def _panel(abort_pending):
    """A ControlPanel with ONLY the fields _apply_abort_hint touches (no Tk init)."""
    cp = ControlPanel.__new__(ControlPanel)
    cp._abort_pending = abort_pending
    return cp


def test_hint_masks_non_terminal_states_while_pending():
    cp = _panel(abort_pending=True)
    # Right after Abort, a still-running/pausing/paused poll is masked by the hint.
    for raw in ("Running", "Pausing...", "Paused"):
        assert cp._apply_abort_hint(raw) == _ABORT_HINT
    assert cp._abort_pending is True            # still in flight


def test_hint_clears_when_scan_stops():
    cp = _panel(abort_pending=True)
    assert cp._apply_abort_hint("Stopped") == "Stopped"   # abort landed
    assert cp._abort_pending is False
    # once cleared, real statuses pass through untouched
    assert cp._apply_abort_hint("Running") == "Running"


def test_hint_clears_on_idle_badge():
    # An abort that drops the runner into the idle/keep-alive state also clears the hint.
    for idle in ("Idle (default)", "Idle (dummy off)", "Idle (last seq)"):
        cp = _panel(abort_pending=True)
        assert cp._apply_abort_hint(idle) == idle
        assert cp._abort_pending is False


def test_no_hint_when_not_pending():
    cp = _panel(abort_pending=False)
    for raw in ("Running", "Paused", "Stopped", "Idle (default)"):
        assert cp._apply_abort_hint(raw) == raw
    assert cp._abort_pending is False


def test_hint_has_a_status_color():
    # The hint must render with a defined color (no fallback-to-black surprise).
    assert _ABORT_HINT in _STATUS_COLORS
