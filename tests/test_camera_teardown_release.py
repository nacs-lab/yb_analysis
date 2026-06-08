"""Graceful camera release before a backend teardown (restart / switch / close).

``control_panel._release_camera_before_teardown`` is the camera-safety core of the monitor's
restart path: a fire-and-forget ``camera_close`` could let the backend be hard-killed while it
still holds the single Orca DCAM handle (a wedge). For pyctrl it aborts any in-flight scan (so
the consume loop can service the close), sends ``camera_close``, then WAITS for ``camera_status``
to report ``connected == False`` before returning -- so the kill that follows is safe.

These pin: the pyctrl abort+close+poll path, the MATLAB close-only path (no abort, no poll), the
timeout fallback, and best-effort error handling. Pure function + injected client/clock/sleep --
no Tk window needed.
"""
from yb_analysis.gui.control_panel import _release_camera_before_teardown


class _FakeClient:
    def __init__(self, status_seq=None, abort_raises=False,
                 close_raises=False, status_raises=False):
        self.calls = []
        self._status_seq = list(status_seq or [])
        self.abort_raises = abort_raises
        self.close_raises = close_raises
        self.status_raises = status_raises

    def abort_seq(self):
        self.calls.append('abort')
        if self.abort_raises:
            raise RuntimeError('abort boom')

    def camera_close(self):
        self.calls.append('close')
        if self.close_raises:
            raise RuntimeError('close boom')

    def camera_status(self):
        self.calls.append('status')
        if self.status_raises:
            raise RuntimeError('status boom')
        if self._status_seq:
            return self._status_seq.pop(0)
        return {'connected': False}            # released once the scripted sequence runs out


class _Clock:
    """Monotonic-ish fake: returns t, advances by `step` each call (deterministic timeout)."""
    def __init__(self, step=0.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class _NoLog:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def _release(client, backend, **kw):
    kw.setdefault('sleep', lambda dt: None)
    kw.setdefault('clock', _Clock(step=0.0))
    kw.setdefault('log', _NoLog())
    return _release_camera_before_teardown(client, backend, **kw)


# --- pyctrl: abort -> close -> poll until released ---------------------------

def test_pyctrl_aborts_closes_then_waits_until_released():
    # Two "still connected" polls, then released.
    c = _FakeClient(status_seq=[{'connected': True}, {'connected': True}, {'connected': False}])
    assert _release(c, 'pyctrl') is True
    # abort first (free the loop), then close, then it polled status until connected==False.
    assert c.calls[0] == 'abort'
    assert c.calls[1] == 'close'
    assert c.calls.count('status') == 3
    assert c.calls[-1] == 'status'


def test_pyctrl_returns_quickly_when_already_released():
    c = _FakeClient(status_seq=[{'connected': False}])
    assert _release(c, 'pyctrl') is True
    assert c.calls == ['abort', 'close', 'status']     # one poll, sees released


# --- MATLAB: close only, no abort, no poll -----------------------------------

def test_matlab_sends_close_only_no_abort_no_poll():
    c = _FakeClient()
    assert _release(c, 'matlab') is True
    assert c.calls == ['close']                        # no abort, no status poll
    assert 'abort' not in c.calls and 'status' not in c.calls


# --- timeout fallback: never blocks forever ----------------------------------

def test_pyctrl_times_out_and_proceeds():
    # Camera never releases; the advancing clock crosses the deadline -> return False.
    c = _FakeClient(status_seq=[{'connected': True}] * 50)
    out = _release(c, 'pyctrl', timeout_s=5.0, clock=_Clock(step=2.0))
    assert out is False                                 # proceeds with teardown anyway
    assert 'close' in c.calls and c.calls.count('status') >= 1


# --- best-effort error handling: never raises --------------------------------

def test_close_error_returns_false_without_polling():
    c = _FakeClient(close_raises=True)
    assert _release(c, 'pyctrl') is False
    assert c.calls == ['abort', 'close']               # close failed -> no poll
    assert 'status' not in c.calls


def test_status_error_stops_waiting():
    c = _FakeClient(status_raises=True, status_seq=[{'connected': True}])
    assert _release(c, 'pyctrl') is False              # can't query -> stop, best-effort
    assert c.calls == ['abort', 'close', 'status']


def test_abort_error_is_swallowed_and_close_still_sent():
    c = _FakeClient(abort_raises=True, status_seq=[{'connected': False}])
    assert _release(c, 'pyctrl') is True               # abort failure must not stop the release
    assert c.calls[:2] == ['abort', 'close']
