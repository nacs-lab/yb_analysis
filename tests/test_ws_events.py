"""Tests for the WebSocket event fan-out (EventHub) -- WITHOUT a live server.

The dashboard pushes tiny shot/queue EVENTS over /ws/events so the browser can
react instantly instead of waiting out its poll interval. The watcher/publisher
logic (frame-advance detection, queue-mtime change, per-client bounded queue
with drop-oldest, and error-tolerance) is factored into `EventHub` with the two
change indicators injectable so it's testable with no files and no socket.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_ws_events.py -v
"""
import queue

import pytest

from yb_analysis.plotting import dashboard as dsh


def _hub(frames, mtimes):
    """An EventHub reading from two mutable one-element lists (frame id, mtime),
    so a test can flip the "current" value between poll_once() calls."""
    return dsh.EventHub(
        read_frame_id=lambda: frames[0],
        read_queue_mtime=lambda: mtimes[0],
    )


def test_first_tick_seeds_no_event():
    """The first watcher tick just latches state -- it must NOT fire a spurious
    startup event for "None -> first observed value"."""
    frames = [("0", 111.0)]
    mtimes = [222.0]
    hub = _hub(frames, mtimes)
    q = hub.register()
    assert hub.poll_once() == []        # seed tick
    assert q.empty()


def test_frame_advance_publishes_exactly_one_shot_per_change():
    frames = [("0", 100.0)]
    mtimes = [None]
    hub = _hub(frames, mtimes)
    q = hub.register()
    hub.poll_once()                     # seed

    # No change -> no event.
    assert hub.poll_once() == []
    assert q.empty()

    # Advance the frame (buffer mtime changed) -> exactly one shot event.
    frames[0] = ("1", 100.5)
    evs = hub.poll_once()
    assert evs == [{"topic": "shot", "frame": "1"}]
    assert q.get_nowait() == {"topic": "shot", "frame": "1"}

    # Same frame again -> no further event.
    assert hub.poll_once() == []
    assert q.empty()

    # Another advance (same pointer content, new mtime) still counts as new.
    frames[0] = ("1", 101.0)
    evs = hub.poll_once()
    assert evs == [{"topic": "shot", "frame": "1"}]
    assert q.get_nowait()["topic"] == "shot"


def test_queue_mtime_change_publishes_queue_event():
    frames = [("0", 5.0)]
    mtimes = [10.0]
    hub = _hub(frames, mtimes)
    q = hub.register()
    hub.poll_once()                     # seed

    assert hub.poll_once() == []        # unchanged
    mtimes[0] = 11.0
    evs = hub.poll_once()
    assert evs == [{"topic": "queue"}]
    assert q.get_nowait() == {"topic": "queue"}


def test_frame_and_queue_change_same_tick():
    frames = [("0", 1.0)]
    mtimes = [1.0]
    hub = _hub(frames, mtimes)
    q = hub.register()
    hub.poll_once()                     # seed
    frames[0] = ("1", 2.0)
    mtimes[0] = 2.0
    evs = hub.poll_once()
    assert {"topic": "shot", "frame": "1"} in evs
    assert {"topic": "queue"} in evs
    got = {q.get_nowait()["topic"], q.get_nowait()["topic"]}
    assert got == {"shot", "queue"}


def test_full_client_queue_drops_oldest_without_blocking():
    """A slow/full client queue must never wedge publish() or its peers: on
    overflow we drop the OLDEST event and keep the newest."""
    hub = dsh.EventHub(client_maxsize=3)
    q = hub.register()
    for i in range(3):
        hub.publish({"topic": "shot", "frame": str(i)})
    assert q.full()
    # One more -> drops the oldest (frame "0"), appends the newest (frame "3").
    hub.publish({"topic": "shot", "frame": "3"})
    frames = []
    while not q.empty():
        frames.append(q.get_nowait()["frame"])
    assert frames == ["1", "2", "3"]    # oldest ("0") dropped, newest kept


def test_publish_no_clients_is_noop():
    """With zero subscribers publish() short-circuits (cheap-when-idle)."""
    hub = dsh.EventHub()
    hub.publish({"topic": "shot", "frame": "0"})   # must not raise


def test_one_full_client_does_not_starve_another():
    hub = dsh.EventHub(client_maxsize=2)
    slow = hub.register()
    fast = hub.register()
    # Fill `slow` to capacity, drain `fast` so it stays empty.
    hub.publish({"topic": "shot", "frame": "a"})
    hub.publish({"topic": "shot", "frame": "b"})
    while not fast.empty():
        fast.get_nowait()
    assert slow.full()
    # New event: `slow` drops-oldest (still delivered), `fast` gets it cleanly.
    hub.publish({"topic": "shot", "frame": "c"})
    assert fast.get_nowait() == {"topic": "shot", "frame": "c"}
    drained = []
    while not slow.empty():
        drained.append(slow.get_nowait()["frame"])
    assert drained[-1] == "c"           # newest delivered to the slow client too


def test_transient_read_error_does_not_kill_the_loop():
    """A read raising the tolerated exception families is swallowed: poll_once
    treats it as 'no value / no change' and the NEXT good read still fires."""
    state = {"raise": False}

    def read_frame():
        if state["raise"]:
            raise OSError("transient pointer race")
        return frames[0]

    frames = [("0", 1.0)]
    hub = dsh.EventHub(read_frame_id=read_frame, read_queue_mtime=lambda: None)
    q = hub.register()
    hub.poll_once()                     # seed

    # A transient error mid-stream must not raise and must publish nothing.
    state["raise"] = True
    frames[0] = ("1", 2.0)             # change hidden behind the error
    assert hub.poll_once() == []       # error swallowed -> no event
    assert q.empty()

    # Recovery: the next good read sees the change and fires.
    state["raise"] = False
    evs = hub.poll_once()
    assert evs == [{"topic": "shot", "frame": "1"}]


def test_unpickling_error_is_tolerated():
    """UnpicklingError (a reader raising mid-decode) is in the swallowed set:
    poll_once returns [] rather than propagating -- so the watcher loop lives."""
    import pickle as _pk

    def boom():
        raise _pk.UnpicklingError("boom")

    hub = dsh.EventHub(read_frame_id=boom, read_queue_mtime=boom)
    hub.register()
    assert hub.poll_once() == []


def test_snapshot_hello_shape():
    hub = dsh.EventHub(
        read_frame_id=lambda: ("1", 9.0),
        read_queue_mtime=lambda: 42.0,
    )
    hello = hub.snapshot_hello()
    assert hello == {"topic": "hello", "frame": "1", "queue_mtime": 42.0}
    # Null-safe when nothing has been written yet.
    hub2 = dsh.EventHub(read_frame_id=lambda: None, read_queue_mtime=lambda: None)
    assert hub2.snapshot_hello() == {
        "topic": "hello", "frame": None, "queue_mtime": None}


def test_register_starts_watcher_lazily():
    """No watcher thread until the first client registers (cheap when idle)."""
    hub = dsh.EventHub(
        read_frame_id=lambda: ("0", 1.0), read_queue_mtime=lambda: None,
        poll_interval=0.01)
    assert hub._thread is None and not hub._started
    hub.register()
    assert hub._started and hub._thread is not None
    assert hub._thread.daemon


# ---- Eager live-figure pre-build (on_shot) -----------------------------------
# On a frame advance the watcher pre-builds the default-arg live figures off the
# request thread (only while a client is connected), so concurrent polls read a
# ready string instead of serializing behind a ~100 ms inline build under the
# lock. The hook is injectable (on_shot) so it's testable with no files/figures.

def test_on_shot_fires_only_on_frame_change_with_clients():
    """on_shot pre-build hook fires exactly once per frame advance, and ONLY
    while >=1 client is registered (no client -> no wasted GIL-held build)."""
    calls = {"n": 0}
    frames = [("0", 1.0)]
    hub = dsh.EventHub(read_frame_id=lambda: frames[0],
                       read_queue_mtime=lambda: None,
                       on_shot=lambda: calls.__setitem__("n", calls["n"] + 1))
    hub.poll_once()                     # seed (no client)
    frames[0] = ("1", 2.0)
    hub.poll_once()                     # frame change but NO client -> no fire
    assert calls["n"] == 0

    q = hub.register()                  # now a client is watching
    hub.poll_once()                     # same frame -> no change -> no fire
    assert calls["n"] == 0
    frames[0] = ("0", 3.0)
    hub.poll_once()                     # frame change WITH client -> fire once
    assert calls["n"] == 1
    _ = q  # keep the client registered


def test_on_shot_error_does_not_kill_poll():
    """A failing pre-build is best-effort: swallowed, the shot event still
    publishes, and the watcher keeps running."""
    frames = [("0", 1.0)]

    def boom():
        raise RuntimeError("prebuild blew up")

    hub = dsh.EventHub(read_frame_id=lambda: frames[0],
                       read_queue_mtime=lambda: None, on_shot=boom)
    q = hub.register()
    hub.poll_once()                     # seed
    frames[0] = ("1", 2.0)
    evs = hub.poll_once()              # must not raise
    assert evs == [{"topic": "shot", "frame": "1"}]
    assert q.get_nowait() == {"topic": "shot", "frame": "1"}


def test_prebuild_populates_default_key_cache(monkeypatch):
    """_prebuild_default_fragments builds every group figure for the current
    frame into _LIVE_FIG_CACHE under the DEFAULT arg key, so a default-arg
    request is a pure cache read. Stub the data + builder so it's file-free."""
    monkeypatch.setattr(dsh, "_read_data", lambda: {"_write_seq": 7})
    built = []
    monkeypatch.setattr(dsh, "_build_fig_json",
                        lambda d, name, **kw: built.append(name) or ('{"n":"%s"}' % name))
    dsh._LIVE_FIG_CACHE["key"] = None
    dsh._LIVE_FIG_CACHE["frag"] = {}
    dsh._prebuild_default_fragments()
    assert dsh._LIVE_FIG_CACHE["key"] == (7,) + dsh._LIVE_FIG_DEFAULT_ARGS
    assert set(dsh._LIVE_FIG_CACHE["frag"].keys()) == set(dsh._LIVE_FIG_NAMES)
    assert set(built) == set(dsh._LIVE_FIG_NAMES)


def test_prebuild_fills_gaps_without_clobbering(monkeypatch):
    """If a request already opened this frame's default key and built some
    fragments, the prebuild fills only the MISSING names (identical data) rather
    than discarding the request's work."""
    monkeypatch.setattr(dsh, "_read_data", lambda: {"_write_seq": 3})
    monkeypatch.setattr(dsh, "_build_fig_json",
                        lambda d, name, **kw: "PREBUILT:%s" % name)
    key = (3,) + dsh._LIVE_FIG_DEFAULT_ARGS
    # A request already installed the key + one hand-built fragment.
    dsh._LIVE_FIG_CACHE["key"] = key
    dsh._LIVE_FIG_CACHE["frag"] = {"array": "REQUEST_BUILT"}
    dsh._prebuild_default_fragments()
    frag = dsh._LIVE_FIG_CACHE["frag"]
    assert frag["array"] == "REQUEST_BUILT"                 # not clobbered
    assert frag["load"] == "PREBUILT:load"                  # gap filled
    assert set(frag.keys()) == set(dsh._LIVE_FIG_NAMES)


def test_prebuild_noop_without_write_seq(monkeypatch):
    """No frame written yet (_write_seq is None) -> prebuild is a safe no-op."""
    monkeypatch.setattr(dsh, "_read_data", lambda: {})
    dsh._LIVE_FIG_CACHE["key"] = "sentinel"
    dsh._LIVE_FIG_CACHE["frag"] = {}
    dsh._prebuild_default_fragments()
    assert dsh._LIVE_FIG_CACHE["key"] == "sentinel"         # unchanged
