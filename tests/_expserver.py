"""Locate the ExptServer.py the protocol tests run against.

ExptServer is owned by the backend trees, not yb_analysis. The protocol tests
exercise the queue/descriptor wire format against a real server. Prefer the
live pyctrl runtime, fall back to the retired matlab_new server, then a
``YB_EXPTSERVER_DIR`` env override.

Returns ``None`` when none is present (e.g. yb_analysis checked out
standalone), in which case the ExptServer-dependent tests skip via
``pytest.importorskip("ExptServer")``.
"""
import os


def expserver_dir():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    candidates = [
        os.path.join(repo_root, 'pyctrl', 'YbExptCtrl'),       # live runtime
        os.path.join(repo_root, 'matlab_new', 'YbExptCtrl'),   # retired backup
        os.environ.get('YB_EXPTSERVER_DIR', ''),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, 'ExptServer.py')):
            return c
    return None
