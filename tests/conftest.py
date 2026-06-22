"""Shared fixtures for yb_analysis tests."""
import importlib
import sys

import pytest

from yb_analysis.tests._expserver import expserver_dir

# Put a backend's ExptServer/ExptClient on sys.path so the protocol tests can
# import it (prefer pyctrl, fall back to matlab_new). When neither backend tree
# is present, those tests skip via pytest.importorskip("ExptServer").
_EXPSERVER = expserver_dir()
if _EXPSERVER and _EXPSERVER not in sys.path:
    sys.path.insert(0, _EXPSERVER)


@pytest.fixture(autouse=True)
def _isolated_queue_file(tmp_path, monkeypatch):
    """Redirect the runner-queue persistence file to a per-test tmp path.

    ExptServer.QUEUE_PATH defaults to the LIVE production queue file. This
    fixture USED to ``os.remove()`` that path before/after every test — so
    running the suite silently wiped the operator's real scan queue + history
    (and, since an idle backend never rewrites the file, the next restart came
    up empty). We now monkeypatch the module constants to a tmp file instead,
    so the suite can never touch the live queue. monkeypatch auto-reverts.
    """
    qp = str(tmp_path / 'runner_queue.json')
    try:
        mod = importlib.import_module('ExptServer')
    except Exception:
        yield                       # no backend on path -> those tests skip anyway
        return
    monkeypatch.setattr(mod, 'QUEUE_PATH', qp, raising=False)
    # Also pin the legacy-migration source so a stray real legacy file can't
    # bleed into a test.
    if hasattr(mod, '_LEGACY_QUEUE_PATH'):
        monkeypatch.setattr(mod, '_LEGACY_QUEUE_PATH', qp, raising=False)
    yield
