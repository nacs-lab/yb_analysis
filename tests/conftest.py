"""Shared fixtures for yb_analysis tests."""
import os
import sys
import tempfile

import pytest

from yb_analysis.tests._expserver import expserver_dir

# Put a backend's ExptServer/ExptClient on sys.path so the protocol tests can
# import it (prefer pyctrl, fall back to matlab_new). When neither backend tree
# is present, those tests skip via pytest.importorskip("ExptServer").
_EXPSERVER = expserver_dir()
if _EXPSERVER and _EXPSERVER not in sys.path:
    sys.path.insert(0, _EXPSERVER)


@pytest.fixture(autouse=True)
def _clean_queue_file():
    """Each test starts with no persisted queue."""
    qp = os.path.join(tempfile.gettempdir(), 'nacsctl', 'runner_queue.json')
    if os.path.exists(qp):
        os.remove(qp)
    yield
    if os.path.exists(qp):
        os.remove(qp)
