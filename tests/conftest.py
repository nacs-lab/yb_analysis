"""Shared fixtures for yb_analysis tests."""
import os
import sys
import tempfile

import pytest

# Make matlab_new/YbExpServer importable so tests can import ExptServer/Client.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_EXPSERVER = os.path.join(_REPO_ROOT, 'matlab_new', 'YbExpServer')
if _EXPSERVER not in sys.path:
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
