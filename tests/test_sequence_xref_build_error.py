"""Tests for surfacing xref-build (provenance producer) failures.

The Sequence-tab auto-build spawns ``provenance_scan.py`` in the background. Before this,
its output went to DEVNULL, so a producer crash (e.g. a config error) was invisible -- the
param<->channel map just silently never appeared, and the failing build was re-spawned on
every poll. These pin the harvest/surface/no-respawn behavior.

Run in the yb_analysis env:
    C:/Users/Ybtweezer-PC2/anaconda3/envs/yb_analysis/python.exe -m pytest \
        yb_analysis/tests/test_sequence_xref_build_error.py -v
"""
import os

from yb_analysis.plotting import dashboard as dsh


class _FakeProc:
    def __init__(self, returncode):
        self.returncode = returncode


def _write_log(tmp_path, text):
    p = tmp_path / "build.log"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_harvest_records_error_from_failed_result(tmp_path):
    key = r"C:\some\scan\A"
    dsh._XREF_BUILD_ERRORS.pop(key, None)
    log = _write_log(tmp_path,
                     'some noise\n'
                     'XREF_RESULT:{"ok": false, "error": "NameError: name \'c\' is not defined"}\n')
    dsh._harvest_xref_build(key, _FakeProc(1), log)
    assert "NameError" in dsh._XREF_BUILD_ERRORS[key]
    assert not os.path.exists(log)                 # temp log cleaned up
    assert dsh._xref_build_error(key) == dsh._XREF_BUILD_ERRORS[key]


def test_harvest_clears_error_on_success(tmp_path):
    key = r"C:\some\scan\B"
    dsh._XREF_BUILD_ERRORS[key] = "stale error from a previous run"
    log = _write_log(tmp_path, 'XREF_RESULT:{"ok": true, "n_seq": 41}\n')
    dsh._harvest_xref_build(key, _FakeProc(0), log)
    assert key not in dsh._XREF_BUILD_ERRORS         # success clears the prior error
    assert dsh._xref_build_error(key) is None


def test_harvest_uses_last_line_when_no_result_and_nonzero_exit(tmp_path):
    key = r"C:\some\scan\C"
    dsh._XREF_BUILD_ERRORS.pop(key, None)
    log = _write_log(tmp_path, 'Traceback (most recent call last):\nImportError: boom\n')
    dsh._harvest_xref_build(key, _FakeProc(1), log)
    assert dsh._XREF_BUILD_ERRORS[key] == "ImportError: boom"


def test_harvest_clean_exit_no_result_is_not_an_error(tmp_path):
    key = r"C:\some\scan\D"
    dsh._XREF_BUILD_ERRORS.pop(key, None)
    log = _write_log(tmp_path, 'nothing useful here\n')
    dsh._harvest_xref_build(key, _FakeProc(0), log)   # rc 0, no XREF_RESULT -> not an error
    assert key not in dsh._XREF_BUILD_ERRORS


def test_xref_build_error_none_for_empty_base():
    assert dsh._xref_build_error(None) is None
    assert dsh._xref_build_error("") is None
