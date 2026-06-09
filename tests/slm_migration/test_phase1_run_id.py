"""Phase 1 lab-side tests — MATLAB code shape + body capture via fake SLM.

The full SLM-side test suite is at
`SLMnet/tests/migration/test_phase1_diag_ledger.py` and runs on the SLM PC
after deploy. This file holds the **lab-PC half**:

1. **Static analysis of MATLAB source** — confirm `RearrangeCommSeq.m` and
   `SLMnet/src/slmnet/experimental/slm_client.m` now thread `scan_id` /
   `seq_id` through the SLM call sites. No MATLAB runtime required.

2. **Fake-SLM body capture** — exercise the `FakeSlmServer` POST endpoints
   that the future MATLAB integration tests will rely on (a Python
   client mimicking what MATLAB sends, asserts the body shape matches).

3. **MATLAB-integration tests** (`@requires_matlab`) — actually invoke
   `matlab -batch` against a tiny fixture and assert what hits the fake
   server. Skipped automatically when MATLAB isn't installed or the env
   var `YB_RUN_MATLAB_TESTS=1` is unset (Matlab startup is ~5–10 s, so
   default CI shouldn't pay it).

Run as:
    python -m yb_analysis.tests.slm_migration.test_phase1_run_id
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from yb_analysis.tests.slm_migration.fake_slm_server import FakeSlmServer


# Repository roots used by the static-analysis tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REARRANGE = _REPO_ROOT / 'matlab_new' / 'YbSeqs' / 'RearrangeCommSeq.m'
_SLM_CLIENT = _REPO_ROOT / 'SLMnet' / 'src' / 'slmnet' / 'experimental' / 'slm_client.m'

# The Tier-1 tests statically grep MATLAB sources that live in sibling trees
# (matlab_new, SLMnet). In a standalone yb_analysis checkout those are absent;
# skip rather than error.
_skip_no_rearrange = pytest.mark.skipif(
    not _REARRANGE.exists(),
    reason="RearrangeCommSeq.m absent (matlab_new tree not alongside)")
_skip_no_slm_client = pytest.mark.skipif(
    not _SLM_CLIENT.exists(),
    reason="slm_client.m absent (SLMnet tree not alongside)")


# ---------------------------------------------------------------------------
# Tier 1 — static analysis of MATLAB source
# ---------------------------------------------------------------------------
# We verify the *.m files contain the Phase 1 code paths without booting
# MATLAB. Cheap and catches the "I edited but forgot to save" class of bug
# that's easy to miss when the actual MATLAB integration test is gated.

@_skip_no_rearrange
def test_rearrange_call_site_has_runid_opts():
    """RearrangeCommSeq.m::hand_over_slm threads runid_opts into rearrange()."""
    src = _REARRANGE.read_text(encoding='utf-8', errors='replace')
    # Both call sites use the same helper-built cell.
    assert 'build_runid_opts(s1)' in src, (
        'RearrangeCommSeq.m should call build_runid_opts(s1) to construct '
        'the {scan_id, ..., seq_id, ...} opts.')
    # Pattern check: the rearrange call passes the opts cell-expanded.
    assert re.search(r'slm_c\.rearrange\(bits,\s*runid_opts\{:\}\)', src), (
        'rearrange() call site should pass runid_opts{:}.')


@_skip_no_rearrange
def test_update_rearrange_call_site_has_runid_opts():
    """RearrangeCommSeq.m::post_run threads runid_opts into update_rearrange."""
    src = _REARRANGE.read_text(encoding='utf-8', errors='replace')
    assert re.search(r'slm_c\.update_rearrange\(bits2,\s*runid_opts\{:\}\)', src), (
        'update_rearrange() call site should pass runid_opts{:}.')


@_skip_no_rearrange
def test_build_runid_opts_local_function_present():
    """The helper is defined as a local function at the end of the file."""
    src = _REARRANGE.read_text(encoding='utf-8', errors='replace')
    assert re.search(
        r'function opts = build_runid_opts\(s1\)', src), (
        'build_runid_opts(s1) local function should be defined.')
    # Kill switch via env var.
    assert "getenv('YB_SLM_DISABLE_RUNID')" in src, (
        'build_runid_opts should check YB_SLM_DISABLE_RUNID env var '
        'for the kill switch.')
    # scan_id is sent as a string via %.0f
    assert "sprintf('%.0f', double(s1.G.scan_id))" in src, (
        'scan_id should be serialized as a 14-digit string to avoid '
        'JS Number precision loss in the JSON body.')


@_skip_no_slm_client
def test_slm_client_rearrange_accepts_runid_opts():
    """slm_client.m::rearrange's arguments block lists scan_id + seq_id."""
    src = _SLM_CLIENT.read_text(encoding='utf-8', errors='replace')
    # Find the rearrange method's arguments block.
    m = re.search(
        r'function r = rearrange\(obj, bits, opts\).*?arguments\s+obj.*?end',
        src, re.DOTALL)
    assert m, 'Could not locate slm_client.rearrange arguments block.'
    args_block = m.group(0)
    assert 'opts.scan_id' in args_block, 'rearrange opts.scan_id missing.'
    assert 'opts.seq_id' in args_block, 'rearrange opts.seq_id missing.'
    # And the body merges them in.
    assert 'body.scan_id = char(string(opts.scan_id))' in src
    assert 'body.seq_id = double(opts.seq_id)' in src


@_skip_no_slm_client
def test_slm_client_update_rearrange_accepts_runid_opts():
    """update_rearrange now has an arguments block with scan_id + seq_id."""
    src = _SLM_CLIENT.read_text(encoding='utf-8', errors='replace')
    m = re.search(
        r'function r = update_rearrange\(obj, results, opts\).*?arguments\s+obj.*?end',
        src, re.DOTALL)
    assert m, ('slm_client.update_rearrange should now accept opts; '
               'check the arguments block.')
    args_block = m.group(0)
    assert 'opts.scan_id' in args_block
    assert 'opts.seq_id' in args_block


# ---------------------------------------------------------------------------
# Tier 2 — FakeSlmServer body-capture exercises
# ---------------------------------------------------------------------------
# Confirm the test harness can capture POST bodies and that our expected
# Phase-1 body shape (scan_id + seq_id at the top level) round-trips
# cleanly. This is what the MATLAB integration test will assert against.

def test_fake_slm_captures_rearrange_body():
    """POST /slm/rearrange with scan_id+seq_id → body captured + response OK."""
    with FakeSlmServer() as fake:
        body = {
            'bits': '01010101',
            'scan_id': '20260528120000',
            'seq_id': 42,
        }
        r = requests.post(fake.url + '/slm/rearrange', json=body, timeout=2)
        assert r.status_code == 200
        assert r.json()['ok'] is True
        captured = fake.captured_bodies('rearrange')
        assert len(captured) == 1
        assert captured[0] == body


def test_fake_slm_captures_legacy_body():
    """Pre-Phase-1 MATLAB build sends only {bits}; fake server captures it."""
    with FakeSlmServer() as fake:
        body = {'bits': '11110000'}
        requests.post(fake.url + '/slm/rearrange', json=body, timeout=2)
        captured = fake.captured_bodies('rearrange')
        assert captured == [body]
        # No scan_id / seq_id in the captured body — verifies the
        # backward-compat test surface.
        assert 'scan_id' not in captured[0]
        assert 'seq_id' not in captured[0]


def test_fake_slm_captures_update_rearrange():
    """POST /slm/results captures the new optional scan_id/seq_id too."""
    with FakeSlmServer() as fake:
        body = {'results': '00011000', 'scan_id': 'A', 'seq_id': 7}
        r = requests.post(fake.url + '/slm/results', json=body, timeout=2)
        assert r.status_code == 200
        captured = fake.captured_bodies('results')
        assert captured == [body]


def test_fake_slm_set_post_payload():
    """Tests can override the response body for the POST endpoints."""
    with FakeSlmServer() as fake:
        fake.set_post_payload('rearrange',
                              {'ok': True, 'handoff_idle': True,
                               'handoff_reason': 'queue_empty'})
        r = requests.post(fake.url + '/slm/rearrange',
                          json={'bits': '0' * 8}, timeout=2)
        body = r.json()
        assert body['handoff_idle'] is True
        assert body['handoff_reason'] == 'queue_empty'


def test_fake_slm_clear_captured():
    """clear_captured() drops all stored bodies."""
    with FakeSlmServer() as fake:
        requests.post(fake.url + '/slm/rearrange',
                      json={'bits': '0' * 4}, timeout=2)
        assert len(fake.captured_bodies('rearrange')) == 1
        fake.clear_captured()
        assert fake.captured_bodies('rearrange') == []


# ---------------------------------------------------------------------------
# Tier 3 — MATLAB integration (skipped unless YB_RUN_MATLAB_TESTS=1)
# ---------------------------------------------------------------------------

_MATLAB_TESTS_ENABLED = os.environ.get('YB_RUN_MATLAB_TESTS') == '1'
_MATLAB_EXE = (os.environ.get('YB_MATLAB_EXE')
               or r'C:\Program Files\MATLAB\R2023a\bin\matlab.exe')
_MATLAB_AVAILABLE = _MATLAB_TESTS_ENABLED and Path(_MATLAB_EXE).exists()

pytestmark_matlab = pytest.mark.skipif(
    not _MATLAB_AVAILABLE,
    reason='MATLAB integration tests gated on YB_RUN_MATLAB_TESTS=1 '
           f'+ MATLAB at {_MATLAB_EXE}')


@pytestmark_matlab
def test_matlab_call_signature():
    """Run a tiny MATLAB script that calls slm_client.rearrange with
    scan_id+seq_id; assert the fake server captures the body shape."""
    pytest.skip('Tier-3 MATLAB integration — to be implemented when '
                'a matlab-batch test fixture is in place. See plan §Phase 5 '
                'for full e2e coverage.')


@pytestmark_matlab
def test_no_seq_id_regression():
    """Set YB_SLM_DISABLE_RUNID=1 → MATLAB-side helper returns {} → body
    shape is the legacy {bits} only (no scan_id/seq_id)."""
    pytest.skip('Tier-3 MATLAB integration — to be implemented.')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
