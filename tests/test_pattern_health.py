"""Unit tests for DataManager._compute_pattern_health -- the Live "phase"
status-chip logic that warns when a loading phase file is missing on the SLM
server or its pattern has no expConfig (ByPattern) entry.

Pure aggregation logic: we bypass the heavy DataManager.__init__ (disk reads,
camera config) with ``object.__new__`` and drive the method with canned
per-pattern phase statuses + an embedded expConfig snapshot. No SLM / no disk
(``_probe_phase_status`` is never reached because every status is pre-set).
"""

import pytest

from yb_analysis.acquisition.data_manager import DataManager


def _spec(name):
    return {'name': name, 'base_phase_path': f'phase/{name}.pt'}


def _health(specs, phase_status, by_pattern=None):
    dm = object.__new__(DataManager)
    dm.config = ({'expConfig': {'ByPattern': by_pattern}}
                 if by_pattern is not None else {})
    dm._pattern_phase_status = dict(phase_status)
    dm._pattern_health = None
    dm._image_pattern_specs = lambda: specs
    dm._compute_pattern_health()
    return dm._pattern_health


def test_no_loading_pattern_is_none():
    assert _health(None, {}) is None
    assert _health([], {}) is None


def test_missing_phase_is_fail():
    h = _health([_spec('typo')], {'typo': 'missing'})
    assert h['state'] == 'fail'
    assert h['phase_missing'] == ['typo']
    assert 'typo' in h['reason']


def test_all_present_is_ok():
    h = _health([_spec('good')], {'good': 'ok'})
    assert h['state'] == 'ok'
    assert h['phase_missing'] == [] and h['no_expconfig'] == []


def test_unreachable_is_warn():
    h = _health([_spec('p')], {'p': 'unreachable'})
    assert h['state'] == 'warn'
    assert h['unreachable'] == ['p']


def test_no_expconfig_warns_only_when_table_populated():
    # ByPattern empty -> the per-array overlay system is unused -> no warning.
    h = _health([_spec('p')], {'p': 'ok'}, by_pattern={})
    assert h['state'] == 'ok'
    # ByPattern in use but this pattern is absent from it -> likely a typo -> warn.
    h = _health([_spec('p')], {'p': 'ok'}, by_pattern={'other': {'Init': {}}})
    assert h['state'] == 'warn'
    assert h['no_expconfig'] == ['p']


def test_present_pattern_with_expconfig_is_ok():
    h = _health([_spec('p')], {'p': 'ok'}, by_pattern={'p': {'Init': {}}})
    assert h['state'] == 'ok'


def test_missing_phase_beats_no_expconfig():
    h = _health([_spec('typo')], {'typo': 'missing'}, by_pattern={'other': {}})
    assert h['state'] == 'fail'


def test_duplicate_pattern_deduped():
    # init == target frame: the same name appears twice but is reported once.
    h = _health([_spec('typo'), _spec('typo')], {'typo': 'missing'})
    assert h['phase_missing'] == ['typo']
    assert list(h['patterns'].keys()) == ['typo']
