"""Programmatic scan submission from Python (Phase 3).

Submit any MATLAB scan to the SequenceRunner without editing a `.m`
file: build a JSON descriptor in Python, send it via
:func:`submit_scan`, and the MATLAB-side ``dispatch_descriptor.m`` will
construct a fresh ``ScanGroup`` from it and feed the resulting payload
through the existing job queue.

Quick start::

    from yb_analysis.scans import submit_scan, sweep_linspace

    # 1D detuning sweep
    job_id = submit_scan(
        seq='CoolingSeq',
        params={'Cooling.Detuning': sweep_linspace(20e6, 30e6, 21)},
        runp={'NumPerGroup': 4000, 'Scramble': True})

The legacy MATLAB-editor workflow (``>> CoolingScan`` from MATLAB) keeps
working unchanged -- this is an additional input path, not a replacement.

See ``yb_analysis/scans/descriptor.schema.json`` for the canonical
descriptor format; ``client.submit_scan`` validates against it.
"""

from yb_analysis.scans.client import (
    submit_scan,
    list_jobs,
    cancel,
    move,
)
from yb_analysis.scans.convenience import (
    sweep_linspace,
    sweep_logspace,
    sweep_values,
    func_handle,
)
from yb_analysis.scans.descriptor import (
    SCHEMA_VERSION,
    validate_descriptor,
)

__all__ = [
    'submit_scan',
    'list_jobs',
    'cancel',
    'move',
    'sweep_linspace',
    'sweep_logspace',
    'sweep_values',
    'func_handle',
    'SCHEMA_VERSION',
    'validate_descriptor',
]
