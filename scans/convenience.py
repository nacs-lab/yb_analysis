"""Sweep / value convenience builders for descriptors.

Each helper returns a JSON-serializable dict that the descriptor parser
accepts directly. Use them in ``submit_scan``'s ``params`` / ``runp``::

    submit_scan(
        seq='CoolingSeq',
        params={'Cooling.Detuning': sweep_linspace(20e6, 30e6, 21)},
        runp={'AWGs': ['AWG556']})
"""

from __future__ import annotations

from typing import Sequence


def sweep_linspace(start: float, stop: float, n: int,
                   axis: int = 1) -> dict:
    """Inclusive linspace sweep on ``axis``. Equivalent to MATLAB::

        g().path.scan(axis) = linspace(start, stop, n);
    """
    if n < 1:
        raise ValueError(f"sweep_linspace: n must be >= 1, got {n}")
    if axis < 1:
        raise ValueError(f"sweep_linspace: axis must be >= 1, got {axis}")
    return {'scan': int(axis),
            'linspace': [float(start), float(stop), int(n)]}


def sweep_logspace(start_exp: float, stop_exp: float, n: int,
                   axis: int = 1) -> dict:
    """Logspace sweep on ``axis`` (MATLAB convention: ``start_exp`` and
    ``stop_exp`` are EXPONENTS of 10). Equivalent to::

        g().path.scan(axis) = logspace(start_exp, stop_exp, n);
    """
    if n < 1:
        raise ValueError(f"sweep_logspace: n must be >= 1, got {n}")
    if axis < 1:
        raise ValueError(f"sweep_logspace: axis must be >= 1, got {axis}")
    return {'scan': int(axis),
            'logspace': [float(start_exp), float(stop_exp), int(n)]}


def sweep_values(values: Sequence, axis: int = 1) -> dict:
    """Explicit-list sweep on ``axis``. ``values`` may be numeric or
    string; the dispatcher passes them through verbatim to the resulting
    ScanGroup."""
    if not values:
        raise ValueError("sweep_values: values must be non-empty")
    if axis < 1:
        raise ValueError(f"sweep_values: axis must be >= 1, got {axis}")
    vals = list(values)
    first = vals[0]
    if isinstance(first, str):
        if not all(isinstance(v, str) for v in vals):
            raise ValueError("sweep_values: mixed types in string array")
        cleaned = [str(v) for v in vals]
    elif isinstance(first, (int, float, bool)):
        if not all(isinstance(v, (int, float, bool)) for v in vals):
            raise ValueError("sweep_values: mixed types in numeric array")
        cleaned = [float(v) if not isinstance(v, bool) else bool(v)
                   for v in vals]
    else:
        raise ValueError(
            f"sweep_values: elements must be number or string, got "
            f"{type(first).__name__}")
    return {'scan': int(axis), 'values': cleaned}


_IDENT_RE = __import__('re').compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def func_handle(name: str) -> dict:
    """Wrap a MATLAB function name as a descriptor function-handle value.

    Use this when a parameter expects an actual function handle (e.g.
    ``runp.AlgoCb = func_handle('myRearrangeAlgo')`` -> MATLAB
    ``g.runp().AlgoCb = @myRearrangeAlgo``).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"func_handle: name must be a string, got {name!r}")
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"func_handle: name must be a MATLAB identifier, got {name!r}")
    return {'@': name}
