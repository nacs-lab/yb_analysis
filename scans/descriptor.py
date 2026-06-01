"""Scan descriptor validation.

Validates a Python dict against ``descriptor.schema.json`` before
serialization. We deliberately keep this lightweight (no jsonschema
dependency required at runtime) -- the validation is structural, not
semantic; the MATLAB-side dispatcher does its own deeper checks before
it touches the ScanGroup.

If ``jsonschema`` is importable, we use it for richer error messages;
otherwise we fall back to hand-rolled checks that cover the same shape.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'descriptor.schema.json')
_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


class DescriptorError(ValueError):
    """Raised when a descriptor dict doesn't conform to the schema."""


def validate_descriptor(desc: Mapping[str, Any]) -> None:
    """Raise :class:`DescriptorError` if `desc` is invalid.

    Uses hand-rolled checks (defined below) so the error messages are
    consistent and pinpoint the offending field. The companion
    ``descriptor.schema.json`` is JSON-Schema documentation for external
    tooling (editors, generators) -- the runtime validator does not load
    it directly.
    """
    if not isinstance(desc, Mapping):
        raise DescriptorError(
            f"descriptor must be a mapping, got {type(desc).__name__}")
    _validate_handrolled(desc)


# ---------------------------------------------------------------------------
# Hand-rolled validation (when jsonschema isn't installed). Keeps the
# package functional in stripped-down conda envs while preserving the
# error messages that matter most to the caller.
# ---------------------------------------------------------------------------

def _validate_handrolled(desc: Mapping[str, Any]) -> None:
    sv = desc.get('schema_version', SCHEMA_VERSION)
    if sv != SCHEMA_VERSION:
        raise DescriptorError(
            f"schema_version must be {SCHEMA_VERSION}, got {sv!r}")

    if 'seq' not in desc:
        raise DescriptorError("descriptor must set 'seq'")
    _check_seq(desc['seq'])

    if 'params' in desc:
        _check_paths(desc['params'], 'params')
    if 'runp' in desc:
        _check_paths(desc['runp'], 'runp')

    if 'opts' in desc:
        _check_opts(desc['opts'])

    if 'label' in desc and not isinstance(desc['label'], str):
        raise DescriptorError(
            f"label must be a string, got {type(desc['label']).__name__}")


def _check_seq(seq: Any) -> None:
    if isinstance(seq, str):
        if seq == 'auto':
            return   # 'auto' is reserved -- MATLAB-side raises clearly
        if not _IDENT_RE.match(seq):
            raise DescriptorError(
                f"seq must be a MATLAB identifier or 'auto', got {seq!r}")
        return
    if isinstance(seq, Mapping):
        if '@' not in seq:
            raise DescriptorError(
                f"seq object must have '@', got keys={list(seq.keys())}")
        fn = seq['@']
        if not isinstance(fn, str) or not _IDENT_RE.match(fn):
            raise DescriptorError(
                f"seq['@'] must be a MATLAB identifier, got {fn!r}")
        return
    raise DescriptorError(
        f"seq must be a string or {{'@': 'Name'}}, got {type(seq).__name__}")


def _check_paths(section: Any, section_name: str) -> None:
    if not isinstance(section, Mapping):
        raise DescriptorError(
            f"{section_name} must be a mapping, got "
            f"{type(section).__name__}")
    for key, val in section.items():
        if not isinstance(key, str) or not key:
            raise DescriptorError(
                f"{section_name} key must be a non-empty string, got {key!r}")
        for part in key.split('.'):
            if not _IDENT_RE.match(part):
                raise DescriptorError(
                    f"{section_name}['{key}']: '{part}' is not a valid "
                    f"MATLAB identifier")
        _check_param_value(val, f"{section_name}['{key}']")


def _check_param_value(val: Any, where: str) -> None:
    if val is None or isinstance(val, (int, float, bool, str)):
        return
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        # Numeric vector OR cell-of-string -- enforce homogeneous type
        if not val:
            return
        first = val[0]
        if isinstance(first, str):
            if not all(isinstance(x, str) for x in val):
                raise DescriptorError(
                    f"{where}: mixed types in string array")
            return
        if isinstance(first, (int, float, bool)):
            if not all(isinstance(x, (int, float, bool)) for x in val):
                raise DescriptorError(
                    f"{where}: mixed types in numeric array")
            return
        raise DescriptorError(
            f"{where}: array elements must be number or string, got "
            f"{type(first).__name__}")
    if isinstance(val, Mapping):
        if '@' in val:
            fn = val['@']
            if not isinstance(fn, str) or not _IDENT_RE.match(fn):
                raise DescriptorError(
                    f"{where}: '@' must be a MATLAB identifier, got {fn!r}")
            return
        if 'scan' in val:
            _check_sweep(val, where)
            return
        raise DescriptorError(
            f"{where}: unrecognized object value (keys={list(val.keys())})")
    raise DescriptorError(
        f"{where}: unsupported value type {type(val).__name__}")


def _check_sweep(sw: Mapping[str, Any], where: str) -> None:
    dim = sw.get('scan')
    if not isinstance(dim, int) or dim < 1:
        raise DescriptorError(
            f"{where}: sweep 'scan' must be a positive int, got {dim!r}")
    kinds = [k for k in ('linspace', 'logspace', 'values') if k in sw]
    if len(kinds) != 1:
        raise DescriptorError(
            f"{where}: sweep must set exactly one of "
            f"{{linspace, logspace, values}}, got {kinds}")
    kind = kinds[0]
    arr = sw[kind]
    if not isinstance(arr, Sequence) or isinstance(arr, (str, bytes)):
        raise DescriptorError(
            f"{where}: sweep '{kind}' must be an array, got "
            f"{type(arr).__name__}")
    if kind in ('linspace', 'logspace'):
        if len(arr) != 3:
            raise DescriptorError(
                f"{where}: '{kind}' must be a 3-element [start,stop,n], "
                f"got len={len(arr)}")
        if not all(isinstance(x, (int, float)) for x in arr):
            raise DescriptorError(
                f"{where}: '{kind}' elements must be numeric")
        if arr[2] < 1:
            raise DescriptorError(
                f"{where}: '{kind}' n must be >= 1, got {arr[2]}")
    else:   # values
        if not arr:
            raise DescriptorError(
                f"{where}: sweep 'values' must be non-empty")
        # Type-homogeneity already checked above via _check_param_value
        # on the parent; for sweep's nested array we just re-check:
        first = arr[0]
        if isinstance(first, str):
            if not all(isinstance(x, str) for x in arr):
                raise DescriptorError(
                    f"{where}: mixed types in 'values'")
        elif isinstance(first, (int, float, bool)):
            if not all(isinstance(x, (int, float, bool)) for x in arr):
                raise DescriptorError(
                    f"{where}: mixed types in 'values'")
        else:
            raise DescriptorError(
                f"{where}: 'values' elements must be number or string")


def _check_opts(opts: Any) -> None:
    if not isinstance(opts, Sequence) or isinstance(opts, (str, bytes)):
        raise DescriptorError(
            f"opts must be a list, got {type(opts).__name__}")
    for i, entry in enumerate(opts):
        if (not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes))
                or len(entry) != 2):
            raise DescriptorError(
                f"opts[{i}]: must be a [key, value] pair")
        if not isinstance(entry[0], str):
            raise DescriptorError(
                f"opts[{i}]: key must be a string, got "
                f"{type(entry[0]).__name__}")
