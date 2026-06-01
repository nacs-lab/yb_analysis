"""Sequence catalog: introspects matlab_new/YbSeqs and matlab_new/YbSteps
to surface each available Seq with the params it accepts.

Used by the dashboard Queue tab to give the operator a picker: choose a
Seq, see the params it exposes (with their default expressions), edit
them, submit.

Static-source analysis — no MATLAB runtime needed. Heuristic regex
parsing of the `.m` files; intentionally conservative (we'd rather
miss a param than invent one). Two passes:

1. **Step pass**: for each ``YbSteps/<Step>.m``, find every
   ``g.<Field>(<default_expr>)`` access. Each match becomes a
   ``Param(field=..., default=...)``. The default expression is kept
   verbatim (e.g. ``Consts().Cool556.Time``) so the dashboard can
   display it as a hint.
2. **Seq pass**: for each ``YbSeqs/<Seq>.m``, find every
   ``s.addStep(@<Step>, s.C.<Namespace>)``. Each match means the seq
   exposes ``<Namespace>.<param>`` for every param the step reads.
   Composes a flat list of ``(path, step, default_expr)`` tuples.

Cached in-memory after first read. Refresh by calling
``invalidate_cache()`` (the dashboard offers a "Refresh catalog"
button).

Public API::

    from yb_analysis.scans.seq_catalog import (
        list_seqs, get_seq, list_steps, get_step,
    )

    seqs = list_seqs()                 # ['CoolingSeq', 'RamseySeq', ...]
    seq = get_seq('CoolingSeq')        # dict with file, steps, params, ...
    print(seq['params'])               # [{path, step, default}]
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Match `s.addStep(@StepName, s.C.Namespace)` patterns. Tolerant of
# whitespace and additional varargs.
_ADDSTEP_RE = re.compile(
    r"s\s*\.\s*addStep\(\s*@(\w+)\s*,\s*s\s*\.\s*C\s*\.\s*(\w+)"
)
# Match `g.field(default_expr)` patterns. Keep balanced by walking
# parens manually below — regex alone can't handle nested expressions.
_GFIELD_HEAD = re.compile(r"\bg\s*\.\s*(\w+)\s*\(")
# Match `g.field` (no parens — bare access). Conservative: only when
# followed by `.` (sub-access) or assignment/operator. Skipped from
# the param list since it indicates internal use, not a leaf param.
_GFIELD_BARE = re.compile(r"\bg\s*\.\s*(\w+)(?!\s*\()")
# Function signature line so we can find the seq's main function.
_FN_RE = re.compile(r"^\s*function\s+(?:[\w,\s\[\]\.]+=\s*)?(\w+)\s*\(", re.MULTILINE)

# Default scan-config knobs that every seq accepts via the standard
# `g.runp().<field>` mechanism. We surface these as a fixed "runp" tier
# in every seq's catalog entry so the dashboard form always shows them.
_DEFAULT_RUNP_PARAMS: List[Dict[str, str]] = [
    {'field': 'NumPerGroup', 'default': '4000',
     'comment': 'Number of shots per scan point (after Scramble).'},
    {'field': 'NumImages',   'default': '2',
     'comment': '1 = loading only, 2 = survival.'},
    {'field': 'Scramble',    'default': 'true',
     'comment': 'Randomize scan-point order.'},
    {'field': 'isInit',      'default': 'false',
     'comment': 'True = histogram-initialization scan (no save).'},
    {'field': 'isHC',        'default': 'false',
     'comment': 'True = HCImage external capture, False = IMAQ camera.'},
    {'field': 'roi',         'default': 'Consts().Orca.ROI',
     'comment': 'Camera ROI [x y w h]; defaults to current runner ROI.'},
]


# ---------------------------------------------------------------------------
# Cache + filesystem
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_CACHE: Dict[str, dict] = {}    # 'steps' | 'seqs' -> {name -> entry}
_CACHE_VERSION = 0


def _matlab_root() -> Path:
    """Locate matlab_new on disk. Prefer the relative path from this
    file (deploy-stable); fall back to an env override."""
    here = Path(__file__).resolve().parent
    candidate = here.parents[1] / 'matlab_new'   # yb_analysis/scans/.. -> repo
    if candidate.is_dir():
        return candidate
    env = os.environ.get('YB_MATLAB_ROOT')
    if env and Path(env).is_dir():
        return Path(env)
    raise RuntimeError(
        f"seq_catalog: could not locate matlab_new (tried {candidate}, $YB_MATLAB_ROOT)")


def invalidate_cache() -> None:
    """Drop the cached catalog so the next access re-reads the .m files.
    Cheap (~10 ms for a full re-scan); the dashboard offers a button to
    trigger it after a researcher edits a Step or Seq."""
    global _CACHE_VERSION
    with _LOCK:
        _CACHE.clear()
        _CACHE_VERSION += 1


def cache_version() -> int:
    return _CACHE_VERSION


# ---------------------------------------------------------------------------
# Step pass
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def _strip_comments(src: str) -> str:
    """Strip MATLAB line comments (%...) and block comments (%{...%}).
    Conservative — keeps string literals intact and ignores ``...`` line
    continuations (those don't affect field-access scanning)."""
    out = []
    in_block = False
    for line in src.splitlines():
        ls = line.lstrip()
        if in_block:
            if ls.startswith('%}'):
                in_block = False
            continue
        if ls.startswith('%{'):
            in_block = True
            continue
        # Drop trailing line comment, but only when % is preceded by
        # whitespace (so MATLAB's `%` inside strings isn't stripped).
        # MATLAB strings use single quotes, doubled to escape; we keep
        # a simple state machine.
        i, n = 0, len(line)
        in_str = False
        truncated = n
        while i < n:
            c = line[i]
            if c == "'" and (i == 0 or not in_str
                             or (i + 1 < n and line[i + 1] != "'")):
                # Heuristic: single quote toggles string mode unless
                # immediately followed by another quote (escape).
                in_str = not in_str
                i += 1
                continue
            if c == '%' and not in_str:
                truncated = i
                break
            i += 1
        out.append(line[:truncated])
    return '\n'.join(out)


def _extract_step_params(src: str) -> List[Dict[str, str]]:
    """Pull every ``g.<field>(<default>)`` from a step .m source."""
    src = _strip_comments(src)
    params: Dict[str, str] = {}    # field -> default_expr (first wins)
    for m in _GFIELD_HEAD.finditer(src):
        field = m.group(1)
        # Skip known method names that aren't params (.runp(), .scan()
        # access by ScanGroup) and the rare g.foo where 'foo' is a
        # MATLAB function rather than a leaf knob.
        if field in ('runp', 'scan', 'getseq', 'nseq'):
            continue
        # Walk the parens to find the matching close.
        i = m.end()
        depth = 1
        n = len(src)
        while i < n and depth > 0:
            c = src[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            i += 1
        if depth != 0:
            continue
        default_expr = src[m.end():i - 1].strip()
        # Preserve order of first appearance.
        if field not in params:
            params[field] = default_expr
    return [{'field': k, 'default': v} for k, v in params.items()]


def _scan_steps_dir() -> Dict[str, dict]:
    steps_dir = _matlab_root() / 'YbSteps'
    if not steps_dir.is_dir():
        return {}
    out: Dict[str, dict] = {}
    for path in sorted(steps_dir.glob('*.m')):
        src = _read_text(path)
        # Function name from the file (typically same as filename stem).
        m = _FN_RE.search(src)
        name = m.group(1) if m else path.stem
        params = _extract_step_params(src)
        out[name] = {
            'name': name,
            'file': str(path.relative_to(_matlab_root().parent)),
            'params': params,
        }
    return out


# ---------------------------------------------------------------------------
# Seq pass
# ---------------------------------------------------------------------------

def _extract_seq_steps(src: str) -> List[Dict[str, str]]:
    """Pull every ``s.addStep(@StepFn, s.C.Namespace)`` in source order."""
    src = _strip_comments(src)
    out = []
    for m in _ADDSTEP_RE.finditer(src):
        out.append({'step': m.group(1), 'namespace': m.group(2)})
    return out


def _compose_seq_params(seq_steps: List[Dict[str, str]],
                        steps_catalog: Dict[str, dict]) -> List[Dict[str, str]]:
    """Walk the seq's addStep entries, looking up each step's params
    and prefixing with the namespace."""
    out: List[Dict[str, str]] = []
    seen = set()
    for entry in seq_steps:
        step_name = entry['step']
        namespace = entry['namespace']
        step = steps_catalog.get(step_name)
        if not step:
            continue
        for p in step['params']:
            path = f"{namespace}.{p['field']}"
            if path in seen:
                continue
            seen.add(path)
            out.append({
                'path': path,
                'step': step_name,
                'default': p['default'],
            })
    return out


def _scan_seqs_dir(steps_catalog: Dict[str, dict]) -> Dict[str, dict]:
    seqs_dir = _matlab_root() / 'YbSeqs'
    if not seqs_dir.is_dir():
        return {}
    out: Dict[str, dict] = {}
    for path in sorted(seqs_dir.glob('*.m')):
        src = _read_text(path)
        m = _FN_RE.search(src)
        name = m.group(1) if m else path.stem
        seq_steps = _extract_seq_steps(src)
        params = _compose_seq_params(seq_steps, steps_catalog)
        out[name] = {
            'name': name,
            'file': str(path.relative_to(_matlab_root().parent)),
            'steps': seq_steps,
            'params': params,
            'runp': list(_DEFAULT_RUNP_PARAMS),
        }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ensure_loaded() -> None:
    with _LOCK:
        if 'seqs' in _CACHE and 'steps' in _CACHE:
            return
        try:
            steps = _scan_steps_dir()
            seqs = _scan_seqs_dir(steps)
            _CACHE['steps'] = steps
            _CACHE['seqs'] = seqs
        except Exception:
            logger.exception('seq_catalog: catalog build failed')
            _CACHE.setdefault('steps', {})
            _CACHE.setdefault('seqs', {})


def list_seqs(*, summary: bool = True) -> List[dict]:
    """Return a list of every available Seq function (sorted).

    Each entry is ``{name, file, n_steps, n_params}`` when
    ``summary=True``; full entry with params + steps when False.
    """
    _ensure_loaded()
    out = []
    for name in sorted(_CACHE['seqs']):
        entry = _CACHE['seqs'][name]
        if summary:
            out.append({
                'name':     entry['name'],
                'file':     entry['file'],
                'n_steps':  len(entry.get('steps') or []),
                'n_params': len(entry.get('params') or []),
            })
        else:
            out.append(dict(entry))
    return out


def get_seq(name: str) -> Optional[dict]:
    _ensure_loaded()
    return _CACHE['seqs'].get(name)


def list_steps(*, summary: bool = True) -> List[dict]:
    _ensure_loaded()
    out = []
    for k in sorted(_CACHE['steps']):
        entry = _CACHE['steps'][k]
        if summary:
            out.append({
                'name': entry['name'],
                'file': entry['file'],
                'n_params': len(entry.get('params') or []),
            })
        else:
            out.append(dict(entry))
    return out


def get_step(name: str) -> Optional[dict]:
    _ensure_loaded()
    return _CACHE['steps'].get(name)


def build_descriptor_template(seq_name: str,
                              swept_path: Optional[str] = None) -> dict:
    """Return a starter JSON descriptor for a given seq.

    Pre-fills the ``seq`` field, leaves every param at its default
    (no value -> uses ``Consts()``), and optionally adds a placeholder
    sweep on ``swept_path``.  The dashboard "fill template" button uses
    this; the user then edits the JSON before submitting.
    """
    desc: dict = {
        'schema_version': 1,
        'seq': seq_name,
        'params': {},
        'runp':   {
            'NumPerGroup': 4000,
            'NumImages':   2,
            'Scramble':    True,
        },
    }
    if swept_path:
        # Example sweep — operator-editable.
        desc['params'][swept_path] = {
            'scan': 1,
            'linspace': [0.0, 1.0, 11],
        }
    return desc
