"""Append-only, shot-stamped audit log for live self-calibration updates.

Every time the live loop nudges the global affine (positions) or a pattern's
detection thresholds, it appends ONE json line here so the drift can be matched
to a specific shot in a specific run and analysed offline. The persisted
calibration files (``affine_transform.json`` / per-pattern ``threshold.mat``)
hold only the CURRENT value and get overwritten; THIS log is the history.

Layout (under ``<PATH_PREFIX>/yb_dashboard_state/update_logs/``):
  * ``affine.jsonl``            -- one line per affine (translation) update.
  * ``thresholds/<name>.jsonl`` -- one line per per-pattern threshold update.

Each record carries at least ``ts`` (local ISO, ms), ``scan_id`` and ``seq_no``
(the completed-sequence / shot index within that run) so a change is pinned to
an exact shot. Writes are thread-safe and best-effort (never raise into the
acquisition loop).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime

from yb_analysis import config as _cfg

_LOCK = threading.Lock()


def _logs_dir() -> str:
    return os.path.join(_cfg.PATH_PREFIX, 'yb_dashboard_state', 'update_logs')


def append(rel_path: str, record: dict) -> None:
    """Append ``record`` (a dict) as one json line to ``<logs_dir>/<rel_path>``.

    Adds ``ts`` (local ISO, ms precision) when absent. Best-effort: any failure
    is swallowed so logging never disturbs acquisition.
    """
    try:
        rec = dict(record)
        rec.setdefault('ts', datetime.now().isoformat(timespec='milliseconds'))
        p = os.path.join(_logs_dir(), rel_path)
        with _LOCK:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, default=float) + '\n')
    except Exception:  # noqa: BLE001 — audit logging must never raise
        pass


def _digits(v) -> str:
    """Digits-only normalisation so a string ``scan_id`` and an int ``scan_id``
    compare equal regardless of how each side stamped it."""
    return ''.join(ch for ch in str(v) if ch.isdigit())


def read_threshold_records(pattern, scan_id=None, max_lines: int = 50000):
    """Read per-pattern threshold-update records (the full per-site
    ``thresholds`` / ``infidelities`` vectors are retained).

    Returns the parsed records in file (chronological) order. When ``scan_id``
    is given, only records stamped with that run are returned (digit-normalised
    match). Best-effort: returns ``[]`` on any read / parse failure or when the
    log doesn't exist (e.g. a day-folder scan that declared no loading pattern).
    """
    try:
        import yb_analysis.analysis.pattern_registry as reg
        name = reg._sanitize_name(pattern)
    except Exception:  # noqa: BLE001
        name = str(pattern)
    path = os.path.join(_logs_dir(), 'thresholds', '%s.jsonl' % name)
    if not os.path.isfile(path):
        return []
    want = _digits(scan_id) if scan_id is not None else None
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if max_lines and len(lines) > max_lines:
            lines = lines[-max_lines:]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except (ValueError, TypeError):
                continue
            if want is not None and _digits(rec.get('scan_id')) != want:
                continue
            out.append(rec)
    except OSError:
        return []
    return out
