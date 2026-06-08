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
