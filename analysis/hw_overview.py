"""Server-side store for the Hardware tab's *Overview* sub-view.

The Overview is a free-form canvas of "tiles", each one a starred view from one
of the hardware sources (SLM / scope / molecube / ...). A tile is
``{source, tab, label}`` plus an optional free-form position (``x``/``y``) and
size (``w``/``h``) -- the operator drags + resizes each tile and both persist. We
store it server-side (not per-browser localStorage) so the layout is shared
across every machine/browser that opens the dashboard -- the same rationale, and
the same JSON-sidecar pattern, as :mod:`yb_analysis.analysis.run_groups`.

File: ``<PATH_PREFIX>/yb_dashboard_state/yb_hw_overview.json`` (override the dir
via ``$YB_RUN_GROUPS_DIR`` -- shared with the other dashboard state). Shape::

    {"tiles": [{"source": "scope", "tab": "169.254.242.0", "label": "trap PD",
                "x": 12, "y": 12, "w": 440, "h": 300},
               {"source": "slm",   "tab": "runs",          "label": "Run Analysis"}]}

The set of *which* tabs exist per source is NOT stored here -- that's discovered
live (the embeds announce themselves over postMessage; molecube is native). This
store only records what the operator starred and in what order.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from yb_analysis import config as _yb_cfg

logger = logging.getLogger(__name__)

_FILENAME = 'yb_hw_overview.json'
_LOCK = threading.Lock()


def _path() -> Path:
    env = os.environ.get('YB_RUN_GROUPS_DIR')
    base = Path(env) if env else (Path(_yb_cfg.PATH_PREFIX) / 'yb_dashboard_state')
    return base / _FILENAME


def _clean_tiles(raw) -> list:
    """Coerce arbitrary input into a list of {source, tab, label} dicts,
    dropping duplicates (same source+tab) while preserving order."""
    out, seen = [], set()
    if not isinstance(raw, list):
        return out
    for t in raw:
        if not isinstance(t, dict):
            continue
        source = str(t.get('source') or '').strip()
        tab = str(t.get('tab') or '').strip()
        if not source or not tab:
            continue
        key = (source, tab)
        if key in seen:
            continue
        seen.add(key)
        rec = {'source': source, 'tab': tab, 'label': str(t.get('label') or tab)}
        # Optional per-tile size (w/h > 0) + free-form position (x/y >= 0) --
        # the Overview tiles are draggable + resizable, both persisted.
        for k in ('w', 'h'):
            v = t.get(k)
            if isinstance(v, (int, float)) and v > 0:
                rec[k] = int(v)
        for k in ('x', 'y'):
            v = t.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                rec[k] = int(v)
        out.append(rec)
    return out


def load() -> list:
    """Return the ordered list of tiles (possibly empty)."""
    with _LOCK:
        p = _path()
        if not p.is_file():
            return []
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return _clean_tiles(data.get('tiles') if isinstance(data, dict) else None)
        except (OSError, json.JSONDecodeError) as ex:
            logger.warning('hw_overview: read %s failed: %s', p, ex)
            return []


def save(tiles) -> list:
    """Persist the (cleaned, de-duplicated) tile list; return what was stored."""
    cleaned = _clean_tiles(tiles)
    with _LOCK:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'tiles': cleaned}, f, indent=2)
        os.replace(tmp, p)
    return cleaned
