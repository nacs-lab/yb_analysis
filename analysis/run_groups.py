"""Non-destructive run groups (Phase 4 dashboard feature).

A "group" is a named collection of scan_ids that an analyst can analyze
together (mean survival across all members, combined sweep, etc.). The
underlying scan data is NEVER modified — groups live as a single JSON
sidecar at ``<groups_dir>/yb_run_groups.json``:

    {
      "groups": {
        "<group_id>": {
          "name":       "blue lac sweep batch 1",
          "created_iso": "2026-05-31T12:34:56",
          "members": [
            {"scan_id": "20260529025015", "added_iso": "..."},
            ...
          ]
        }
      }
    }

Public API (used by the dashboard /api/runs/groups* endpoints)::

    list_groups()                 -> {group_id: <full group entry>}
    create_group(name)            -> group_id
    delete_group(group_id)        -> bool
    add_member(group_id, scan_id) -> bool
    remove_member(group_id, scan_id) -> bool
    get_group(group_id)           -> entry | None
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from yb_analysis import config as _yb_cfg

logger = logging.getLogger(__name__)


_GROUPS_FILENAME = 'yb_run_groups.json'
_LOCK = threading.Lock()


def _groups_dir() -> Path:
    """Where the groups JSON lives. Prefer the same dir as
    yb_dash_data.pkl (the lab-PC working dir) so it travels with the
    rest of the dashboard's state. Override via $YB_RUN_GROUPS_DIR."""
    env = os.environ.get('YB_RUN_GROUPS_DIR')
    if env:
        return Path(env)
    # Default: PATH_PREFIX/yb_dashboard_state (creates if absent).
    base = Path(_yb_cfg.PATH_PREFIX) / 'yb_dashboard_state'
    return base


def _groups_path() -> Path:
    return _groups_dir() / _GROUPS_FILENAME


def _read() -> dict:
    p = _groups_path()
    if not p.is_file():
        return {'groups': {}}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'groups' not in data:
            return {'groups': {}}
        return data
    except (OSError, json.JSONDecodeError) as ex:
        logger.warning('run_groups: read %s failed: %s', p, ex)
        return {'groups': {}}


def _write(data: dict) -> None:
    p = _groups_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_groups() -> Dict[str, dict]:
    with _LOCK:
        return _read()['groups']


def get_group(group_id: str) -> Optional[dict]:
    with _LOCK:
        return _read()['groups'].get(group_id)


def create_group(name: str) -> str:
    name = (name or '').strip()
    if not name:
        raise ValueError('group name required')
    gid = uuid4().hex[:12]
    with _LOCK:
        data = _read()
        data['groups'][gid] = {
            'id': gid,
            'name': name,
            'created_iso': _now_iso(),
            'members': [],
        }
        _write(data)
    return gid


def delete_group(group_id: str) -> bool:
    with _LOCK:
        data = _read()
        if group_id not in data['groups']:
            return False
        del data['groups'][group_id]
        _write(data)
        return True


def rename_group(group_id: str, new_name: str) -> bool:
    new_name = (new_name or '').strip()
    if not new_name:
        raise ValueError('group name required')
    with _LOCK:
        data = _read()
        g = data['groups'].get(group_id)
        if not g:
            return False
        g['name'] = new_name
        _write(data)
        return True


def add_member(group_id: str, scan_id: str) -> bool:
    scan_id = str(scan_id).strip()
    if not scan_id:
        return False
    with _LOCK:
        data = _read()
        g = data['groups'].get(group_id)
        if not g:
            return False
        if any(m['scan_id'] == scan_id for m in g['members']):
            return True
        g['members'].append({'scan_id': scan_id, 'added_iso': _now_iso()})
        _write(data)
        return True


def remove_member(group_id: str, scan_id: str) -> bool:
    scan_id = str(scan_id).strip()
    with _LOCK:
        data = _read()
        g = data['groups'].get(group_id)
        if not g:
            return False
        before = len(g['members'])
        g['members'] = [m for m in g['members'] if m['scan_id'] != scan_id]
        if len(g['members']) == before:
            return False
        _write(data)
        return True
