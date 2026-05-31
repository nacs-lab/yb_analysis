"""On-demand source retrieval for analysts working with synced scans.

Once `slm_sync.sync_scan` has saved `<scan_dir>/slm_code.json` (just the
hashes — no source bytes), an analyst can later pull the actual source
of any tracked file via this module. The bytes come over the wire from
the SLM PC's `/slm/code/by_hash/{sha256}` endpoint.

Example::

    src = get_protocol_source(
        scan_id='20260528123045',
        rel_path='SLMnet/src/slmnet/experimental/tools/rearrange_protocols.py')
    print(src)

If the manifest isn't available locally (e.g. the scan synced before
Phase 2 shipped, or the analyst is working from a path without an
``slm_code.json`` sidecar), the manifest is fetched from the SLM PC
on demand.
"""

import json
import logging
from pathlib import Path

from yb_analysis.slm_sync.client import SlmSyncClient
from yb_analysis.slm_sync.sync import CODE_JSON

logger = logging.getLogger(__name__)


# Default file when the caller just wants "the rearrangement protocol".
_DEFAULT_PROTOCOL_REL = (
    'SLMnet/src/slmnet/experimental/tools/rearrange_protocols.py')


def get_protocol_source(scan_id, rel_path=_DEFAULT_PROTOCOL_REL, *,
                        scan_dir=None, client=None):
    """Return the source string of `rel_path` as it was for `scan_id`.

    Args:
        scan_id: 14-digit MATLAB scan_id.
        rel_path: Path relative to the slmnet repo root. Defaults to the
                  rearrangement protocol file (most common query).
        scan_dir: Optional local scan directory. If provided and
                  `slm_code.json` exists there, the hash is read from
                  the local sidecar (no second SLM PC roundtrip).
        client:   Optional SlmSyncClient.

    Returns:
        The file's source as a string, or ``None`` if the snapshot
        couldn't be located.
    """
    if client is None:
        client = SlmSyncClient()
    sha = _resolve_hash(scan_id, rel_path, scan_dir, client)
    if sha is None:
        logger.info(
            'get_protocol_source: no hash for %s in scan_id=%s',
            rel_path, scan_id)
        return None
    src = client.get_code_by_hash(sha)
    if src is None:
        logger.warning(
            'get_protocol_source: SLM has no blob for sha %s (scan_id=%s)',
            sha, scan_id)
    return src


def list_tracked_files(scan_id, *, scan_dir=None, client=None):
    """Return a list of `(rel_path, sha256)` for every file tracked in
    the code snapshot for ``scan_id``.

    Useful to discover which files are available before retrieving any.
    """
    manifest = _resolve_manifest(scan_id, scan_dir, client or SlmSyncClient())
    if manifest is None:
        return []
    # The manifest's 'files' entries each have {src_rel, leaf, sha256, ...}.
    return [(entry.get('src_rel'), entry.get('sha256'))
            for entry in (manifest.get('files') or [])
            if entry.get('sha256')]


def _resolve_hash(scan_id, rel_path, scan_dir, client):
    manifest = _resolve_manifest(scan_id, scan_dir, client)
    if manifest is None:
        return None
    rel_norm = rel_path.replace('\\', '/')
    for entry in (manifest.get('files') or []):
        if entry.get('src_rel') == rel_norm:
            return entry.get('sha256')
    return None


def _resolve_manifest(scan_id, scan_dir, client):
    """Read the manifest from a local `slm_code.json` if present;
    fall back to fetching it from the SLM PC."""
    if scan_dir is not None:
        local = Path(scan_dir) / CODE_JSON
        if local.exists():
            try:
                payload = json.loads(local.read_text(encoding='utf-8'))
                m = payload.get('manifest')
                if isinstance(m, dict):
                    return m
            except (OSError, json.JSONDecodeError):
                pass  # fall through to remote
    remote = client.get_code_manifest(scan_id)
    if remote is None:
        return None
    return remote.get('manifest')
