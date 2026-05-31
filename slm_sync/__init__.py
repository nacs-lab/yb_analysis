"""Lab-PC sync of SLM-side per-scan diagnostics + code snapshots.

Phase 2 of the lab-PC migration plan: when a scan finishes, pull every
per-shot diag row from the SLM PC's ledger (`/slm/runs/{scan_id}/diag`)
and persist as `<scan_dir>/slm_diag.h5` next to the regular HDF5. Also
pull the code-snapshot manifest (`/slm/runs/{scan_id}/code`) and stash
the hash → blob-path map as `<scan_dir>/slm_code.json` so an analyst can
later retrieve the exact protocol source via `ondemand.get_protocol_source`.

The actual source bytes stay on the SLM PC — only hashes travel by
default. Avoids bulk-copying ~50-100 KB of Python source per scan while
keeping reproducibility intact.

Public API:

    from yb_analysis.slm_sync import sync_scan, get_protocol_source

    # At scan end (called automatically by DataManager.save_data):
    sync_scan(scan_id='20260528123045', scan_dir='D:/.../data_xxx/')

    # Later, in a notebook:
    src = get_protocol_source(scan_id='20260528123045')
    print(src)  # the rearrange_protocols.py that ran for that scan
"""

from yb_analysis.slm_sync.client import SlmSyncClient
from yb_analysis.slm_sync.sync import sync_scan
from yb_analysis.slm_sync.ondemand import get_protocol_source

__all__ = ['SlmSyncClient', 'sync_scan', 'get_protocol_source']
