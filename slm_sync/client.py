"""HTTP client wrapper for the SLM PC's Phase 1/2 retrieval endpoints.

A thin wrapper around `requests` that handles:
- Tailscale base URL + optional basic-auth password.
- Retries with backoff for the SLM-PC gate-busy 503s (the SLM server's
  middleware blocks non-essential reads during active rearrange calls;
  it returns 503 'server busy: rearrange'. We back off 0.3s and retry).
- Connect/read timeouts from `yb_analysis.config`.

Endpoints wrapped:
- GET /slm/runs/{scan_id}/diag           — full ledger (Phase 1, deployed)
- GET /slm/runs/{scan_id}/diag?since_seq_id=N  — incremental
- GET /slm/runs/{scan_id}/code           — code-snapshot manifest (Phase 2)
- GET /slm/code/by_hash/{sha256}         — source bytes for a blob (Phase 2)

Used by `sync.py` for the at-scan-end batch sync and by `ondemand.py`
for analyst on-demand source retrieval.
"""

import logging
import time

import requests
from requests.auth import HTTPBasicAuth

from yb_analysis import config

logger = logging.getLogger(__name__)


# The SLM server's gate-busy 503 returns this exact detail prefix.
# When we see it, we back off and retry rather than failing the sync.
_GATE_BUSY_PREFIX = 'server busy'


class SlmSyncClient:
    """Thin requests wrapper with retry-on-503 and Tailscale defaults.

    Construct once per sync; reuse across calls so the underlying
    requests.Session pools TCP connections.
    """

    def __init__(self, slm_url=None, password=None, timeout_s=None,
                 verify_tls=None, max_retries=8, retry_backoff_s=0.3):
        self._url = (slm_url or config.SLM_URL).rstrip('/')
        self._timeout = timeout_s or config.SLM_HTTP_TIMEOUT_S
        self._verify_tls = (verify_tls if verify_tls is not None
                            else config.SLM_VERIFY_TLS)

        # Auth: prefer the explicit password arg, then SLM_PASSWORD_PATH.
        # Inside the tailnet, the password is not strictly needed but the
        # SLM server's middleware requires it for POST routes (and some
        # GETs depending on config). Best to always send when we have it.
        if password is None:
            pw_path = config.SLM_PASSWORD_PATH
            if pw_path:
                try:
                    with open(pw_path, 'r') as f:
                        password = f.read().strip()
                except OSError:
                    pass
        # Hard-coded fallback to match the lab's MATLAB SLMClient default
        # so a fresh deployment doesn't need extra env wiring.
        password = password or '174171'
        self._auth = HTTPBasicAuth('admin', password) if password else None

        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._session = requests.Session()

    def _get(self, path, **kw):
        """GET with retry on 503-gate-busy. Raises for other HTTP errors
        and lets connection errors bubble up so the caller can mark
        slm_sync_status='partial'."""
        url = self._url + path
        for attempt in range(self._max_retries):
            r = self._session.get(
                url, timeout=self._timeout, verify=self._verify_tls,
                auth=self._auth, **kw)
            if r.status_code == 503:
                # SLM server's gate middleware: try again in a moment.
                try:
                    detail = r.json().get('detail', '')
                except Exception:
                    detail = r.text
                if _GATE_BUSY_PREFIX in detail:
                    time.sleep(self._retry_backoff_s)
                    continue
            return r
        # Final attempt; return whatever we got even if it's still 503.
        return r

    # ---- Endpoint wrappers ----

    def get_diag(self, scan_id, since_seq_id=None):
        """GET /slm/runs/{scan_id}/diag (optionally ?since_seq_id=N).

        Returns the parsed JSON: ``{scan_id, count, overflow, entries}``.
        Returns ``None`` if the SLM PC is unreachable. Raises
        ``requests.HTTPError`` on 4xx (e.g. malformed scan_id).
        """
        path = f'/slm/runs/{scan_id}/diag'
        if since_seq_id is not None:
            path += f'?since_seq_id={int(since_seq_id)}'
        try:
            r = self._get(path)
        except (requests.ConnectionError, requests.Timeout):
            return None
        if r.status_code == 404:
            # The SLM server returns 404 for an unknown scan_id; but
            # actually our endpoint returns 200 with count=0 for unknown
            # scan_ids. A 404 here means the entire endpoint is missing
            # (pre-Phase-2 SLM build), which we surface as None too.
            return None
        r.raise_for_status()
        return r.json()

    def get_code_manifest(self, scan_id):
        """GET /slm/runs/{scan_id}/code.

        Returns ``{scan_id, safe_run_id, manifest, manifest_path}`` or
        ``None`` if the SLM PC is unreachable or no code snapshot exists
        for this scan_id (404).
        """
        try:
            r = self._get(f'/slm/runs/{scan_id}/code')
        except (requests.ConnectionError, requests.Timeout):
            return None
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def get_code_by_hash(self, sha256):
        """GET /slm/code/by_hash/{sha256}. Returns the source bytes as
        a UTF-8 string, or ``None`` if unreachable / not found.

        ``sha256`` MUST be a 64-char lowercase hex string; otherwise the
        SLM server returns 400 (and we raise HTTPError).
        """
        try:
            r = self._get(f'/slm/code/by_hash/{sha256}')
        except (requests.ConnectionError, requests.Timeout):
            return None
        if r.status_code == 404:
            return None
        r.raise_for_status()
        # Server returns text/plain; charset=utf-8. Use .text for
        # cross-platform line-ending normalisation.
        return r.text

    def get_grid_sidecar(self, scan_id):
        """GET /slm/runs/{scan_id}/grid_sidecar.

        Phase 4 addition: fetches the per-run grid sidecar produced by
        the SLM server's ``rearrange_grid_sidecar.write_grid_sidecar``
        path (upstream commit ``2b4e179``). The sidecar carries the
        EXACT derived+reordered (init/target knm coords in bit order +
        gridLocations reference + grid_rotation + affine diag) so
        lab-side analysis can reconstruct the scoring lattice by replay
        instead of re-deriving from the WGS phase. Returns ``None`` if
        the endpoint is missing (SLM PC not yet on a build that exposes
        it) or no grid was written for this scan_id.
        """
        try:
            r = self._get(f'/slm/runs/{scan_id}/grid_sidecar')
        except (requests.ConnectionError, requests.Timeout):
            return None
        if r.status_code == 404:
            return None
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            # Non-JSON body — surface as None rather than crashing the
            # whole sync.
            logger.warning("get_grid_sidecar(%s): non-JSON body", scan_id)
            return None
