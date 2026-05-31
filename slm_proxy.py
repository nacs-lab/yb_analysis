"""Read-only proxy that polls the SLM PC and writes a shared pickle for the dashboard.

The lab-PC dashboard needs live SLM-side panels (phase preview, camera frame,
lock status, rearrangement diag, health). Rather than have the Dash subprocess
open its own HTTP connections to the SLM PC, we centralize polling here: one
daemon thread per endpoint runs on the main `run_monitor` process, pickles the
latest snapshot to `yb_dash_slm.pkl`, and the Dash subprocess reads it on its
3 s tick the same way it already reads `yb_dash_data.pkl` and `yb_dash_queue.pkl`.

Failure semantics:

- Each endpoint has its own thread + last-error timestamp. One slow/dead
  endpoint can't block the others.
- Every `requests.get` uses `SLM_HTTP_TIMEOUT_S` from config. If the SLM PC
  is unreachable, the threads spin in their normal cadence but each
  request fails fast; `slm_offline` is surfaced in the pickle so the
  dashboard greys panels out.
- If the proxy is started with `--no-slm`, no threads spawn and the pickle
  is never written; the dashboard treats the file's absence as "SLM proxy
  disabled".
"""

import logging
import os
import pickle
import tempfile
import threading
import time

logger = logging.getLogger(__name__)


# Shared pickle file path. Sits next to the existing dashboard data and
# queue pickles in tempdir so the dashboard subprocess can read all three
# the same way.
SLM_DATA_FILE = os.path.join(tempfile.gettempdir(), 'yb_dash_slm.pkl')


class SlmProxy:
    """Daemon-thread proxy that polls SLM-PC endpoints and writes a pickle.

    One thread per endpoint. Each thread sleeps for the configured interval
    between polls. All threads share a single output dict + lock; the pickle
    write happens after every successful update.
    """

    # Endpoint catalog — name -> (HTTP path, response handler, kind).
    # kind='json' parses .json(); kind='png' stores raw bytes (base64'd at
    # pickle time so the bytes survive the dashboard's data-URI rendering).
    _ENDPOINTS = (
        ('health',         '/health',                'json'),
        ('devices',        '/devices',               'json'),
        ('lock_status',    '/lock/status',           'json'),
        ('camera_png',     '/camera/capture_png',    'png'),
        ('phase_png',      '/slm/phase_png',         'png'),
        ('rearrange_diag', '/slm/rearrange_diag?n=20', 'json'),
    )

    def __init__(self, slm_url, intervals_ms, timeout_s=(2.0, 5.0),
                 verify_tls=False, password=None):
        """
        slm_url       — base URL like 'http://100.114.207.118:8551'.
        intervals_ms  — dict {endpoint_name: poll_interval_ms}.
        timeout_s     — (connect, read) tuple for requests.get.
        verify_tls    — passed through to requests.get (False inside tailnet).
        password      — if set, used as HTTP basic-auth password with 'admin' user.
        """
        self._url = slm_url.rstrip('/')
        self._intervals_ms = intervals_ms
        self._timeout_s = timeout_s
        self._verify_tls = verify_tls
        self._auth = ('admin', password) if password else None

        # Shared state across worker threads. Always updated under _lock.
        self._lock = threading.Lock()
        # File-write serialization. Six poll threads racing on
        # `os.replace(tmp, dest)` on Windows can hit WinError 32 (file in use).
        # Serializing the entire write critical-section is cheap (pickle is
        # ~few KB) and bullet-proof. _lock guards the in-memory state;
        # _write_lock guards the on-disk file dance.
        self._write_lock = threading.Lock()
        self._state = {
            'slm_url': self._url,
            'slm_offline': True,    # flipped to False on first successful poll
            'last_poll_ts': {},     # endpoint name -> epoch
            'last_error_ts': {},    # endpoint name -> epoch
            'last_error_msg': {},   # endpoint name -> str
        }
        for name, _path, _kind in self._ENDPOINTS:
            self._state[name] = None  # latest payload (dict for json, bytes for png)

        # Lifecycle.
        self._stop = threading.Event()
        self._threads = []

    # ---- Lifecycle ----

    def start(self):
        """Spawn one daemon thread per endpoint. Idempotent."""
        if self._threads:
            return
        for name, path, kind in self._ENDPOINTS:
            interval_ms = self._intervals_ms.get(name, 5000)
            t = threading.Thread(
                target=self._poll_loop,
                args=(name, path, kind, interval_ms / 1000.0),
                name=f'slm-proxy-{name}',
                daemon=True)
            t.start()
            self._threads.append(t)
        logger.info('SLM proxy started: %s (%d endpoints)',
                    self._url, len(self._threads))

    def stop(self, timeout_s=2.0):
        """Signal threads to exit and wait for them. Idempotent."""
        if not self._threads:
            return
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout_s)
        alive = [t.name for t in self._threads if t.is_alive()]
        if alive:
            logger.warning('SLM proxy threads still alive after %.1fs: %s',
                           timeout_s, alive)
        else:
            logger.info('SLM proxy stopped')
        self._threads = []

    # ---- Polling ----

    def _poll_loop(self, name, path, kind, interval_s):
        """Run forever (until self._stop): poll one endpoint at fixed cadence."""
        # requests is imported here (not module-level) so the proxy module
        # remains importable when requests isn't installed and --no-slm is used.
        import requests
        session = requests.Session()
        url = self._url + path
        while not self._stop.is_set():
            try:
                r = session.get(url, timeout=self._timeout_s,
                                verify=self._verify_tls, auth=self._auth)
                r.raise_for_status()
                payload = r.json() if kind == 'json' else r.content
                with self._lock:
                    self._state[name] = payload
                    self._state['last_poll_ts'][name] = time.time()
                    # Any successful poll flips offline -> online. We do not
                    # require all endpoints to succeed; one is enough.
                    self._state['slm_offline'] = False
                    # Clear the per-endpoint error if it cleared.
                    self._state['last_error_msg'].pop(name, None)
                self._write_pickle()
            except Exception as e:
                with self._lock:
                    self._state['last_error_ts'][name] = time.time()
                    self._state['last_error_msg'][name] = str(e)[:200]
                    # If every endpoint has errored since their last success,
                    # mark offline. Cheap: just check whether any poll_ts is
                    # newer than its error_ts.
                    ep_names = [n for n, _, _ in self._ENDPOINTS]
                    if not any(self._state['last_poll_ts'].get(n, 0)
                               > self._state['last_error_ts'].get(n, 0)
                               for n in ep_names):
                        self._state['slm_offline'] = True
                self._write_pickle()
            # Interruptible sleep — wake immediately when stop is set.
            self._stop.wait(interval_s)

    # ---- Pickle write ----

    def _write_pickle(self):
        """Snapshot state under the lock, then write atomically.

        Atomic via tempfile + os.replace; partial writes are never observed.
        The Dash subprocess reads this on its existing 3 s callback tick.
        Disk I/O is serialized under `_write_lock` to avoid Windows
        file-in-use races between the six concurrent poll threads.
        """
        with self._lock:
            snapshot = dict(self._state)
            # The inner dicts are small but copy them so the writer is fully
            # detached from any concurrent mutation.
            snapshot['last_poll_ts'] = dict(self._state['last_poll_ts'])
            snapshot['last_error_ts'] = dict(self._state['last_error_ts'])
            snapshot['last_error_msg'] = dict(self._state['last_error_msg'])
        tmp = SLM_DATA_FILE + '.tmp'
        try:
            with self._write_lock:
                with open(tmp, 'wb') as f:
                    pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, SLM_DATA_FILE)
        except OSError as e:
            logger.warning('SLM proxy pickle write failed: %s', e)


def _read_slm_data():
    """Read the latest SLM-proxy snapshot. Called from the Dash subprocess.

    Returns the unpickled dict, or None if the file is missing / unreadable.
    """
    try:
        with open(SLM_DATA_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError):
        return None
