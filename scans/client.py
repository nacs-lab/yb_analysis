"""High-level Python client for submitting scans to the MATLAB runner.

These thin wrappers package a descriptor, validate it, JSON-serialize,
and hand it off through the existing ``ZmqClient`` to the SequenceRunner.
A new ``ZmqClient`` is created on demand using ``Consts().MatlabURL``-
equivalent defaults (``yb_analysis.config.MATLAB_URL``).

Example::

    from yb_analysis.scans import submit_scan, sweep_linspace

    desc_id = submit_scan(
        seq='CoolingSeq',
        params={'Cooling.Detuning': sweep_linspace(20e6, 30e6, 21)},
        runp={'NumPerGroup': 4000, 'Scramble': True})

The returned id is the descriptor's queue position. How it maps to the
running job depends on the live backend: the **MATLAB** runner dispatches the
descriptor into a distinct-id job (follow the descriptor row's
``built_job_id`` in :func:`list_jobs` output), while the **pyctrl** backend
reuses the descriptor's id for the job -- the returned id IS the job id, and
there is no ``built_job_id`` (the descriptor row is dropped, not archived).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Sequence

from yb_analysis.scans.descriptor import (
    SCHEMA_VERSION,
    DescriptorError,
    validate_descriptor,
)

logger = logging.getLogger(__name__)


# Default ZMQ URL. yb_analysis.config exposes MATLAB_URL when present;
# fall back to the canonical Consts().MatlabURL value otherwise.
def _default_matlab_url() -> str:
    try:
        from yb_analysis import config
        url = getattr(config, 'MATLAB_URL', None)
        if url:
            return url
    except Exception:
        pass
    return 'tcp://127.0.0.1:1408'


_CLIENT_CACHE: dict = {}


def _get_client(url: Optional[str] = None):
    """Lazy-build a single ZmqClient per URL. Cached so a notebook
    session reuses one socket across many submit_scan calls."""
    if url is None:
        url = _default_matlab_url()
    if url not in _CLIENT_CACHE:
        from yb_analysis.acquisition.zmq_client import ZmqClient
        _CLIENT_CACHE[url] = ZmqClient(url)
    return _CLIENT_CACHE[url]


def submit_scan(seq: Any,
                params: Optional[Mapping[str, Any]] = None,
                runp: Optional[Mapping[str, Any]] = None,
                opts: Optional[Sequence] = None,
                label: str = '',
                *,
                url: Optional[str] = None,
                client=None,
                validate: bool = True) -> int:
    """Submit a scan descriptor to the MATLAB SequenceRunner.

    Parameters
    ----------
    seq : str | dict
        MATLAB sequence function name (e.g. ``'CoolingSeq'``) or a
        function-handle wrapper from :func:`func_handle`. ``'auto'`` is
        reserved for future param-based seq selection (raises today).
    params : dict, optional
        Map of dotted ScanGroup paths to scalar / vector / sweep specs.
        Anything omitted falls back to the seq's ``Consts()`` defaults
        via the standard ``g.field(default)`` pattern.
    runp : dict, optional
        Map of dotted ``runp()`` paths (``NumPerGroup``, ``NumImages``,
        ``Scramble``, ``AWGs``, ``isInit``, ``isHC`` ...). Same value
        shapes as ``params``.
    opts : list of [key, value], optional
        Extra varargin pairs forwarded verbatim through ``ybBuildScanPayload``
        into the MATLAB runner.
    label : str, optional
        Human-readable label shown in the queue UI. Defaults to ``seq``.

    Returns
    -------
    int
        The descriptor's queue id. To follow the resulting scan through to
        disk: under the **MATLAB** backend, watch :func:`list_jobs` for the
        descriptor row's ``built_job_id`` (a distinct job id); under the
        **pyctrl** backend, the job reuses this id (it IS the job id -- there
        is no ``built_job_id``, and the descriptor row is dropped).
    """
    desc: dict = {'schema_version': SCHEMA_VERSION, 'seq': seq}
    if params:
        desc['params'] = dict(params)
    if runp:
        desc['runp'] = dict(runp)
    if opts:
        desc['opts'] = [list(p) for p in opts]
    if label:
        desc['label'] = str(label)

    if validate:
        try:
            validate_descriptor(desc)
        except DescriptorError:
            logger.error(
                "submit_scan: descriptor failed validation: %s",
                json.dumps(_redact(desc))[:500])
            raise

    desc_json = json.dumps(desc, ensure_ascii=False)
    if client is None:
        client = _get_client(url)
    return int(client.submit_scan_descriptor(desc_json, label=label or _seq_name(seq)))


def list_jobs(*, url: Optional[str] = None, client=None) -> dict:
    """Return the current queue snapshot (jobs + descriptors).

    Output mirrors ``ZmqClient.queue_list``:
    ``{'queued': [...], 'running': {...} | None, 'history': [...]}``.
    Each entry carries a ``kind`` field (``'job'`` or ``'descriptor'``)
    so callers can filter.
    """
    if client is None:
        client = _get_client(url)
    return client.queue_list()


def cancel(entry_id: int, kind: str = 'auto', *, url: Optional[str] = None,
           client=None) -> bool:
    """Cancel a queued job or descriptor by id.

    ``kind`` can be ``'job'``, ``'descriptor'``, or ``'auto'`` (default;
    looks up the entry in the current queue and dispatches accordingly).
    Returns True on success.
    """
    if client is None:
        client = _get_client(url)
    if kind == 'auto':
        snap = client.queue_list()
        kind = _kind_of(entry_id, snap)
        if kind is None:
            return False
    if kind == 'job':
        rep = client.queue_remove(int(entry_id))
    elif kind == 'descriptor':
        rep = client.descriptor_remove(int(entry_id))
    else:
        raise ValueError(f"cancel: kind must be job|descriptor|auto, got {kind!r}")
    return isinstance(rep, str) and rep.lower().startswith('ok')


def move(entry_id: int, direction: str, *, url: Optional[str] = None,
         client=None) -> bool:
    """Move a queued entry up or down within its own kind."""
    if direction not in ('up', 'down'):
        raise ValueError(f"move: direction must be up|down, got {direction!r}")
    if client is None:
        client = _get_client(url)
    rep = client.queue_move(int(entry_id), direction)
    return isinstance(rep, str) and rep.lower().startswith('ok')


def requeue(entry_id: int, *, with_code: bool = False,
            url: Optional[str] = None, client=None, validate: bool = True) -> int:
    """Re-submit an existing queue/history entry's descriptor.

    Looks ``entry_id`` up in the live queue snapshot (``running`` /
    ``queued`` / ``history``), pulls the original descriptor JSON it was
    submitted with, and re-submits it -- producing a new descriptor whose
    parameters are byte-identical to the original. Returns the new
    descriptor's queue id.

    ``with_code`` (reproducibility, #3): also pin the new descriptor to the
    source run's captured code snapshot (``code_snapshot.scan_id`` = the
    run's data id), so the pyctrl run loop replays the exact experiment
    source (YbSeqs/YbSteps/YbScans) that ran originally instead of the live
    tree. Requires the source entry to carry a data id (``file_id``) -- i.e.
    a pyctrl run that actually executed and was snapshotted.

    Only scans submitted through the descriptor path carry a replayable
    descriptor; a job submitted some other way (e.g. a raw payload from
    MATLAB outside ``submit_scan_descriptor``) has nothing to replay.

    Raises
    ------
    LookupError
        If no entry has that id, the entry has no stored descriptor, or
        (``with_code``) the entry has no data id to locate its snapshot.
    DescriptorError
        If ``validate`` is set and the resulting descriptor doesn't validate.
    """
    if client is None:
        client = _get_client(url)
    entry = _find_entry(int(entry_id), client.queue_list())
    if entry is None:
        raise LookupError(f"no queue/history entry with id {entry_id}")
    desc_json = entry.get('descriptor')
    if not desc_json:
        raise LookupError(
            f"entry {entry_id} has no stored descriptor to re-queue "
            "(only scans submitted via the descriptor path can be replayed)")
    if with_code:
        desc_json = _pin_code_snapshot(desc_json, entry)
    if validate:
        try:
            validate_descriptor(json.loads(desc_json))
        except DescriptorError:
            raise
        except Exception as ex:
            raise DescriptorError(
                f"stored descriptor for id {entry_id} is unreadable: {ex}")
    # Preserve the original display label; fall back to the seq name.
    label = entry.get('label') or ''
    if not label:
        try:
            label = _seq_name(json.loads(desc_json).get('seq'))
        except Exception:
            label = ''
    return int(client.submit_scan_descriptor(desc_json, label=label))


def _pin_code_snapshot(desc_json: str, entry: Mapping) -> str:
    """Add ``code_snapshot.scan_id`` (from the entry's data id) to the descriptor JSON,
    so a re-queue replays the source run's captured code. Raises LookupError if the entry
    has no usable data id. Params are otherwise untouched (only the pin is added)."""
    file_id = entry.get('file_id') or ''
    digits = ''.join(ch for ch in str(file_id) if ch.isdigit())
    if len(digits) != 14:
        raise LookupError(
            f"entry {entry.get('id')} has no 14-digit data id (got {file_id!r}); "
            "cannot locate its code snapshot to re-queue with original code")
    try:
        d = json.loads(desc_json)
    except Exception as ex:
        raise DescriptorError(f"stored descriptor is unreadable: {ex}")
    d['code_snapshot'] = {'scan_id': int(digits)}
    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq_name(seq: Any) -> str:
    if isinstance(seq, str):
        return seq
    if isinstance(seq, Mapping):
        return str(seq.get('@', ''))
    return str(seq)


def _kind_of(entry_id: int, snap: Mapping) -> Optional[str]:
    """Locate an entry in queue_list output and return its kind."""
    eid = int(entry_id)
    for section in ('queued', 'history'):
        rows = snap.get(section) or []
        for r in rows:
            if isinstance(r, Mapping) and r.get('id') == eid:
                return r.get('kind', 'job')
    running = snap.get('running')
    if isinstance(running, Mapping) and running.get('id') == eid:
        return running.get('kind', 'job')
    return None


def _find_entry(entry_id: int, snap: Mapping) -> Optional[Mapping]:
    """Return the queue_list entry with ``id == entry_id`` (searching
    running, then queued, then history), or None if absent."""
    eid = int(entry_id)
    running = snap.get('running')
    if isinstance(running, Mapping) and running.get('id') == eid:
        return running
    for section in ('queued', 'history'):
        for r in snap.get(section) or []:
            if isinstance(r, Mapping) and r.get('id') == eid:
                return r
    return None


def _redact(desc: Mapping) -> dict:
    """Trim long arrays from the descriptor before logging an error,
    so a 2000-point sweep doesn't fill the log line."""
    out = {}
    for k, v in desc.items():
        if isinstance(v, Mapping):
            out[k] = {kk: _redact_val(vv) for kk, vv in v.items()}
        else:
            out[k] = _redact_val(v)
    return out


def _redact_val(v):
    if isinstance(v, list) and len(v) > 8:
        return v[:4] + ['...', len(v), '...'] + v[-2:]
    if isinstance(v, Mapping):
        return {k: _redact_val(x) for k, x in v.items()}
    return v
