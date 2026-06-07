"""Param<->channel provenance for the Sequence tab (gated behind real provenance).

The accurate param<->channel relationship is **build-time provenance**: which parameter's
``SeqVal`` actually flows into which channel output. Numeric value-matching is NOT that --
value equality is a coincidence, not a dependency: ubiquitous levels (0/1, idle holds) matched
nearly every channel, ramp samples produced spurious hits, and unit-mismatched params (MHz vs
Hz) never matched at all. So the heuristic was removed (SEQPLOTTER_INTEGRATION_PLAN.md §8) and
the feature is **gated behind provenance produced by the engine build** (the reconstruction /
B3 path), cached as ``<scan>/sequence/xref.json``. This module only READS that artifact.

Artifact schema (written by the engine-side builder, keyed by the ``.seq`` filename)::

    {
      "scan_id": "...",
      "by_file": {
        "point_00001__seqid_1.seq": {
          "param_to_channels": {"Init.EOM616.Freq": ["FreqEOM616"], ...},
          "channel_to_params": {"FreqEOM616": ["Init.EOM616.Freq"], ...},
          "pulses": {"60": {"channel": "VBiasCoilY (NiDAQ/Dev1/1)",
                            "params": ["GreenMOT.BFieldRampTime", ...]}, ...},
          "param_to_pids": {"GreenMOT.BFieldRampTime": [60, ...], ...}
        }
      }
    }

``pulses`` is keyed by the pulse id (== the ``.seq``'s per-point ``pid``), so a clicked plot
point maps to exactly its segment's params; ``param_to_pids`` is the inverse, for highlighting
a parameter's region(s). Both are absent in pre-region artifacts (the aggregate maps remain).

Until that artifact exists the feature is dormant: :func:`load_xref` returns
``available=False`` and the dashboard shows no param<->channel affordance.
"""

import json
import os

XREF_NAME = "xref.json"


def _empty():
    return {"available": False, "version": 0, "param_to_channels": {},
            "channel_to_params": {}, "pulses": {}, "param_to_pids": {}, "time_regions": {}}


def load_xref(seq_dir, fname=None):
    """Return param<->channel provenance for ``fname`` in ``seq_dir``.

    ``{available, param_to_channels, channel_to_params}``. ``available`` is False when there
    is no ``xref.json`` or no entry for ``fname`` (the feature stays dormant). When ``fname``
    is None and the artifact has exactly one entry, that entry is used.
    """
    if not seq_dir:
        return _empty()
    path = os.path.join(seq_dir, XREF_NAME)
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return _empty()
    by_file = doc.get("by_file") or {}
    entry = by_file.get(os.path.basename(str(fname))) if fname else None
    if entry is None and fname is None and len(by_file) == 1:
        entry = next(iter(by_file.values()))
    if not isinstance(entry, dict):
        return _empty()
    return {
        "available": True,
        "version": int(doc.get("v") or 0),       # 0 == pre-versioning (viewer upgrades it)
        "param_to_channels": entry.get("param_to_channels") or {},
        "channel_to_params": entry.get("channel_to_params") or {},
        # Per-pulse (region) provenance: ``pulses`` is {pid(str): {channel, params}} and
        # ``param_to_pids`` is {param: [pid]} -- the viewer maps a clicked plot point
        # (customdata=pid) to just that segment's params, and a clicked param to its
        # region(s). Absent in older artifacts -> empty (the aggregate maps still work).
        "pulses": entry.get("pulses") or {},
        "param_to_pids": entry.get("param_to_pids") or {},
        # Wait/timing regions: {param: [[t0_ms, t1_ms], ...]} -> shaded time-axis bands.
        "time_regions": entry.get("time_regions") or {},
    }


def write_xref(seq_dir, by_file, *, scan_id=None):
    """Write the ``xref.json`` provenance artifact (used by the engine-side builder).

    ``by_file`` maps each ``.seq`` filename -> ``{param_to_channels, channel_to_params}``.
    Atomic (tmp + replace). Best-effort: returns the path, or None on failure.
    """
    try:
        os.makedirs(seq_dir, exist_ok=True)
        doc = {"scan_id": scan_id, "by_file": by_file}
        tmp = os.path.join(seq_dir, XREF_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        dst = os.path.join(seq_dir, XREF_NAME)
        os.replace(tmp, dst)
        return dst
    except OSError:
        return None
