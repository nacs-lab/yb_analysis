"""Build the Sequence-tab PARAMETERS tree from a scan's ``.json`` sidecar.

Per the 2026-06-06 design review (SEQPLOTTER_INTEGRATION_PLAN.md §12.2), the
parameters card is reconstructed from the scan's ``data_<stamp>.json`` sidecar --
NOT from the ``.seq`` blob and NOT via a run-time dump-side walk that would touch
the engine. This is **engine-free** and works for **any scan with a ``.json``**,
with or without a ``.seq`` (so a never-dumped scan still shows its params).

The sidecar (written by pyctrl ``scan_prep.write_scan_config`` /
``scan_summary.scangroup_scan_config``) carries:

  * ``expConfig``                 -- the baseline ``SeqConfig.consts`` snapshot;
  * ``ScanGroup.base.params``     -- the fixed / ``g()``-override struct;
  * ``ScanGroup.base.vars``       -- the swept axes (``{params:[dim-struct...], size:[...]}``).

We fold these into the viewer's nested status-code tree, where each leaf is
``{"value", "type", "config_value"?}`` and ``type`` is the SeqPlotter status code:

  * ``1`` = config (an ``expConfig`` baseline leaf, unchanged);
  * ``3`` = overwritten-config (a config leaf changed by ``base.params``);
  * ``2`` = overwritten-ordinary (a ``base.params`` leaf with no config baseline);
  * ``0`` = default (a swept leaf injected so the scanned axis is visible).

Swept axes are returned separately as ``scanned_paths`` (the dashboard JS marks
those leaves "scanned"); a swept leaf absent from config/overrides is injected so
the highlight always has a target. Owning this writer **resolves the historical
``type``-tag-vs-status-code ambiguity** (we emit the status code directly).

**Fidelity note (§12.2):** this is the *configured* params (full ``expConfig`` +
overrides + scanned). It does NOT capture build-time resolution (Step-pulled
defaults, computed ``SeqVal``s, runtime globals). A strict resolved-``seq.C`` view
is a build-only replay from the code snapshot (the reconstruction driver) -- this
``.json`` tree is the always-available default.
"""

import json
import os
import re

# A scan's config sidecar is ``data_<8 digit date>_<6 digit time>.json`` and shares
# the scan directory's basename. This pattern excludes siblings like
# ``analysis_cache.json`` / ``slm_grid.json`` / ``focus_metrics.json``.
_SIDECAR_RE = re.compile(r"^data_\d{8}_\d{6}\.json$")


# --------------------------------------------------------------------------- #
# Leaf helpers
# --------------------------------------------------------------------------- #
def _is_leaf(x):
    """A viewer leaf: a dict carrying both ``value`` and ``type`` (matches the JS)."""
    return isinstance(x, dict) and "value" in x and "type" in x


def _baseline_tree(cfg_node):
    """``expConfig`` subtree -> viewer tree; every scalar leaf is a type-1 config leaf."""
    out = {}
    for k, v in cfg_node.items():
        if isinstance(v, dict):
            out[k] = _baseline_tree(v)
        else:
            out[k] = {"value": v, "type": 1, "config_value": v}
    return out


def _overlay_overrides(tree, ov_node, stats, have_baseline):
    """Overlay ``base.params`` onto the baseline tree, tagging modified leaves.

    A config leaf (in ``expConfig``) changed by an override -> type 3 (counted
    modified); an override equal to its baseline stays type 1 (not modified). An
    override with NO baseline -> type 2 when a baseline exists (a genuine override
    of a non-config param), else type 1 (older scans with no ``expConfig`` snapshot
    -> show the params plainly instead of all-red).
    """
    for k, v in ov_node.items():
        if isinstance(v, dict):
            child = tree.get(k)
            if not isinstance(child, dict) or _is_leaf(child):
                child = {}
                tree[k] = child
            _overlay_overrides(child, v, stats, have_baseline)
            continue
        existing = tree.get(k)
        if _is_leaf(existing):
            base_val = existing.get("config_value")
            existing["value"] = v
            if not (existing.get("type") == 1 and _values_equal(v, base_val)):
                existing["type"] = 3            # overwritten config
                stats["modified"] += 1
        elif have_baseline:
            tree[k] = {"value": v, "type": 2}   # overwritten ordinary (no config baseline)
            stats["modified"] += 1
        else:
            tree[k] = {"value": v, "type": 1}   # no baseline at all -> show plainly


def _values_equal(a, b):
    try:
        return a == b
    except Exception:  # noqa: BLE001
        return False


def _count_leaves(tree):
    n = 0
    for v in tree.values():
        if _is_leaf(v):
            n += 1
        elif isinstance(v, dict):
            n += _count_leaves(v)
    return n


# --------------------------------------------------------------------------- #
# Scanned axes (from ScanGroup.base.vars)
# --------------------------------------------------------------------------- #
def _scanned_axes(base_vars):
    """``{dotted.path: swept_values}`` for every swept leaf across the dim-structs."""
    out = {}
    if not isinstance(base_vars, dict):
        return out
    for dim_struct in (base_vars.get("params") or []):
        _collect_swept(dim_struct, "", out)
    return out


def _collect_swept(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            _collect_swept(v, (prefix + "." + k) if prefix else k, out)
    elif prefix:                                # a non-dict leaf == the swept values
        out[prefix] = node


def _inject_leaf(tree, keys, value):
    """Ensure a leaf exists at ``keys``; create a type-0 leaf if absent (don't clobber)."""
    node = tree
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict) or _is_leaf(nxt):
            nxt = {}
            node[k] = nxt
        node = nxt
    last = keys[-1]
    if not _is_leaf(node.get(last)):
        node[last] = {"value": value, "type": 0}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_params_tree(config):
    """Build the viewer params tree from a parsed ``.json`` sidecar ``config`` dict.

    Returns ``{params, has_params, scanned_paths, stats}`` where ``params`` is the
    nested status-code tree, ``scanned_paths`` is the list of swept dotted paths,
    and ``stats`` is ``{n_leaves, n_modified, n_scanned}``.
    """
    config = config or {}
    exp = config.get("expConfig")
    have_baseline = isinstance(exp, dict) and bool(exp)
    sg_base = ((config.get("ScanGroup") or {}).get("base")) or {}
    overrides = sg_base.get("params") or {}
    base_vars = sg_base.get("vars") or {}

    stats = {"modified": 0}
    tree = _baseline_tree(exp) if have_baseline else {}
    if isinstance(overrides, dict):
        _overlay_overrides(tree, overrides, stats, have_baseline)

    scanned = _scanned_axes(base_vars)
    for path, vals in scanned.items():
        _inject_leaf(tree, path.split("."), vals)

    return {
        "params": tree,
        "has_params": bool(tree),
        "scanned_paths": sorted(scanned.keys()),
        "stats": {
            "n_leaves": _count_leaves(tree),
            "n_modified": stats["modified"],
            "n_scanned": len(scanned),
        },
    }


def find_config_sidecar(scan_folder):
    """Locate the ``data_<stamp>.json`` config sidecar for ``scan_folder`` (or None).

    Accepts a scan dir or its ``sequence/`` subdir (checks the parent too). Prefers
    the canonical ``<dir-basename>.json``; falls back to any ``data_<stamp>.json``.
    """
    if not scan_folder:
        return None
    scan_folder = os.path.abspath(scan_folder)
    candidates = [scan_folder, os.path.dirname(scan_folder)]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        canonical = os.path.join(d, os.path.basename(d) + ".json")
        if os.path.exists(canonical):
            return canonical
        try:
            for f in sorted(os.listdir(d)):
                if _SIDECAR_RE.match(f):
                    return os.path.join(d, f)
        except OSError:
            continue
    return None


def build_params_tree_from_file(path):
    """Load a ``.json`` sidecar and build its params tree; None if unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None
    return build_params_tree(config)


def build_params_tree_for_folder(scan_folder):
    """Convenience: find the sidecar for ``scan_folder`` and build its tree, or None."""
    path = find_config_sidecar(scan_folder)
    if not path:
        return None
    built = build_params_tree_from_file(path)
    if built is not None:
        built["source_file"] = path
    return built
