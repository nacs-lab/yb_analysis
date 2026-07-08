"""Global per-site analysis mask -- "analyze only these tweezer sites".

One place that turns whatever the user supplies (a bool array, an index list, a
path to a ``.npy``, or a registered NAME) into a boolean mask over the array's
sites, and applies it to the unpacked ``logic[nSites, nParams, nReps]`` cubes.

The mask is applied by NaN-ing out the excluded site rows (not dropping them), so
the full 1068-wide site axis is preserved -- per-site (x,y) alignment, the pattern
grid, and detection column order all stay intact, while every array-averaged
readout (survival / loading / FP) skips the excluded sites via ``nanmean``.

Registered names live in ``MASK_REGISTRY`` (name -> path); regenerate the file
behind a name with ``pyctrl/tools/build_stable_site_mask.py``.

Scoping: a site mask is intrinsically ARRAY-specific (its indices ARE that
pattern's detection-column order), so the default lives PER PATTERN in the
pattern registry (``record.json['site_mask']``, read via
``pattern_registry.load_pattern_site_mask``). An analysis call with no explicit
``site_mask`` auto-uses the scan's own pattern mask (or None). Pass an explicit
spec to override; pass ``False`` to force the full array even if the pattern has
a mask configured.
"""
from __future__ import annotations

import os
import numpy as np

# name -> .npy path (bool[nSites] or int index list). Keep paths absolute so the
# resolver works regardless of the caller's cwd.
_ROOT = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
MASK_REGISTRY = {
    # 744 sites whose relative trap depth changed <5% between the 06-22 post-feedback
    # map and the 07-05 measurement (regenerate: _daily/_stable_mask.py).
    "stable": os.path.join(_ROOT, "_daily", "stable_sites_lt5pct.npy"),
}


def resolve_site_mask(spec, n_sites):
    """Return a bool[n_sites] mask, or None for 'use all sites'.

    ``spec`` may be:
      - None            -> None (no masking)
      - a registered name (str in MASK_REGISTRY)
      - a path to a .npy (str ending in .npy) holding bool[n_sites] or int indices
      - a bool array (len n_sites) or an int index array/list
    Raises ValueError on a length mismatch or unknown name.
    """
    if spec is None:
        return None

    arr = None
    if isinstance(spec, str):
        if spec in MASK_REGISTRY:
            path = MASK_REGISTRY[spec]
        elif spec.endswith(".npy"):
            path = spec
        else:
            raise ValueError(
                "unknown site-mask %r (not a registered name %s nor a .npy path)"
                % (spec, sorted(MASK_REGISTRY)))
        if not os.path.exists(path):
            raise ValueError("site-mask file not found: %s" % path)
        arr = np.load(path, allow_pickle=False)
    else:
        arr = np.asarray(spec)

    arr = np.asarray(arr)
    if arr.dtype == bool:
        if arr.size != n_sites:
            raise ValueError("bool site-mask length %d != n_sites %d" % (arr.size, n_sites))
        mask = arr.astype(bool)
    else:
        # integer index list -> boolean
        idx = arr.astype(np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= n_sites):
            raise ValueError("site-mask index out of range for n_sites %d" % n_sites)
        mask = np.zeros(n_sites, dtype=bool)
        mask[idx] = True
    return mask


def mask_site_resolved(arr_sr, mask):
    """NaN-out excluded site rows of a site-resolved array ``[nSites, nParams]``.

    This is the masking primitive: the per-site probability/loading arrays are
    computed on the FULL array (bool cubes untouched -> no change to the tested
    MATLAB-parity math), then this NaNs the excluded rows. Every site-averaged
    readout is a ``nanmean``/``nansum`` over sites, so the excluded rows drop out
    automatically; per-site (x,y) maps keep full width (excluded -> NaN -> grey).
    No-op if mask is None or the shapes don't line up. Returns a copy.
    """
    if mask is None or arr_sr is None:
        return arr_sr
    a = np.asarray(arr_sr, dtype=float)
    if a.ndim != 2 or a.shape[0] != mask.size:
        return arr_sr
    a = a.copy()
    a[~mask, :] = np.nan
    return a


def effective_spec(spec, pattern=None):
    """Resolve which site-mask SPEC applies, without needing n_sites yet.

    Precedence:
      - ``spec is False`` -> force the FULL array (returns None), even if the
        pattern configures a mask. Use to opt OUT per call.
      - ``spec`` given (not None/False) -> use it verbatim (explicit override).
      - ``spec is None`` -> the pattern's configured mask (registry
        ``record.json['site_mask']``), or None if the pattern has none.
    Returns the spec string/array or None. The bool array is resolved later by
    ``resolve_site_mask`` against the scan's n_sites."""
    if spec is False:
        return None
    if spec is not None:
        return spec
    if pattern:
        try:
            from yb_analysis.analysis import pattern_registry as _pr
            return _pr.load_pattern_site_mask(pattern)
        except Exception:
            return None
    return None
