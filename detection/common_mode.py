"""Per-shot COMMON-MODE brightness normalization for atom detection.

The 399 imaging fluorescence breathes shot-to-shot as a GLOBAL (common-mode)
GAIN: the whole array brightens/dims together (img1 vs img2 of the SAME shot
correlate at 0.987; per-shot blob-brightness CV ~22-24%; lag-1 autocorrelation
0.70, decaying over ~2-3 shots). It is MULTIPLICATIVE on the atom signal and
NOT additive on the background (empty-site std 0.04 ADU, only 0.42-correlated),
and it is the DOMINANT per-site imaging-fidelity limiter: removing it offline
lifted per-site fidelity 0.993 -> 0.9996, worst-5% 0.96 -> 0.998, median d'
4.4 -> 7.3, and measured 2198 survival 0.979 -> 0.987. (Problem-memory
``open-imaging-common-mode-shot-wobble``; offline prototype section (A) of
``pyctrl/tools/offline_imaging_diag.py``.)

This module is the LIVE, CAUSAL version of that prototype. The offline version
divided every shot's atom signal by that shot's global loaded-site brightness
and re-normalized to the RUN mean -- not available while a scan is running --
so here the run mean is replaced by a causal EWMA of the preceding shots.

Estimator (per camera frame, per shot)::

    sig   = I - mu_empty                 # atom signal above the site's baseline
    ref   = mu_atom - mu_empty           # that site's CALIBRATED atom signal
    good  = finite fits with ref >= CM_NORM_MIN_SEP
    sel   = good AND (I > threshold)     # DETECTED-bright sites, RAW comparison
    g_raw = median( sig[sel] / ref[sel] )
    g     = clip(g_raw / ref_of_previous_g_raw, CM_NORM_GAIN_MIN, ..._MAX)
            # ref = debiased EWMA: running MEAN until the exponential
            #       memory (1/CM_NORM_EMA shots) is filled, then EWMA
    I_norm[good] = mu_empty + (I - mu_empty) / g

Two deliberate choices:

* **Robust statistic over DETECTED-bright sites, each divided by its OWN
  calibrated atom signal** (rather than a plain mean of the atom signal). The
  per-site division makes the estimator invariant to WHICH sites happen to be
  filled -- essential here, because the filled subset changes every shot in a
  rearrangement run and differs per frame (~0.7 fill on the 3013 loading frame,
  ~0.9 on the 2198 verify frame). A plain mean would read a bright/dim subset as
  a brightness wobble. The median over ~2000 sites has ~1% statistical error
  against a ~24% wobble. Empty sites are deliberately excluded: the wobble is a
  gain on the ATOM signal, so empties carry no information about it.
* **A causal EWMA reference, not the calibration itself.** Dividing by the
  stored calibration would (a) inherit any staleness of the per-pattern
  ``threshold.mat`` (the 2198/2078 stores are days older than the 3013 one and
  sit at a different absolute brightness) and (b) fight the existing cheap
  threshold tracker, which already follows SLOW drift. Referencing the EWMA of
  the preceding shots removes any static offset exactly (selection bias,
  stale calibration, global level change) and leaves ONLY the fast shot-to-shot
  deviation -- the documented pathology -- to this corrector. The EWMA memory
  (~1/CM_NORM_EMA shots) is far longer than the ~2-3 shot wobble correlation
  time, so the wobble is not absorbed into the reference.

Equivalent view: comparing ``mu_e + sig/g`` against a fixed cut ``thr`` is the
same as comparing the RAW intensity against ``mu_e + g*(thr - mu_e)`` -- i.e. the
per-site cut is held at its calibrated placement ratio relative to THIS shot's
actual atom brightness. Sites without a usable fit are left untouched (never
NaN-ed into a forced "empty").

**The threshold accumulators, the per-site histograms and the stored HDF5
``intensities*`` datasets keep RAW intensities** -- only the logicals comparison
uses the normalized value. So the double-Gaussian refit, its degeneracy guard
and the cheap inter-fit tracker all see exactly what they see today (no new
corruption channel -- cf. ``bug-threshold-dim-scan-contamination`` /
``bug-pyctrl-threshold-degenerate-fit-drift``), the wobble stays measurable in
the saved data, and an A/B of the SAME run is a pure offline recomputation.
"""

import collections
import threading

import numpy as np

from yb_analysis.config import (
    CM_NORM_ENABLED, CM_NORM_MIN_SITES, CM_NORM_MIN_SEP, CM_NORM_EMA,
    CM_NORM_WARMUP, CM_NORM_GAIN_MIN, CM_NORM_GAIN_MAX, CM_NORM_HISTORY,
)

# Runtime kill switch. The process-wide default comes from config/$YB_CM_NORM;
# set_enabled() flips it live (no restart) for an A/B without touching the
# accumulators -- every DataManager consults it per frame.
_enabled = bool(CM_NORM_ENABLED)
_enabled_lock = threading.Lock()


def is_enabled():
    """True when the live detection path applies the correction."""
    with _enabled_lock:
        return _enabled


def set_enabled(flag):
    """Turn the live correction on/off at runtime. Returns the new state.
    The per-frame gain is TRACKED either way (so the wobble stays visible on
    the dashboard with the correction off)."""
    global _enabled
    with _enabled_lock:
        _enabled = bool(flag)
        return _enabled


def reference_from_fits(gauss_fits, num_sites, min_sep=CM_NORM_MIN_SEP):
    """Per-site brightness reference from a double-Gaussian fit list.

    ``gauss_fits`` is the package-wide ``[{'params': [mu_e, s_e, A_e, mu_a,
    s_a, A_a]}, ...]`` structure (empty peak first). Returns
    ``(mu_empty (M,), atom_ref (M,), good (M,) bool)`` or ``None`` when no site
    has a usable fit. ``atom_ref = mu_atom - mu_empty`` is the site's calibrated
    atom signal; ``good`` marks the sites that may be normalized AND may enter
    the gain estimate.
    """
    n = int(num_sites)
    if n <= 0 or not gauss_fits:
        return None
    mu_e = np.full(n, np.nan, dtype=np.float64)
    ref = np.full(n, np.nan, dtype=np.float64)
    for s in range(min(n, len(gauss_fits))):
        f = gauss_fits[s]
        p = f.get('params') if isinstance(f, dict) else None
        if p is None:
            continue
        pr = np.ravel(np.asarray(p, dtype=np.float64))
        if pr.size < 4:
            continue
        mu_e[s] = pr[0]
        ref[s] = pr[3] - pr[0]
    good = np.isfinite(mu_e) & np.isfinite(ref) & (ref >= float(min_sep))
    if not good.any():
        return None
    return mu_e, ref, good


def shot_gain(intensities, thresholds, mu_empty, atom_ref, good,
              min_sites=CM_NORM_MIN_SITES):
    """RAW per-shot brightness gain relative to the calibration reference.

    Median of ``(I - mu_empty) / (mu_atom - mu_empty)`` over the sites this
    frame detected as bright (RAW intensity above the RAW threshold -- the
    selection never sees a normalized value, so there is no feedback loop).
    NaN when too few sites qualify (blank frame, pushed-out frame, tiny array)
    -- the caller then leaves the shot alone.
    """
    I = np.asarray(intensities, dtype=np.float64).ravel()
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    if I.size == 0 or thr.size != I.size or good.size != I.size:
        return np.nan
    sel = good & np.isfinite(I) & (I > thr)
    if int(sel.sum()) < int(min_sites):
        return np.nan
    g = float(np.median((I[sel] - mu_empty[sel]) / atom_ref[sel]))
    if not np.isfinite(g) or g <= 0.0:
        return np.nan
    return g


def normalize(intensities, mu_empty, good, gain):
    """Divide the ATOM SIGNAL of every usable site by ``gain``, keeping each
    site's own empty baseline: ``mu_e + (I - mu_e)/gain``. Sites without a
    usable fit (``good`` False) keep their raw value -- never NaN, which the
    ``> threshold`` comparison would silently read as 'empty'."""
    I = np.asarray(intensities, dtype=np.float64)
    out = I.astype(np.float64, copy=True)
    g = float(gain)
    if not np.isfinite(g) or g <= 0.0 or g == 1.0:
        return out
    out[good] = mu_empty[good] + (I[good] - mu_empty[good]) / g
    return out


class CommonModeTracker:
    """Per-frame EWMA reference + per-shot gain bookkeeping.

    One instance per DataManager (i.e. per scan); ``key`` is the camera frame
    index, so the 3-frame rearrangement scans (3013 loading / 2198 verify /
    2078 target) each get their OWN reference -- the three arrays sit at
    different absolute brightness and must never share one.

    Deliberately per-scan, not module-global: a new scan can be a different
    array / imaging config, and the warmup is only ``CM_NORM_WARMUP`` shots.
    """

    def __init__(self, ema=CM_NORM_EMA, warmup=CM_NORM_WARMUP,
                 gain_min=CM_NORM_GAIN_MIN, gain_max=CM_NORM_GAIN_MAX,
                 min_sites=CM_NORM_MIN_SITES, history=CM_NORM_HISTORY):
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.min_sites = int(min_sites)
        self.history = int(history)
        self._frames = {}

    def _frame(self, key):
        st = self._frames.get(key)
        if st is None:
            st = {'ewma': None, 'n': 0, 'n_applied': 0, 'n_skipped': 0,
                  'last_raw': None, 'last_gain': None,
                  'raw_hist': collections.deque(maxlen=self.history)}
            self._frames[key] = st
        return st

    def observe(self, key, intensities, thresholds, mu_empty, atom_ref, good,
                apply_correction=True):
        """Fold one frame into ``key``'s reference and return the gain to apply.

        Returns ``1.0`` (an exact no-op) whenever the shot cannot be corrected:
        too few detected-bright sites, no usable fits, still in warmup, or
        ``apply_correction=False`` (the caller bypasses it but still wants the
        wobble tracked). The EWMA is updated from the RAW estimate whether or
        not the correction is applied, so toggling the correction never
        discontinuously moves the reference.
        """
        st = self._frame(key)
        g_raw = shot_gain(intensities, thresholds, mu_empty, atom_ref, good,
                          min_sites=self.min_sites)
        if not np.isfinite(g_raw):
            st['n_skipped'] += 1
            st['last_gain'] = None
            return 1.0
        prev = st['ewma']                      # reference from PRECEDING shots
        st['n'] += 1
        # DEBIASED EWMA: weight max(ema, 1/n) makes the reference the plain
        # RUNNING MEAN until the exponential memory (1/ema shots) is filled,
        # then exponential. A cold-started EWMA is anchored on shot 1 and needs
        # ~1/ema shots to forget it, so a single unlucky first shot (measured:
        # a 0.30 opener on a CV-0.27 series) would drag the reference far below
        # the truth and the correction would slam into its rail for ~50 shots.
        w = max(self.ema, 1.0 / st['n'])
        st['ewma'] = g_raw if prev is None else (1.0 - w) * prev + w * g_raw
        st['last_raw'] = float(g_raw)
        st['raw_hist'].append(float(g_raw))
        if not apply_correction or prev is None or st['n'] <= self.warmup:
            st['last_gain'] = None
            return 1.0
        gain = float(np.clip(g_raw / prev, self.gain_min, self.gain_max))
        st['last_gain'] = gain
        st['n_applied'] += 1
        return gain

    def frame_stats(self, key):
        """``{n, n_applied, n_skipped, last_raw, last_gain, ewma, cv}`` for one
        frame; ``cv`` is the coefficient of variation of the recent RAW gains --
        i.e. the size of the common-mode wobble this frame is seeing."""
        st = self._frames.get(key)
        if st is None:
            return None
        h = np.asarray(st['raw_hist'], dtype=np.float64)
        cv = (float(np.std(h) / np.mean(h))
              if h.size >= 3 and np.mean(h) > 0 else None)
        return {'n': int(st['n']), 'n_applied': int(st['n_applied']),
                'n_skipped': int(st['n_skipped']),
                'last_raw': st['last_raw'], 'last_gain': st['last_gain'],
                'ewma': (float(st['ewma']) if st['ewma'] is not None else None),
                'cv': cv}

    def snapshot(self):
        """Dashboard payload: the live enable state + per-frame gain stats."""
        return {'enabled': is_enabled(),
                'ema': self.ema, 'warmup': self.warmup,
                'frames': {int(k): self.frame_stats(k)
                           for k in sorted(self._frames)}}
