"""Probability calculations: survival (P11), loss (P10), loading rate.

Port of MATLAB's get_prob11.m, get_prob10.m, get_loadingRate_siteResolved.m.
"""

import warnings

import numpy as np


def _inter_site_sem(mean_sr):
    """Standard error of the mean across sites.

    Works correctly for any number of reps including 1, where the
    binomial SEM formula gives 0 (binary per-site outcomes).

    Parameters
    ----------
    mean_sr : ndarray (nSites, nParams) — per-site means, NaN where no data

    Returns
    -------
    sem : ndarray (nParams,)
    """
    n_loaded = np.sum(~np.isnan(mean_sr), axis=0)  # sites with data per param
    with np.errstate(invalid='ignore'):
        sem = np.nanstd(mean_sr, axis=0, ddof=1) / np.sqrt(np.maximum(n_loaded, 1))
    sem = np.where(n_loaded > 1, sem, np.nan)
    return sem


# ---- Survival: P(1→1 | loaded) ----

def prob11_site_resolved(logic1, logic2):
    """Site-resolved survival probability P(img1=1 AND img2=1 | img1=1).

    Parameters
    ----------
    logic1, logic2 : ndarray (nSites, nParams, nReps) bool

    Returns
    -------
    mean_sr : ndarray (nSites, nParams) — NaN where no data
    sem_sr  : ndarray (nSites, nParams)
    """
    joint = np.sum(logic1 & logic2, axis=2)   # (nSites, nParams)
    loaded = np.sum(logic1, axis=2)            # conditioned on img1=1

    mean_sr = np.full(joint.shape, np.nan)
    sem_sr = np.full(joint.shape, np.nan)

    mask = loaded > 0
    p11 = joint[mask] / loaded[mask]
    mean_sr[mask] = p11
    sem_sr[mask] = np.sqrt(p11 * (1 - p11) / loaded[mask])

    return mean_sr, sem_sr


def prob11(logic1, logic2):
    """Site-averaged survival probability.

    Grand mean of the per-site survival ratios across sites (each site weighted
    equally), with the per-site binomial SEMs propagated. This matches the
    MATLAB reference ``get_prob11.m``:
        mean[p] = mean_s prob11_sr[s, p]          (omitnan)
        sem[p]  = sqrt(Σ_s sem_sr[s, p]²) / nSites
    and keeps this consistent with the live dashboard and with ``prob10`` /
    ``loading_rate`` here (which also average over sites). NOTE: this is *not*
    the pooled Σjoint/Σloaded — that load-weights sites unequally and diverges
    from MATLAB when sites load at different rates.

    Returns
    -------
    mean : ndarray (nParams,)
    sem  : ndarray (nParams,)
    """
    mean_sr, sem_sr = prob11_site_resolved(logic1, logic2)
    n_sites = mean_sr.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN columns → NaN
        mean = np.nanmean(mean_sr, axis=0)
    sem = np.sqrt(np.nansum(sem_sr**2, axis=0)) / max(n_sites, 1)
    return mean, sem


# ---- Loss: P(1→0 | loaded) ----

def prob10_site_resolved(logic1, logic2):
    """Site-resolved loss probability P(img1=1 AND img2=0 | img1=1).

    Parameters
    ----------
    logic1, logic2 : ndarray (nSites, nParams, nReps) bool

    Returns
    -------
    mean_sr, sem_sr : ndarray (nSites, nParams)
    """
    loss = np.sum(logic1 & ~logic2, axis=2)
    loaded = np.sum(logic1, axis=2)

    mean_sr = np.full(loss.shape, np.nan)
    sem_sr = np.full(loss.shape, np.nan)

    mask = loaded > 0
    p10 = loss[mask] / loaded[mask]
    mean_sr[mask] = p10
    sem_sr[mask] = np.sqrt(p10 * (1 - p10) / loaded[mask])

    return mean_sr, sem_sr


def prob10(logic1, logic2):
    """Site-averaged loss probability.

    Returns
    -------
    mean, sem : ndarray (nParams,)
    """
    mean_sr, _ = prob10_site_resolved(logic1, logic2)
    mean = np.nanmean(mean_sr, axis=0)
    sem = _inter_site_sem(mean_sr)
    return mean, sem


# ---- Loading rate: P(img1=1) ----

def loading_rate_site_resolved(logic1, reps_per_param=None):
    """Site-resolved loading rate.

    Parameters
    ----------
    logic1 : ndarray (nSites, nParams, nReps) bool
    reps_per_param : ndarray (nParams,) int, optional
        Actual reps per param (from ``unpack_scan_logicals``).  When omitted,
        ``logic1.shape[2]`` is assumed uniform across params — correct only
        when every param has exactly the same number of reps.  Pass it in for
        scans with non-uniform reps (mid-scan aborts, scrambled scans where
        the run ended before all combinations got equal coverage).

    Returns
    -------
    mean_sr, sem_sr : ndarray (nSites, nParams)
    """
    if reps_per_param is None:
        n_per_param = np.full(logic1.shape[1], logic1.shape[2], dtype=int)
    else:
        n_per_param = np.asarray(reps_per_param).astype(int)
    # Padded slots are False, so summing gives the true loaded-event count per
    # site-param.  Divide each column by that param's actual rep count.
    loaded = logic1.sum(axis=2).astype(float)            # (nSites, nParams)
    denom = np.maximum(n_per_param, 1).astype(float)     # (nParams,)
    mean_sr = np.where(n_per_param > 0, loaded / denom, np.nan)
    sem_sr = np.where(n_per_param > 0,
                      np.sqrt(mean_sr * (1 - mean_sr) / denom),
                      np.nan)
    return mean_sr, sem_sr


# ---- Paired-site conditional probabilities: P(img2=AB | img1=11) ----

def _pair_binomial(both_loaded, outcome, axis):
    """Binomial mean and SEM for a paired-site outcome.

    Parameters
    ----------
    both_loaded : ndarray, int counts of events where both sites loaded
    outcome     : ndarray, int counts of the specific img2 outcome
    axis        : None or tuple, axes to sum over (for pooling)

    Returns
    -------
    mean, sem : ndarray (NaN where both_loaded == 0)
    """
    if axis is not None:
        both_loaded = np.sum(both_loaded, axis=axis)
        outcome = np.sum(outcome, axis=axis)
    both_loaded = both_loaded.astype(float)
    mask = both_loaded > 0
    mean = np.where(mask, outcome / np.maximum(both_loaded, 1), np.nan)
    sem = np.where(mask,
                   np.sqrt(mean * (1 - mean) / np.maximum(both_loaded, 1)),
                   np.nan)
    return mean, sem


def pair_prob_site_resolved(logic1, logic2):
    """Paired-site conditional probabilities, site-resolved.

    Pairs neighboring sites (0&1, 2&3, ...).  Conditions on both sites
    loaded in img1 (img1=11 for the pair).

    Parameters
    ----------
    logic1, logic2 : ndarray (nSites, nParams, nReps) bool
        nSites must be even.

    Returns
    -------
    p1111_sr, p1111_sem_sr : ndarray (nPairs, nParams)
    p1100_sr, p1100_sem_sr : ndarray (nPairs, nParams)
    p1110_sr, p1110_sem_sr : ndarray (nPairs, nParams)
    p1101_sr, p1101_sem_sr : ndarray (nPairs, nParams)
    """
    assert logic1.shape[0] % 2 == 0, "nSites must be even for pairing"

    l1A, l1B = logic1[0::2], logic1[1::2]  # (nPairs, nParams, nReps)
    l2A, l2B = logic2[0::2], logic2[1::2]

    both_loaded = np.sum(l1A & l1B, axis=2)  # (nPairs, nParams)

    n11 = np.sum(l1A & l1B & l2A & l2B, axis=2)
    n00 = np.sum(l1A & l1B & ~l2A & ~l2B, axis=2)
    n10 = np.sum(l1A & l1B & l2A & ~l2B, axis=2)
    n01 = np.sum(l1A & l1B & ~l2A & l2B, axis=2)

    assert (n11 + n00 + n10 + n01 == both_loaded).all(), "Outcome counts must sum to total loaded"
    
    p1111_sr, p1111_sem_sr = _pair_binomial(both_loaded, n11, axis=None)
    p1100_sr, p1100_sem_sr = _pair_binomial(both_loaded, n00, axis=None)
    p1110_sr, p1110_sem_sr = _pair_binomial(both_loaded, n10, axis=None)
    p1101_sr, p1101_sem_sr = _pair_binomial(both_loaded, n01, axis=None)

    return (p1111_sr, p1111_sem_sr, p1100_sr, p1100_sem_sr,
            p1110_sr, p1110_sem_sr, p1101_sr, p1101_sem_sr)


def pair_prob(logic1, logic2):
    """Paired-site conditional probabilities, pooled across all pairs.

    Pools both-loaded events across all pairs and reps for each parameter
    point, then computes binomial mean and SEM.

    Returns
    -------
    p1111, p1111_sem : ndarray (nParams,)
    p1100, p1100_sem : ndarray (nParams,)
    p1110, p1110_sem : ndarray (nParams,)
    p1101, p1101_sem : ndarray (nParams,)
    """
    assert logic1.shape[0] % 2 == 0, "nSites must be even for pairing"

    l1A, l1B = logic1[0::2], logic1[1::2]
    l2A, l2B = logic2[0::2], logic2[1::2]

    cond = l1A & l1B  # (nPairs, nParams, nReps)

    both_loaded = np.sum(cond, axis=(0, 2))  # (nParams,)
    n11 = np.sum(cond & l2A & l2B, axis=(0, 2))
    n00 = np.sum(cond & ~l2A & ~l2B, axis=(0, 2))
    n10 = np.sum(cond & l2A & ~l2B, axis=(0, 2))
    n01 = np.sum(cond & ~l2A & l2B, axis=(0, 2))

    p1111, p1111_sem = _pair_binomial(both_loaded, n11, axis=None)
    p1100, p1100_sem = _pair_binomial(both_loaded, n00, axis=None)
    p1110, p1110_sem = _pair_binomial(both_loaded, n10, axis=None)
    p1101, p1101_sem = _pair_binomial(both_loaded, n01, axis=None)

    return (p1111, p1111_sem, p1100, p1100_sem,
            p1110, p1110_sem, p1101, p1101_sem)


def loading_rate(logic1, reps_per_param=None):
    """Site-averaged loading rate.

    Parameters
    ----------
    logic1 : ndarray (nSites, nParams, nReps) bool
    reps_per_param : ndarray (nParams,) int, optional
        Actual reps per param.  See ``loading_rate_site_resolved`` — required
        for correct results when reps are non-uniform.

    Returns
    -------
    mean, sem : ndarray (nParams,)
    """
    mean_sr, _ = loading_rate_site_resolved(logic1, reps_per_param)
    mean = np.nanmean(mean_sr, axis=0)
    sem = _inter_site_sem(mean_sr)
    return mean, sem


def per_shot_rate_stats(logic1, logic2=None, reps_per_param=None):
    """Per-parameter statistics computed ACROSS SHOTS (not across sites).

    For each shot the survival rate is ``(#sites loaded AND survived) /
    (#sites loaded)`` over that shot's sites, and the loading rate is
    ``(#sites loaded) / nSites``. We then take the mean / sample-std
    (ddof=1) / SEM (= std/√n) across the *eligible* shots of each scan
    parameter. This is the "per-shot" error convention (treat each shot
    as one sample of the array-averaged rate), distinct from the per-site
    binomial SEM that ``prob11`` / ``loading_rate`` return.

    Parameters
    ----------
    logic1 : ndarray (nSites, nParams, nReps) bool
    logic2 : ndarray (nSites, nParams, nReps) bool, optional
        When omitted, only loading stats are returned.
    reps_per_param : ndarray (nParams,) int, optional
        Real rep count per param (padded reps are all-False and would
        otherwise be counted as genuine zero-loading shots). Pass the
        value from ``unpack_scan_logicals`` for scrambled / aborted scans.

    Returns
    -------
    dict with keys (each a list of length nParams, NaN where undefined):
      ``loading_mean`` ``loading_std_pershot`` ``loading_sem_pershot``
      ``loading_n_shots`` and, when ``logic2`` given, the same with the
      ``survival_`` prefix.
    """
    if logic1 is None or logic1.ndim != 3 or logic1.size == 0:
        return {}
    n_sites, n_params, max_reps = logic1.shape
    if reps_per_param is None:
        reps = np.full(n_params, max_reps, dtype=int)
    else:
        reps = np.clip(np.asarray(reps_per_param, dtype=int), 0, max_reps)

    nan = float('nan')
    out = {k: [nan] * n_params for k in (
        'loading_mean', 'loading_std_pershot', 'loading_sem_pershot')}
    out['loading_n_shots'] = [0] * n_params
    have2 = logic2 is not None and logic2.size
    if have2:
        for k in ('survival_mean', 'survival_std_pershot',
                  'survival_sem_pershot'):
            out[k] = [nan] * n_params
        out['survival_n_shots'] = [0] * n_params

    for p in range(n_params):
        R = int(reps[p])
        if R <= 0:
            continue
        l1 = logic1[:, p, :R]
        loaded_per_shot = l1.sum(axis=0).astype(float)   # (R,)
        load_rates = loaded_per_shot / max(n_sites, 1)
        out['loading_mean'][p] = float(load_rates.mean())
        out['loading_n_shots'][p] = R
        if R > 1:
            s = float(load_rates.std(ddof=1))
            out['loading_std_pershot'][p] = s
            out['loading_sem_pershot'][p] = s / np.sqrt(R)
        else:
            out['loading_std_pershot'][p] = 0.0
        if have2:
            joint = (l1 & logic2[:, p, :R]).sum(axis=0).astype(float)
            elig = loaded_per_shot > 0
            n_el = int(elig.sum())
            if n_el > 0:
                sr = joint[elig] / loaded_per_shot[elig]
                out['survival_mean'][p] = float(sr.mean())
                out['survival_n_shots'][p] = n_el
                if n_el > 1:
                    s = float(sr.std(ddof=1))
                    out['survival_std_pershot'][p] = s
                    out['survival_sem_pershot'][p] = s / np.sqrt(n_el)
                else:
                    out['survival_std_pershot'][p] = 0.0
    return out


def rearrangement_success_rate(logic2, reps_per_param=None):
    """Per-parameter average of ``logic2`` over array-2 sites and reps.

    In two-array mode (isGrid2=1), image-2 captures atoms after a
    rearrangement step targeting a defect-free array. The fraction of
    occupied array-2 sites is the rearrangement success rate. Mathematically
    identical to ``loading_rate(logic2)`` — provided as a more discoverable
    name for the two-array use case.

    Returns
    -------
    mean, sem : ndarray (nParams,)
    """
    return loading_rate(logic2, reps_per_param)
