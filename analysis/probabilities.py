"""Probability calculations: survival (P11), loss (P10), loading rate.

Port of MATLAB's get_prob11.m, get_prob10.m, get_loadingRate_siteResolved.m.
"""

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

    The SEM for each parameter point is a pooled binomial SEM computed from
    the total loaded-atom events across all sites and reps for that point:
        sem[p] = sqrt(p̄ * (1 - p̄) / total_loaded[p])
    This gives each point its own error bar that scales with its actual
    repetition count — parameters with fewer reps receive larger error bars.

    Returns
    -------
    mean : ndarray (nParams,)
    sem  : ndarray (nParams,)
    """
    # Pool loaded and joint events across all sites and reps
    joint_total = np.sum(logic1 & logic2, axis=(0, 2)).astype(float)  # (nParams,)
    loaded_total = np.sum(logic1, axis=(0, 2)).astype(float)           # (nParams,)

    mean = np.where(loaded_total > 0, joint_total / loaded_total, np.nan)
    sem = np.where(loaded_total > 0,
                   np.sqrt(mean * (1 - mean) / np.maximum(loaded_total, 1)),
                   np.nan)
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

def loading_rate_site_resolved(logic1):
    """Site-resolved loading rate.

    Parameters
    ----------
    logic1 : ndarray (nSites, nParams, nReps) bool

    Returns
    -------
    mean_sr, sem_sr : ndarray (nSites, nParams)
    """
    n_reps = logic1.shape[2]
    mean_sr = logic1.mean(axis=2).astype(float)
    sem_sr = np.sqrt(mean_sr * (1 - mean_sr) / max(n_reps, 1))
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


def loading_rate(logic1):
    """Site-averaged loading rate.

    Returns
    -------
    mean, sem : ndarray (nParams,)
    """
    mean_sr, _ = loading_rate_site_resolved(logic1)
    mean = np.nanmean(mean_sr, axis=0)
    sem = _inter_site_sem(mean_sr)
    return mean, sem
