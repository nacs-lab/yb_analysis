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
