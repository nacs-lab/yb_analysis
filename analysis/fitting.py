"""Curve fitting: Lorentzian, exponential, Gaussian.

Port of MATLAB's fit_lorentzian_siteResolved.m, plotScan_and_fit_lorentzian_peak.m,
plotScan_and_fit_exponential.m.
"""

import numpy as np
from scipy.optimize import curve_fit


# ---- Models ----

def lorentzian_dip(x, y0, A, x0, w):
    """Lorentzian dip: y = y0 - A * (w/2)^2 / ((x-x0)^2 + (w/2)^2)"""
    return y0 - A * (w/2)**2 / ((x - x0)**2 + (w/2)**2)


def lorentzian_peak(x, y0, A, x0, w):
    """Lorentzian peak: y = y0 + A * (w/2)^2 / ((x-x0)^2 + (w/2)^2)"""
    return y0 + A * (w/2)**2 / ((x - x0)**2 + (w/2)**2)


def exponential_decay(x, a, tau, c):
    """Exponential decay: y = a * exp(-x/tau) + c"""
    return a * np.exp(-x / tau) + c


def gaussian_peak(x, y0, A, x0, sigma):
    """Gaussian peak: y = y0 + A * exp(-(x-x0)^2 / (2*sigma^2))"""
    return y0 + A * np.exp(-(x - x0)**2 / (2 * sigma**2))


def double_lorentzian_dip(x, y0, A1, x01, w1, A2, x02, w2):
    """Double Lorentzian dip."""
    return (y0
            - A1 * (w1/2)**2 / ((x - x01)**2 + (w1/2)**2)
            - A2 * (w2/2)**2 / ((x - x02)**2 + (w2/2)**2))


# ---- Fitting functions ----

def fit_lorentzian(x, y, yerr=None, mode='dip'):
    """Fit a Lorentzian peak or dip.

    Parameters
    ----------
    x, y : ndarray
        Data points.
    yerr : ndarray, optional
        Error bars for weighted fit.
    mode : 'dip' or 'peak'

    Returns
    -------
    dict with: params (y0, A, x0, w), pcov, model_func, x_fit, y_fit, r_squared
    """
    func = lorentzian_dip if mode == 'dip' else lorentzian_peak

    mask = np.isfinite(y) & np.isfinite(x)
    x, y = x[mask], y[mask]
    if yerr is not None:
        yerr = yerr[mask]

    if len(x) < 4:
        return None

    # Initial guess
    y0_g = np.median(y)
    if mode == 'dip':
        idx_min = np.argmin(y)
        A_g = y0_g - y[idx_min]
        x0_g = x[idx_min]
    else:
        idx_max = np.argmax(y)
        A_g = y[idx_max] - y0_g
        x0_g = x[idx_max]
    w_g = (x.max() - x.min()) / 5

    try:
        sigma = 1.0 / yerr if yerr is not None and np.all(yerr > 0) else None
        popt, pcov = curve_fit(func, x, y, p0=[y0_g, max(A_g, 0.01), x0_g, w_g],
                                sigma=sigma, absolute_sigma=True, maxfev=5000)

        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = func(x_fit, *popt)
        ss_res = np.sum((y - func(x, *popt))**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'params': popt,  # [y0, A, x0, w]
            'pcov': pcov,
            'model': func,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'r_squared': r2,
            'center': popt[2],
            'width': popt[3],
        }
    except Exception:
        return None


def fit_exponential(x, y, yerr=None):
    """Fit exponential decay: y = a * exp(-x/tau) + c.

    Returns
    -------
    dict with: params (a, tau, c), x_fit, y_fit, r_squared
    """
    mask = np.isfinite(y) & np.isfinite(x)
    x, y = x[mask], y[mask]
    if yerr is not None:
        yerr = yerr[mask]

    if len(x) < 3:
        return None

    a_g = y[0] - y[-1]
    tau_g = (x.max() - x.min()) / 3
    c_g = y[-1]

    try:
        sigma = 1.0 / yerr if yerr is not None and np.all(yerr > 0) else None
        popt, pcov = curve_fit(exponential_decay, x, y, p0=[a_g, tau_g, c_g],
                                sigma=sigma, absolute_sigma=True, maxfev=5000)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = exponential_decay(x_fit, *popt)
        ss_res = np.sum((y - exponential_decay(x, *popt))**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'params': popt,  # [a, tau, c]
            'pcov': pcov,
            'model': exponential_decay,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'r_squared': r2,
            'tau': popt[1],
        }
    except Exception:
        return None


def fit_lorentzian_site_resolved(scan_params, prob_sr, sem_sr=None, mode='dip'):
    """Fit Lorentzian to each site independently.

    Parameters
    ----------
    scan_params : ndarray (nParams,)
    prob_sr : ndarray (nSites, nParams)
    sem_sr : ndarray (nSites, nParams), optional
    mode : 'dip' or 'peak'

    Returns
    -------
    centers : ndarray (nSites,) — NaN where fit failed
    widths  : ndarray (nSites,)
    params  : ndarray (nSites, 4) — [y0, A, x0, w]
    fits    : list of fit dicts (or None per site)
    """
    n_sites = prob_sr.shape[0]
    centers = np.full(n_sites, np.nan)
    widths = np.full(n_sites, np.nan)
    params = np.full((n_sites, 4), np.nan)
    fits = []

    for s in range(n_sites):
        err = sem_sr[s] if sem_sr is not None else None
        result = fit_lorentzian(scan_params, prob_sr[s], yerr=err, mode=mode)
        fits.append(result)
        if result is not None:
            centers[s] = result['center']
            widths[s] = result['width']
            params[s] = result['params']

    return centers, widths, params, fits
