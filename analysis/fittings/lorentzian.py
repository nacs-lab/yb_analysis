"""Lorentzian fitting: single peak/dip, double dip, and site-resolved."""

import numpy as np
from scipy.optimize import curve_fit


# ---- Models ----

def lorentzian_dip(x, y0, A, x0, w):
    """Lorentzian dip: y = y0 - A * (w/2)^2 / ((x-x0)^2 + (w/2)^2)"""
    return y0 - A * (w/2)**2 / ((x - x0)**2 + (w/2)**2)


def lorentzian_peak(x, y0, A, x0, w):
    """Lorentzian peak: y = y0 + A * (w/2)^2 / ((x-x0)^2 + (w/2)^2)"""
    return y0 + A * (w/2)**2 / ((x - x0)**2 + (w/2)**2)


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


def fit_double_lorentzian(x, y, yerr=None, mode='dip'):
    """Fit two overlapping Lorentzian dips (e.g. an mj-split / two-component line).

    Mirrors :func:`fit_lorentzian` (same weighting convention) but for the
    seven-parameter :func:`double_lorentzian_dip` model. Components are returned
    sorted by center frequency.

    Parameters
    ----------
    x, y : ndarray
        Data points.
    yerr : ndarray, optional
        Error bars for the weighted fit.
    mode : 'dip'
        Only 'dip' is supported (the spectroscopy push-out convention).

    Returns
    -------
    dict with: params ([y0, A1, x01, w1, A2, x02, w2]), pcov, model, x_fit,
    y_fit, comp1_fit, comp2_fit (the two single-component curves over x_fit),
    r_squared, centers (sorted ascending), widths (matched to centers),
    splitting (|x02 - x01|). Returns None if too few points, the fit fails, or
    the two components collapse onto each other (degenerate -> use the single fit).
    """
    if mode != 'dip':
        raise ValueError("fit_double_lorentzian supports mode='dip' only")

    mask = np.isfinite(y) & np.isfinite(x)
    x, y = x[mask], y[mask]
    if yerr is not None:
        yerr = yerr[mask]

    if len(x) < 7:  # 7 free parameters
        return None

    span = x.max() - x.min()
    # baseline ~ off-resonant survival; seed two components straddling the minimum
    y0_g = np.percentile(y, 90)
    idx_min = np.argmin(y)
    A_g = max(y0_g - y[idx_min], 0.01)
    x0 = x[idx_min]
    x1_g, x2_g = x0 - span / 10, x0 + span / 10
    w_g = span / 8

    p0 = [y0_g, A_g, x1_g, w_g, A_g, x2_g, w_g]
    lo = [-np.inf, 0, x.min(), 1e-9, 0, x.min(), 1e-9]
    hi = [np.inf, np.inf, x.max(), span, np.inf, x.max(), span]

    try:
        sigma = 1.0 / yerr if yerr is not None and np.all(yerr > 0) else None
        popt, pcov = curve_fit(double_lorentzian_dip, x, y, p0=p0, sigma=sigma,
                               absolute_sigma=True, maxfev=100000, bounds=(lo, hi))
    except Exception:
        return None

    y0, A1, x01, w1, A2, x02, w2 = popt
    # Degenerate: components merged (centers closer than half a combined HWHM) ->
    # the doublet collapsed to a single peak; caller should prefer the 1-peak fit.
    if abs(x02 - x01) < 0.25 * (abs(w1) + abs(w2)):
        return None

    x_fit = np.linspace(x.min(), x.max(), 400)
    y_fit = double_lorentzian_dip(x_fit, *popt)
    comp1 = lorentzian_dip(x_fit, y0, A1, x01, w1)
    comp2 = lorentzian_dip(x_fit, y0, A2, x02, w2)
    ss_res = np.sum((y - double_lorentzian_dip(x, *popt)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # sort the two components by center
    comps = sorted([(x01, abs(w1), A1), (x02, abs(w2), A2)], key=lambda c: c[0])
    centers = np.array([comps[0][0], comps[1][0]])
    widths = np.array([comps[0][1], comps[1][1]])

    return {
        'params': popt,
        'pcov': pcov,
        'model': double_lorentzian_dip,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'comp1_fit': comp1,
        'comp2_fit': comp2,
        'r_squared': r2,
        'centers': centers,
        'widths': widths,
        'amplitudes': np.array([comps[0][2], comps[1][2]]),
        'splitting': float(abs(x02 - x01)),
    }


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
