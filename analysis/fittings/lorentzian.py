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
        # curve_fit's `sigma` is the per-point standard deviation, NOT a weight:
        # passing 1/yerr inverts the weighting and voids pcov.
        sigma = yerr if yerr is not None and np.all(yerr > 0) else None
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

    # index-distance logic below assumes x ascending
    order_x = np.argsort(x)
    x, y = x[order_x], y[order_x]
    if yerr is not None:
        yerr = yerr[order_x]

    span = x.max() - x.min()
    y0_g = np.percentile(y, 90)  # baseline ~ off-resonant survival
    idx_min = int(np.argmin(y))
    A_g = max(y0_g - y[idx_min], 0.01)
    x0 = x[idx_min]

    # Seed candidates. The legacy heuristic straddles the global minimum at
    # +/- span/10, which only works when the splitting is <~ span/5 -- on a
    # window sized to the doublet (e.g. an Autler-Townes scan at +/-1 MHz with a
    # 0.9 MHz splitting) BOTH seeds land on the same dip and the fit converges to
    # a spurious narrow/broad pair. So also seed from the two deepest
    # well-separated minima, then keep whichever candidate fits best.
    seeds = [(x0 - span / 10, x0 + span / 10, span / 8)]

    min_sep = max(3, len(x) // 10)  # samples; keep the seeds off the same dip
    deepest = np.argsort(y)
    idx_second = next((int(i) for i in deepest if abs(int(i) - idx_min) >= min_sep), None)
    if idx_second is not None:
        x2nd = x[idx_second]
        sep = abs(x2nd - x0)
        seeds.append((min(x0, x2nd), max(x0, x2nd), max(sep / 3, span / 40)))

    # A component narrower than the sampling step is not a resolvable feature --
    # it is the fitter buying residual by spiking a single noisy point (which is
    # how the bad seeds above used to "win"). Floor both widths at one step.
    dx = float(np.median(np.diff(x))) if len(x) > 1 else span
    w_min = max(dx, 1e-9)

    lo = [-np.inf, 0, x.min(), w_min, 0, x.min(), w_min]
    hi = [np.inf, np.inf, x.max(), span, np.inf, x.max(), span]
    # curve_fit's `sigma` is the per-point standard deviation, NOT a weight:
    # passing 1/yerr inverts the weighting and voids pcov.
    sigma = yerr if yerr is not None and np.all(yerr > 0) else None

    best = None
    for x1_g, x2_g, w_g in seeds:
        x1_g = float(np.clip(x1_g, x.min(), x.max()))
        x2_g = float(np.clip(x2_g, x.min(), x.max()))
        w_g = float(np.clip(w_g, w_min, span))
        p0 = [y0_g, A_g, x1_g, w_g, A_g, x2_g, w_g]
        try:
            popt, pcov = curve_fit(double_lorentzian_dip, x, y, p0=p0, sigma=sigma,
                                   absolute_sigma=True, maxfev=100000, bounds=(lo, hi))
        except Exception:
            continue
        ss_res_c = np.sum((y - double_lorentzian_dip(x, *popt)) ** 2)
        if best is None or ss_res_c < best[2]:
            best = (popt, pcov, ss_res_c)

    if best is None:
        return None
    popt, pcov = best[0], best[1]

    y0, A1, x01, w1, A2, x02, w2 = popt
    # Degenerate: components merged (centers closer than half a combined HWHM) ->
    # the doublet collapsed to a single peak; caller should prefer the 1-peak fit.
    if abs(x02 - x01) < 0.25 * (abs(w1) + abs(w2)):
        return None

    # Spurious second component: when the line is genuinely SINGLE, the spare
    # component has nothing to fit, so it parks on a noise excursion -- shrinking
    # to the width floor and/or with a depth comparable to the point error. Either
    # signature means "not a resolvable doublet"; return None so the caller falls
    # back to the single fit rather than reporting an invented splitting.
    noise = (float(np.median(yerr)) if sigma is not None
             else float(np.std(y - double_lorentzian_dip(x, *popt))))
    for A_i, w_i in ((A1, w1), (A2, w2)):
        if abs(w_i) <= 1.01 * w_min:      # pinned at the floor -> a delta spike
            return None
        if abs(A_i) < 3.0 * noise:        # depth not significant vs the errors
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
