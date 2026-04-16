"""Exponential decay fitting."""

import numpy as np
from scipy.optimize import curve_fit


def exponential_decay(x, a, tau, c):
    """Exponential decay: y = a * exp(-x/tau) + c"""
    return a * np.exp(-x / tau) + c


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
