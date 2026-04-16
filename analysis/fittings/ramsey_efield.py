"""Ramsey E-field fitting: cosine with quadratic phase."""

import numpy as np
from scipy.optimize import curve_fit


def ramsey_efield_model(V, A, phi0, alpha, B, V0):
    """y = A * cos(phi0 + alpha * (V - V0)^2) + B"""
    return A * np.cos(phi0 + alpha * (V - V0)**2) + B


def fit_ramsey_efield(x, y, yerr=None):
    """Fit Ramsey E-field fringes: y = A*cos(phi0 + alpha*(V - V0)^2) + B.

    Evaluates residuals on a coarse (V0, alpha, phi0) grid analytically
    to find the best starting point, then runs curve_fit once to refine.

    Parameters
    ----------
    x : ndarray
        Voltage values.
    y : ndarray
        Survival rate.
    yerr : ndarray, optional
        Error bars for weighted fit.

    Returns
    -------
    dict with: params (A, phi0, alpha, B, V0), perr, pcov,
               model, x_fit, y_fit, r_squared, V0
    None if fit fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)[mask]

    if len(x) < 5:
        return None

    B0 = np.mean(y)
    V_range = x.max() - x.min()
    ss_tot = np.sum((y - B0)**2)
    sigma = yerr if yerr is not None and np.all(yerr > 0) else None

    # Estimate alpha from zero crossings
    y_centered = y - B0
    n_crossings = len(np.where(np.diff(np.sign(y_centered)))[0])
    n_osc = max(n_crossings / 2, 1)
    alpha_est = n_osc * 2 * np.pi / (V_range / 2)**2

    # Coarse grid — evaluate model analytically (no curve_fit).
    # For each (V0, alpha), the optimal A, B, phi0 can be found via
    # linear projection: y = A*cos(theta + phi0) + B is linear in
    # (A*cos(phi0), A*sin(phi0), B), so we solve with least squares.
    V0_grid = np.linspace(x.min() + 0.1 * V_range, x.max() - 0.1 * V_range, 25)
    alpha_grid = np.linspace(max(alpha_est * 0.3, 10), alpha_est * 2.5, 30)

    best_ss = np.inf
    best_p0 = None

    for V0_try in V0_grid:
        dV2 = (x - V0_try)**2
        for alpha_try in alpha_grid:
            theta = alpha_try * dV2
            # y ≈ c1*cos(theta) + c2*sin(theta) + c3
            # Solve [cos(theta) | sin(theta) | 1] @ [c1, c2, c3]^T = y
            M = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(x)])
            coeffs, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
            c1, c2, c3 = coeffs
            ss = np.sum((y - M @ coeffs)**2)
            if ss < best_ss:
                best_ss = ss
                A_init = np.hypot(c1, c2)
                phi0_init = np.arctan2(-c2, c1)
                best_p0 = [A_init, phi0_init, alpha_try, c3, V0_try]

    if best_p0 is None:
        return None

    # Refine with curve_fit from the best grid point
    bounds = ([0, -np.pi, 0, 0, x.min() - 0.5 * V_range],
              [np.inf, np.pi, np.inf, 1.0, x.max() + 0.5 * V_range])
    try:
        popt, pcov = curve_fit(ramsey_efield_model, x, y, p0=best_p0,
                               bounds=bounds, sigma=sigma,
                               absolute_sigma=True, maxfev=20000)
    except Exception:
        return None

    perr = np.sqrt(np.diag(pcov))
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = ramsey_efield_model(x_fit, *popt)
    ss_res = np.sum((y - ramsey_efield_model(x, *popt))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    A, phi0, alpha, B, V0 = popt
    label = (f'V\u2080={V0:.5f}\u00b1{perr[4]:.5f} V, '
             f'\u03b1={alpha:.2f}, R\u00b2={r2:.3f}')
    return {
        'fit_label': label,
        'params': popt,       # [A, phi0, alpha, B, V0]
        'perr': perr,
        'pcov': pcov,
        'model': ramsey_efield_model,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'r_squared': r2,
        'center': V0,  # used by plot_scan_interactive for vertical line
        'V0': V0,
        'V0_err': perr[4],
    }
