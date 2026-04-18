"""Dipolar exchange fitting: joint damped cosine for paired-site SS/PP.

Model:
    P_SS(t) = B_SS + A * cos(omega*t + phi) * exp(-gamma*t)
    P_PP(t) = B_PP - A * cos(omega*t + phi) * exp(-gamma*t)

SS and PP share the same frequency (omega) and damping (gamma).
"""

import numpy as np
from scipy.optimize import curve_fit


def dipolar_exchange_model(t, A, omega, phi, gamma, B_SS, B_PP):
    """Dipolar exchange oscillation for a single time array.

    Returns
    -------
    P_SS, P_PP : ndarray
    """
    osc = A * np.cos(omega * t + phi) * np.exp(-gamma * t)
    return B_SS + osc, B_PP - osc


def _joint_model(t_both, A, omega, phi, gamma, B_SS, B_PP):
    """Concatenated model for curve_fit: [P_SS(t); P_PP(t)]."""
    n = len(t_both) // 2
    t = t_both[:n]
    osc = A * np.cos(omega * t + phi) * np.exp(-gamma * t)
    return np.concatenate([B_SS + osc, B_PP - osc])


def _estimate_omega(x, y):
    """Rough frequency estimate from zero-crossings."""
    y_centered = y - np.mean(y)
    crossings = np.where(np.diff(np.sign(y_centered)))[0]
    T_range = x[-1] - x[0]
    if len(crossings) >= 2:
        period_est = 2 * T_range / len(crossings)
        return 2 * np.pi / period_est
    return 2 * np.pi / T_range


def _confidence_band(t, popt, pcov):
    """1-sigma confidence band via covariance propagation.

    Returns
    -------
    sigma_ss, sigma_pp : ndarray, same length as t
    """
    A, omega, phi, gamma, _, _ = popt
    cos_term = np.cos(omega * t + phi)
    sin_term = np.sin(omega * t + phi)
    exp_term = np.exp(-gamma * t)
    osc = cos_term * exp_term

    J_ss = np.column_stack([
        osc,                              # dP/dA
        -A * t * sin_term * exp_term,     # dP/domega
        -A * sin_term * exp_term,         # dP/dphi
        -A * t * osc,                     # dP/dgamma
        np.ones_like(t),                  # dP/dB_SS
        np.zeros_like(t),                 # dP/dB_PP
    ])
    J_pp = np.column_stack([
        -osc,
        A * t * sin_term * exp_term,
        A * sin_term * exp_term,
        A * t * osc,
        np.zeros_like(t),
        np.ones_like(t),
    ])

    sigma_ss = np.sqrt(np.sum(J_ss @ pcov * J_ss, axis=1))
    sigma_pp = np.sqrt(np.sum(J_pp @ pcov * J_pp, axis=1))
    return sigma_ss, sigma_pp


def fit_dipolar_exchange(x, y_ss, y_pp, yerr_ss=None, yerr_pp=None):
    """Joint fit of dipolar exchange oscillation for SS and PP channels.

    Parameters
    ----------
    x : ndarray
        Time values.
    y_ss, y_pp : ndarray
        P(11|11) and P(00|11) probabilities.
    yerr_ss, yerr_pp : ndarray, optional
        Error bars for weighted fit.

    Returns
    -------
    dict with: params [A, omega, phi, gamma, B_SS, B_PP], perr, pcov,
               x_fit, y_fit_ss, y_fit_pp, sigma_ss, sigma_pp,
               r_squared, fit_label, freq, omega, gamma
    None if fit fails.
    """
    x = np.asarray(x, dtype=float)
    y_ss = np.asarray(y_ss, dtype=float)
    y_pp = np.asarray(y_pp, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y_ss) & np.isfinite(y_pp)
    if yerr_ss is not None and yerr_pp is not None:
        yerr_ss = np.asarray(yerr_ss, dtype=float)
        yerr_pp = np.asarray(yerr_pp, dtype=float)
        valid &= (yerr_ss > 0) & (yerr_pp > 0)

    xv = x[valid]
    if len(xv) < 6:
        return None

    y_ss_v, y_pp_v = y_ss[valid], y_pp[valid]
    t_cat = np.concatenate([xv, xv])
    y_cat = np.concatenate([y_ss_v, y_pp_v])

    if yerr_ss is not None and yerr_pp is not None:
        s_cat = np.concatenate([yerr_ss[valid], yerr_pp[valid]])
    else:
        s_cat = None

    # Initial guesses
    A0 = (np.nanmax(y_ss_v) - np.nanmin(y_ss_v)) / 2
    omega0 = _estimate_omega(xv, y_ss_v)
    T_range = xv[-1] - xv[0]
    gamma0 = 1.0 / T_range

    p0 = [A0, omega0, 0.0, gamma0, np.mean(y_ss_v), np.mean(y_pp_v)]
    bounds_lo = [0, 0, -np.pi, 0, 0, 0]
    bounds_hi = [1, np.inf, np.pi, np.inf, 1, 1]

    try:
        popt, pcov = curve_fit(_joint_model, t_cat, y_cat, p0=p0,
                               sigma=s_cat, absolute_sigma=False,
                               bounds=(bounds_lo, bounds_hi), maxfev=20000)
    except Exception:
        return None

    A, omega, phi, gamma, B_SS, B_PP = popt
    perr = np.sqrt(np.diag(pcov))

    # Evaluate fit and confidence band on fine grid
    x_fit = np.linspace(xv[0], xv[-1], 500)
    y_fit_ss, y_fit_pp = dipolar_exchange_model(x_fit, *popt)
    sigma_ss, sigma_pp = _confidence_band(x_fit, popt, pcov)

    # R-squared (joint)
    ss_res = np.sum((y_cat - _joint_model(t_cat, *popt))**2)
    ss_tot = np.sum((y_cat - np.mean(y_cat))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    freq = omega / (2 * np.pi)
    label = (f'f={freq:.4g} Hz, \u03b3={gamma:.4g} s\u207b\u00b9, '
             f'R\u00b2={r2:.3f}')

    return {
        'fit_label': label,
        'params': popt,       # [A, omega, phi, gamma, B_SS, B_PP]
        'perr': perr,
        'pcov': pcov,
        'x_fit': x_fit,
        'y_fit_ss': y_fit_ss,
        'y_fit_pp': y_fit_pp,
        'sigma_ss': sigma_ss,
        'sigma_pp': sigma_pp,
        'r_squared': r2,
        'freq': freq,
        'omega': omega,
        'gamma': gamma,
    }
