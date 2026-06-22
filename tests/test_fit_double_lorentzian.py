"""Unit tests for fit_double_lorentzian (two-component / mj-split dips, e.g. 399)."""

import numpy as np

from yb_analysis.analysis.fittings.lorentzian import (
    double_lorentzian_dip,
    lorentzian_dip,
    fit_double_lorentzian,
)


def _grid(lo, hi, n=61):
    return np.linspace(lo, hi, n)


def test_recovers_two_components():
    """A synthetic two-dip line is recovered: both centers, splitting, sorted order."""
    x = _grid(280e6, 320e6, 81)
    truth = dict(y0=1.0, A1=0.7, x01=293e6, w1=9e6, A2=0.6, x02=308e6, w2=7e6)
    y = double_lorentzian_dip(x, **truth)
    ye = np.full_like(y, 0.01)

    fit = fit_double_lorentzian(x, y, ye, mode="dip")
    assert fit is not None
    c1, c2 = fit["centers"]
    # components come back sorted ascending in center frequency
    assert c1 < c2
    assert abs(c1 - 293e6) < 0.5e6
    assert abs(c2 - 308e6) < 0.5e6
    assert abs(fit["splitting"] - 15e6) < 1e6
    assert fit["r_squared"] > 0.99


def test_single_dip_is_degenerate_returns_none():
    """A genuine single dip should NOT be force-split into two -> None (use 1-peak)."""
    x = _grid(280e6, 320e6, 81)
    y = lorentzian_dip(x, 1.0, 0.8, 300e6, 12e6)
    ye = np.full_like(y, 0.01)

    fit = fit_double_lorentzian(x, y, ye, mode="dip")
    # components collapse onto each other -> rejected as degenerate
    assert fit is None


def test_too_few_points_returns_none():
    x = np.linspace(0, 1, 6)  # < 7 free params
    y = np.ones_like(x)
    assert fit_double_lorentzian(x, y, mode="dip") is None


def test_peak_mode_unsupported():
    x = _grid(280e6, 320e6, 41)
    y = np.ones_like(x)
    try:
        fit_double_lorentzian(x, y, mode="peak")
    except ValueError:
        return
    raise AssertionError("expected ValueError for mode='peak'")
