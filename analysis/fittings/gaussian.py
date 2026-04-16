"""Gaussian peak model."""

import numpy as np


def gaussian_peak(x, y0, A, x0, sigma):
    """Gaussian peak: y = y0 + A * exp(-(x-x0)^2 / (2*sigma^2))"""
    return y0 + A * np.exp(-(x - x0)**2 / (2 * sigma**2))
