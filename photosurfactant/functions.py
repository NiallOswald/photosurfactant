"""Example light intensity and surface perturbations."""

import numpy as np


def laser_square(x, delta):
    """Return a smoothed square wave."""
    return 0.5 * (np.tanh((x + 1) / delta) - np.tanh((x - 1) / delta))


def laser_gauss(x):
    """Return a Gaussian-like function."""
    return np.exp(-4 * x**2)
