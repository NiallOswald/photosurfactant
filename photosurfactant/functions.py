"""Example light intensity and surface perturbations."""

import numpy as np


def laser_pointer(delta, sign=1.0):
    """Return a laser-pointer-like function."""

    def _(x):
        return sign * 0.5 * (np.tanh((x + 1) / delta) - np.tanh((x - 1) / delta))

    return _
