"""Example light intensity and surface perturbations."""

import numpy as np
import scipy as sp


def gaussian(x):
    return np.exp(-4 * x**2)


def square_wave(x):
    return float(abs(x) < 1)


def smoothed_square(x, delta):
    return 0.5 * (np.tanh((x + 1) / delta) - np.tanh((x - 1) / delta))


def mollifier(delta):
    def _(x):
        if abs(x) < delta:
            return np.exp(-(delta**2) / (delta**2 - x**2))
        else:
            return 0.0

    return lambda x: _(x) / sp.integrate.quad(_, -1.0, 1.0)[0]
