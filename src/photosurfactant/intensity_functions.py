"""Example light intensity and surface perturbations."""

from typing import Callable

import numpy as np
import scipy as sp


def gaussian(x: float, d=1.0) -> float:
    return super_gaussian(x, 2.0, d)


def super_gaussian(x: float, k: float, d=1.0) -> float:
    return np.exp(-(abs(x / d) ** k))


def square_wave(x: float) -> float:
    return float(abs(x) < 1)


def smoothed_square(x: float, delta: float) -> float:
    return 0.5 * (np.tanh((x + 1) / delta) - np.tanh((x - 1) / delta))


def mollifier(delta: float) -> Callable[[float], float]:
    def _(x):
        if abs(x) < delta:
            return np.exp(-(delta**2) / (delta**2 - x**2))
        else:
            return 0.0

    return lambda x: _(x) / sp.integrate.quad(_, -1.0, 1.0)[0]
