"""Utility functions for the semi-analytic model."""

from enum import Enum
from typing import Callable

import numpy as np
import numpy.typing as npt

Y = np.poly1d([1, 0])  # Polynomial for differentiation


def hyperder(n: int) -> Callable[[float], float]:
    """Return the n-th derivative of cosh."""
    return np.sinh if n % 2 else np.cosh


def cosh(x: float, x_order: int = 0) -> float:
    return hyperder(x_order)(x)


def sinh(x: float, x_order: int = 0) -> float:
    return hyperder(x_order + 1)(x)


def polyder(p: np.poly1d, n: int) -> np.poly1d:
    """Wrap np.polyder and np.polyint."""
    return p.deriv(n) if n > 0 else p.integ(-n)


def to_arr(vals: dict, unknowns: Enum) -> npt.NDArray:
    """Convert a dictionary of values to an array."""
    arr = np.zeros(len(unknowns), dtype=complex)
    for key in vals.keys():
        if key not in unknowns:
            raise ValueError(f"Unknown key: {key}")

    for i, key in enumerate(unknowns):
        try:
            arr[i] = vals[key]
        except KeyError:
            pass

    return arr
