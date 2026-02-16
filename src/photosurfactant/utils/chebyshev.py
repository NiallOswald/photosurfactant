import numpy as np
from numpy.typing import NDArray


def chebyshev(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct the Chebyshev differentiation matrix on a grid of n points."""
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    c = (np.array([[2, *np.ones(n - 2), 2]]) * (-1) ** np.arange(n)).T
    X = np.tile(x[:, np.newaxis], (1, n))
    dX = X - X.T
    D = (c @ (1 / c).T) / (dX + np.eye(n))
    D -= np.diag(np.sum(D.T, axis=0))

    return D, x
