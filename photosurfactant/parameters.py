"""A module for the Parameters class."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    """The Parameters for the model.

    :param L: The aspect ratio of the domain.
    :param Dam_tr: The Damkohler number for the trans surfactant.
    :param Dam_ci: The Damkohler number for the cis surfactant.
    :param Pen_tr: The Peclet number for the trans surfactant.
    :param Pen_ci: The Peclet number for the cis surfactant.
    :param Pen_tr_s: The Peclet number for the trans surfactant on the
        interface.
    :param Pen_ci_s: The Peclet number for the cis surfactant on the interface.
    :param Bit_tr: The Biot number for the trans surfactant.
    :param Bit_ci: The Biot number for the cis surfactant.
    :param Man: The Marangoni number.
    :param k_tr: The adsorption rate for the trans surfactant.
    :param k_ci: The adsorption rate for the cis surfactant.
    :param chi_tr: The desorption rate for the trans surfactant.
    :param chi_ci: The desorption rate for the cis surfactant.
    """

    # Aspect ratio
    L: float = 10.0

    # Damkohler numbers
    Dam_tr: float = 1.0
    Dam_ci: float = 2.0

    # Peclet numbers
    Pen_tr: float = 10.0
    Pen_ci: float = 10.0

    # Interfacial Peclet numbers
    Pen_tr_s: float = 10.0
    Pen_ci_s: float = 10.0

    # Biot numbers
    Bit_tr: float = 3.33
    Bit_ci: float = 1.0e3

    # Marangoni number
    Man: float = 2.0

    # Adsorption and desorption rates
    k_tr: float = 30.0
    k_ci: float = 1.0

    chi_tr: float = 1.0
    chi_ci: float = 30.0

    def __post_init__(self):  # noqa: D105
        if self.k_tr * self.chi_tr != self.k_ci * self.chi_ci:
            raise ValueError(
                "Adsorption rates do not satisfy the condition " "k * chi = const."
            )

        self.alpha = self.Dam_ci / self.Dam_tr
        self.eta = self.Pen_tr / self.Pen_ci
        self.zeta = self.Pen_tr * self.Dam_tr + self.Pen_ci * self.Dam_ci

        self.D = np.array([[self.Dam_tr, -self.Dam_ci], [-self.Dam_tr, self.Dam_ci]])
        self.P = np.array([[self.Pen_tr, 0.0], [0.0, self.Pen_ci]])
        self.P_s = np.array([[self.Pen_tr_s, 0.0], [0.0, self.Pen_ci_s]])
        self.B = np.array([[self.Bit_tr, 0.0], [0.0, self.Bit_ci]])
        self.K = np.array([[self.k_tr, 0.0], [0.0, self.k_ci]])

        self.A = self.P @ self.D
        self.A_s = self.P_s @ self.D

        self.V = np.array([[self.alpha, self.eta], [1.0, -1.0]])
        self.Lambda = np.array([[0.0, 0.0], [0.0, self.zeta]])
