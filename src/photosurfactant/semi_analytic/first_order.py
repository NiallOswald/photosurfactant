"""First order solution to the photosurfactant model."""

from enum import Enum
from functools import wraps
from typing import Callable

import numpy as np

from photosurfactant.parameters import Parameters
from photosurfactant.semi_analytic.leading_order import LeadingOrder
from photosurfactant.semi_analytic.utils import Y, cosh, polyder, sinh, to_arr


class Symbols(Enum):  # TODO: This is unnecessary
    """Symbols used in the first order solution."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    gamma_tr = "Gamma_tr"
    gamma_ci = "Gamma_ci"
    J_tr = "J_tr"
    J_ci = "J_ci"
    S = "S"
    f = "f"


class Variables(object):  # TODO: Try to make this an enum
    """Variables used in the first order solution."""

    A = to_arr({Symbols.A: 1}, Symbols)
    B = to_arr({Symbols.B: 1}, Symbols)
    C = to_arr({Symbols.C: 1}, Symbols)
    D = to_arr({Symbols.D: 1}, Symbols)
    E = to_arr({Symbols.E: 1}, Symbols)
    F = to_arr({Symbols.F: 1}, Symbols)
    G = to_arr({Symbols.G: 1}, Symbols)
    H = to_arr({Symbols.H: 1}, Symbols)
    gamma_tr = to_arr({Symbols.gamma_tr: 1}, Symbols)
    gamma_ci = to_arr({Symbols.gamma_ci: 1}, Symbols)
    J_tr = to_arr({Symbols.J_tr: 1}, Symbols)
    J_ci = to_arr({Symbols.J_ci: 1}, Symbols)
    S = to_arr({Symbols.S: 1}, Symbols)
    f = to_arr({Symbols.f: 1}, Symbols)


class FirstOrder(object):
    """First order solution to the photosurfactant model."""

    def __init__(
        self,
        wavenumbers: np.ndarray,
        params: Parameters,
        leading: LeadingOrder,
    ):
        """Initalise solution to the first order model.

        :param wavenumbers: Array of wavenumbers.
        :param params: :class:`~.parameters.Parameters` object containing the
            model parameters.
        :param leading: :class:`~.leadingrder.LeadingOrder` object containing
            the leading order solution.
        """
        self.wavenumbers = wavenumbers
        self.params = params
        self.leading = leading

        self.solution = np.zeros([len(self.wavenumbers), len(Symbols)], dtype=complex)

    def solve(self, constraint: Callable[[int], tuple]):
        """Initialize the first order solution.

        :param constraint: Prescription to close the system. Should be a linear
            function in the given variables.
        """
        zeros = np.zeros([len(Symbols) - 1], dtype=complex)

        bc = BoundaryConditions(self)
        for n, k in enumerate(self.wavenumbers):
            # Formulate the boundary conditions
            sys = bc.formulate(k)

            # Apply prescrition
            cond, val = constraint(n)
            sys = np.vstack([sys, cond])

            # Solve the system of equations
            self.solution[n, :] = np.linalg.solve(sys, np.hstack([zeros, val]))

    def _invert(self, func):
        """Invert the function to real space."""

        @wraps(func)
        def wrapper(x, *args, x_order=0, **kwargs):
            vals = np.array([func(k, *args, **kwargs) for k in self.wavenumbers])
            coeffs = np.einsum("ij,ij->i", self.solution, vals)

            res = (
                coeffs[0] * (x_order == 0)
                + 2
                * np.sum(
                    (1.0j * self.wavenumbers[1:, np.newaxis]) ** x_order
                    * coeffs[1:, np.newaxis]
                    * np.exp(
                        1.0j
                        * self.wavenumbers[1:, np.newaxis]
                        * (x if isinstance(x, np.ndarray) else np.array([x]))[
                            np.newaxis, :
                        ]
                    ),
                    axis=0,
                )
            ).real

            return res if isinstance(x, np.ndarray) else res[0]

        return wrapper

    # Real space variables
    def psi(self, x, y, *, x_order=0, z_order=0):
        """Stream function at first order."""
        return self._invert(self._psi)(x, y, x_order=x_order, z_order=z_order)

    def u(self, x, y, *, x_order=0, z_order=0):
        """Horizontal velocity at first order."""
        return self.psi(x, y, x_order=x_order, z_order=z_order + 1)

    def w(self, x, y, *, x_order=0, z_order=0):
        """Vertical velocity at first order."""
        return -self.psi(x, y, x_order=x_order + 1, z_order=z_order)

    def p(self, x, y, *, x_order=0, z_order=0):
        """Pressure at first order."""
        return self._invert(self._p)(x, y, x_order=x_order, z_order=z_order)

    def c_tr(self, x, y, *, x_order=0, z_order=0):
        """Concentration of trans surfactant at first order."""
        return self._invert(self._c_tr)(x, y, x_order=x_order, z_order=z_order)

    def c_ci(self, x, y, *, x_order=0, z_order=0):
        """Concentration of cis surfactant at first order."""
        return self._invert(self._c_ci)(x, y, x_order=x_order, z_order=z_order)

    def i_c_tr(self, x, y, y_s=0, *, x_order=0):
        """Integral of trans surfactant concentration at first order."""
        return self._invert(self._c_tr_i)(x, y, y_s, x_order=x_order)

    def i_c_ci(self, x, y, y_s=0, *, x_order=0):
        """Integral of cis surfactant concentration at first order."""
        return self._invert(self._c_ci_i)(x, y, y_s, x_order=x_order)

    def Gamma_tr(self, x, *, x_order=0):
        """Surface excess of trans surfactant at first order."""
        return self._invert(lambda k: Variables.gamma_tr)(x, x_order=x_order)

    def Gamma_ci(self, x, *, x_order=0):
        """Surface excess of cis surfactant at first order."""
        return self._invert(lambda k: Variables.gamma_ci)(x, x_order=x_order)

    def J_tr(self, x, *, x_order=0):
        """Kinetic flux of trans surfactant at first order."""
        return self._invert(lambda k: Variables.J_tr)(x, x_order=x_order)

    def J_ci(self, x, *, x_order=0):
        """Kinetic flux of cis surfactant at first order."""
        return self._invert(lambda k: Variables.J_ci)(x, x_order=x_order)

    def S(self, x, *, x_order=0):
        """Interface shape at first order."""
        return self._invert(lambda k: Variables.S)(x, x_order=x_order)

    def f(self, x, *, x_order=0):
        """Light intensity at first order."""
        return self._invert(lambda k: Variables.f)(x, x_order=x_order)

    def gamma(self, x, *, x_order=0):
        """Surface tension at first order."""
        return self._invert(lambda k: self._gamma)(x, x_order=x_order)

    # Fourier space variables
    def _psi(self, k, z, z_order=0):
        """Stream function at first order in Fourier space."""
        if k == 0:
            return (
                Variables.A * polyder(Y**3, z_order)(z)
                + Variables.B * polyder(Y**2, z_order)(z)
                + Variables.C * polyder(Y, z_order)(z)
                + Variables.D * polyder(Y**0, z_order)(z)
            )
        else:
            return (
                Variables.A * k ** (z_order - 1) * (z_order + k * z) * np.exp(k * z)
                + Variables.B * k**z_order * np.exp(k * z)
                + Variables.C
                * (-k) ** (z_order - 1)
                * (z_order - k * z)
                * np.exp(-k * z)
                + Variables.D * (-k) ** z_order * np.exp(-k * z)
            )

    def _p(self, k, z, z_order=0):
        """Pressure at first order in Fourier space."""
        if k == 0:
            return 0 * Variables.f
        else:
            return (
                self._psi(k, z, z_order=z_order + 3)
                - k**2 * self._psi(k, z, z_order=z_order + 1)
            ) / (1.0j * k)

    def _c_tr(self, k, z, z_order=0):
        """Concentration of trans surfactant at first order in Fourier space."""
        return self._c(k, z, z_order)[0, ...]

    def _c_ci(self, k, z, z_order=0):
        """Concentration of cis surfactant at first order in Fourier space."""
        return self._c(k, z, z_order)[1, ...]

    def _c(self, k, z, z_order=0):
        """Concentration of surfactant at first order in Fourier space."""
        return self.params.V @ self._q(k, z, z_order)

    def _c_tr_i(self, k, z, z_s=0):
        """Integral of the trans concentration at first order in Fourier space."""
        return self._c_i(k, z, z_s)[0, ...]

    def _c_ci_i(self, k, z, z_s=0):
        """Integral of the cis concentration at first order in Fourier space."""
        return self._c_i(k, z, z_s)[1, ...]

    def _c_i(self, k, z, z_s=0):
        """Integral of the surfactant concentration at first order in Fourier space."""
        return np.einsum("ij,j...->i...", self.params.V, self._q_i(k, z, z_s))

    @property
    def _gamma(self):
        """Surface tension at first order in Fourier space."""
        return (
            -self.params.Ma
            * (Variables.gamma_tr + Variables.gamma_ci)
            / (1 - self.leading.Gamma_tr - self.leading.Gamma_ci)
        )

    # Private variables
    def _q(self, k, z, z_order=0):
        return (
            self._q_0(k, z, z_order)
            + self._q_1(k, z, z_order)
            + self._q_2(k, z, z_order)
        )

    def _q_i(self, k, z, z_s=0):
        return self._q(k, z, z_order=-1) - self._q(k, z_s, z_order=-1)

    def _q_0(self, k, z, z_order=0):
        zeta = self.params.zeta
        if k == 0:
            return np.array(
                [
                    Variables.E * polyder(Y, z_order)(z)
                    + Variables.F * polyder(Y**0, z_order)(z),
                    np.sqrt(zeta) ** z_order
                    * (
                        Variables.G * sinh(z * np.sqrt(zeta), z_order)
                        + Variables.H * cosh(z * np.sqrt(zeta), z_order)
                    ),
                ]
            )
        else:
            return np.array(
                [
                    k**z_order
                    * (
                        Variables.E * sinh(k * z, z_order)
                        + Variables.F * cosh(k * z, z_order)
                    ),
                    np.sqrt(zeta + k**2) ** z_order
                    * (
                        Variables.G * sinh(z * np.sqrt(zeta + k**2), z_order)
                        + Variables.H * cosh(z * np.sqrt(zeta + k**2), z_order)
                    ),
                ]
            )

    def _q_1(self, k, z, z_order=0):
        zeta = self.params.zeta
        if k == 0:
            return (
                Variables.f
                * self.leading.B
                * np.sqrt(zeta) ** z_order
                / 2
                * (
                    z_order * cosh(z * np.sqrt(zeta), z_order)
                    + z * np.sqrt(zeta) * sinh(z * np.sqrt(zeta), z_order)
                )
            )[np.newaxis, :] * np.array([0, 1])[:, np.newaxis]
        else:
            return (
                Variables.f
                * np.sqrt(zeta) ** z_order
                * -self.leading.B
                * zeta
                / k**2
                * cosh(z * np.sqrt(zeta), z_order)
            )[np.newaxis, :] * np.array([0, 1])[:, np.newaxis]

    def _q_2(self, k, z, z_order=0):
        alpha, eta, zeta = self.params.alpha, self.params.eta, self.params.zeta
        null = to_arr(dict(), Symbols)
        if k == 0:
            return np.array([null, null])
        else:
            return (
                -1.0j
                * k
                * self.leading.B
                * np.sqrt(zeta)
                * self.params.Pe_ci
                / (2 * (alpha + eta))
                * np.einsum(
                    "ij...,j->i...",
                    np.array(
                        [
                            [self._q_2_scalar(k, z, 0, z_order), null],
                            [null, self._q_2_scalar(k, z, 1, z_order)],
                        ]
                    ),
                    np.array(
                        [
                            eta**2 - eta,
                            eta**2 + alpha,
                        ]
                    ),
                )
            )

    def _q_2_scalar(self, k, z, index, z_order=0):
        return (
            self._q_2_gfunc(
                k, z, self._a_c(k, index), self._b_c(k, index), 1, 1, z_order
            )
            + self._q_2_gfunc(
                k, z, self._c_c(k, index), self._d_c(k, index), 1, -1, z_order
            )
            + self._q_2_gfunc(
                k, z, self._e_c(k, index), self._f_c(k, index), -1, -1, z_order
            )
            + self._q_2_gfunc(
                k, z, self._g_c(k, index), self._h_c(k, index), -1, 1, z_order
            )
        )

    def _q_2_gfunc(self, k, z, a, b, s_1, s_2, z_order=0):
        return (
            z_order * self._q_2_base(k, s_1, s_2) ** (z_order - 1) * a
            + self._q_2_base(k, s_1, s_2) ** z_order * (a * z + b)
        ) * np.exp(self._q_2_base(k, s_1, s_2) * z)

    def _q_2_base(self, k, s_1, s_2):
        return s_1 * (k + s_2 * np.sqrt(self.params.zeta))

    def _a_c(self, k, index):
        zeta = self.params.zeta
        return Variables.A / (2 * k * np.sqrt(zeta) + zeta * (index == 0))

    def _c_c(self, k, index):
        zeta = self.params.zeta
        return Variables.A / (2 * k * np.sqrt(zeta) - zeta * (index == 0))

    def _e_c(self, k, index):
        zeta = self.params.zeta
        return -Variables.C / (2 * k * np.sqrt(zeta) - zeta * (index == 0))

    def _g_c(self, k, index):
        zeta = self.params.zeta
        return -Variables.C / (2 * k * np.sqrt(zeta) + zeta * (index == 0))

    def _b_c(self, k, index):
        zeta = self.params.zeta
        return (Variables.B - 2 * self._a_c(k, index) * (k + np.sqrt(zeta))) / (
            2 * k * np.sqrt(zeta) + zeta * (index == 0)
        )

    def _d_c(self, k, index):
        zeta = self.params.zeta
        return (Variables.B + 2 * self._c_c(k, index) * (k - np.sqrt(zeta))) / (
            2 * k * np.sqrt(zeta) - zeta * (index == 0)
        )

    def _f_c(self, k, index):
        zeta = self.params.zeta
        return -(Variables.D + 2 * self._e_c(k, index) * (k - np.sqrt(zeta))) / (
            2 * k * np.sqrt(zeta) - zeta * (index == 0)
        )

    def _h_c(self, k, index):
        zeta = self.params.zeta
        return -(Variables.D - 2 * self._g_c(k, index) * (k + np.sqrt(zeta))) / (
            2 * k * np.sqrt(zeta) + zeta * (index == 0)
        )


class BoundaryConditions(object):
    """Boundary conditions for the photosurfactant model."""

    def __init__(self, first: FirstOrder):
        """Initialise boundary conditions for the photosurfactant model.

        :param first: :class:`~.first_order.FirstOrder` object containing the
            first order solution.
        """
        self.params = first.params
        self.leading = first.leading
        self.first = first

    def formulate(self, k):
        """Formulate the boundary conditions."""
        return np.vstack(
            [
                self.no_slip(k),
                self.normal_stress(k),
                self.tangential_stress(k),
                self.kinematic(k),
                self.no_flux(k),
                self.kinetic_flux(k),
                self.surface_excess(k),
                self.mass_balance(k),
            ]
        )

    def no_slip(self, k):
        """No slip boundary condition."""
        return np.array(
            [
                self.first._psi(k, 0),
                self.first._psi(k, 0, z_order=1),
            ]
        )

    def normal_stress(self, k):
        """Normal stress boundary condition."""  # noqa: D401
        if k == 0:
            return self.first._psi(k, 1, z_order=3)[np.newaxis, ...]
        else:
            return (
                self.first._psi(k, 1, z_order=3)
                - 3 * k**2 * self.first._psi(k, 1, z_order=1)
                - 1.0j * k**3 * self.leading.gamma * Variables.S
            )[np.newaxis, ...]

    def tangential_stress(self, k):
        """Tangential stress boundary condition."""
        if k == 0:
            return self.first._psi(k, 1, z_order=2)[np.newaxis, ...]
        else:
            return (
                self.first._psi(k, 1, z_order=2)
                + k**2 * self.first._psi(k, 1)
                - 1.0j * k * self.first._gamma
            )[np.newaxis, ...]

    def kinematic(self, k):
        """Kinematic boundary condition."""
        if k == 0:
            # Replace with conservation of mass
            return Variables.S[np.newaxis, ...]
        else:
            return self.first._psi(k, 1)[np.newaxis, ...]

    def no_flux(self, k):
        """No flux boundary condition."""
        return self.first._c(k, 0, z_order=1)

    def kinetic_flux(self, k):
        """Kinetic flux boundary condition."""
        return self.params.B @ (
            self.params.K
            @ (
                (
                    self.first._c(k, 1)
                    + Variables.S[np.newaxis, :]
                    * self.leading.c(1, z_order=1)[:, np.newaxis]
                )
                * (1 - self.leading.Gamma_tr - self.leading.Gamma_ci)
                - self.leading.c(1)[:, np.newaxis]
                * (Variables.gamma_tr + Variables.gamma_ci)
            )
            - np.array([Variables.gamma_tr, Variables.gamma_ci])
        ) - np.array([Variables.J_tr, Variables.J_ci])

    def surface_excess(self, k):
        """Surface excess boundary condition."""
        gamma_vec = np.array([Variables.gamma_tr, Variables.gamma_ci])
        J_vec = np.array([Variables.J_tr, Variables.J_ci])

        d_psi_vec = self.first._psi(k, 1, z_order=1)

        return (
            1.0j
            * k
            * (self.params.P_s @ self.leading.Gamma)[:, np.newaxis]
            * d_psi_vec[np.newaxis, :]  # Fix this line
            + k**2 * gamma_vec
            - self.params.P_s @ J_vec
            + self.params.A_s
            @ (
                np.array([Variables.gamma_tr, Variables.gamma_ci])
                + Variables.f[np.newaxis, :] * self.leading.Gamma[:, np.newaxis]
            )
        )

    def mass_balance(self, k):
        """Mass balance boundary condition."""
        cond = (self.params.k_tr * self.params.chi_tr) * (
            self.first._c(k, 1, z_order=1)
            + Variables.S[np.newaxis, :] * self.leading.c(1, z_order=2)[:, np.newaxis]
        ) + self.params.P @ np.array([Variables.J_tr, Variables.J_ci])

        if k == 0:
            # Replace one mass balance with conservation of surfactant
            cond[0, ...] = (
                1
                / (self.params.k_tr * self.params.chi_tr)
                * (Variables.gamma_tr + Variables.gamma_ci)
                + self.first._c_tr_i(k, 1)
                + self.first._c_ci_i(k, 1)
            )

        return cond
