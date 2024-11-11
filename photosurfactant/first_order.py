"""First order solution to the photosurfactant model."""

from .parameters import Parameters
from .leading_order import LeadingOrder
from abc import ABC, abstractmethod
import numpy as np


class FourierVariables(ABC):
    """Variables for the first order solution in Fourier space."""

    def __init__(self, omega, params: Parameters, leading: LeadingOrder):
        """Initialize the Fourier variables.

        :param omega: Array of angular frequencies.
        :param params: :class:`~.parameters.Parameters` object containing the
            model parameters.
        :param leading: :class:`~.leadingrder.LeadingOrder` object containing
            the leading order solution.
        """
        self.omega = omega
        self.params = params
        self.leading = leading

    @staticmethod
    def to_arr(vals, omega, unknowns):
        """Convert a dictionary of values to an array."""
        arr = np.zeros((len(unknowns), len(omega)), dtype=complex)
        for key in vals.keys():
            if key not in unknowns:
                raise ValueError(f"Unknown key: {key}")

        for i, key in enumerate(unknowns):
            try:
                arr[i, :] = vals[key]
            except KeyError:
                pass

        return arr

    @property
    @abstractmethod
    def unknowns(self):
        """Return the unknown variables."""
        pass

    def __getitem__(self, key):
        """Return the index of an unknown variable."""
        return self.unknowns.index(key)


class NonZeroFourierVariables(FourierVariables):
    """Variables for the non-zero frequiences in Fourier space."""

    unknowns = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "gamma_tr",
        "gamma_ci",
        "f",
        "S",
        "J_tr",
        "J_ci",
        "const",
    ]

    def to_arr(self, vals):
        """Convert a dictionary of unknowns to a vector."""
        return super().to_arr(vals, self.omega, self.unknowns)

    @property
    def Lambda_inv(self):
        const = self.to_arr({"const": 1})

        return np.array(
            [
                [1 / self.omega**2 * const, 0 * const],
                [0 * const, 1 / (self.omega**2 + self.params.zeta) * const],
            ]
        )

    def psi(self, y):
        return self.to_arr(
            {
                "A": y * np.exp(self.omega * y),
                "B": np.exp(self.omega * y),
                "C": y * np.exp(-self.omega * y),
                "D": np.exp(-self.omega * y),
            }
        )

    def d_psi(self, y):
        return self.to_arr(
            {
                "A": (1 + self.omega * y) * np.exp(self.omega * y),
                "B": self.omega * np.exp(self.omega * y),
                "C": (1 - self.omega * y) * np.exp(-self.omega * y),
                "D": -self.omega * np.exp(-self.omega * y),
            }
        )

    def d2_psi(self, y):
        return self.to_arr(
            {
                "A": self.omega * (2 + self.omega * y) * np.exp(self.omega * y),
                "B": self.omega**2 * np.exp(self.omega * y),
                "C": -self.omega * (2 - self.omega * y) * np.exp(-self.omega * y),
                "D": self.omega**2 * np.exp(-self.omega * y),
            }
        )

    def d3_psi(self, y):
        return self.to_arr(
            {
                "A": self.omega**2 * (3 + self.omega * y) * np.exp(self.omega * y),
                "B": self.omega**3 * np.exp(self.omega * y),
                "C": self.omega**2 * (3 - self.omega * y) * np.exp(-self.omega * y),
                "D": -self.omega**3 * np.exp(-self.omega * y),
            }
        )

    @property
    def a_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) + params.zeta)
        ) * self.to_arr(
            {
                "A": 1,
            }
        )

    @property
    def b_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) + params.zeta) ** 2
        ) * self.to_arr(
            {
                "A": -2 * (self.omega + np.sqrt(params.zeta)),
                "B": 2 * self.omega * np.sqrt(params.zeta) + params.zeta,
            }
        )

    @property
    def c_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) - params.zeta)
        ) * self.to_arr(
            {
                "A": 1,
            }
        )

    @property
    def d_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) - params.zeta) ** 2
        ) * self.to_arr(
            {
                "A": 2 * (self.omega - np.sqrt(params.zeta)),
                "B": 2 * self.omega * np.sqrt(params.zeta) - params.zeta,
            }
        )

    @property
    def e_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) - params.zeta)
        ) * self.to_arr(
            {
                "C": -1,
            }
        )

    @property
    def f_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) - params.zeta) ** 2
        ) * self.to_arr(
            {
                "C": 2 * (self.omega - np.sqrt(params.zeta)),
                "D": -(2 * self.omega * np.sqrt(params.zeta) - params.zeta),
            }
        )

    @property
    def g_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) + params.zeta)
        ) * self.to_arr(
            {
                "C": -1,
            }
        )

    @property
    def h_c_1(self):
        params = self.params
        return (
            1 / (2 * self.omega * np.sqrt(params.zeta) + params.zeta) ** 2
        ) * self.to_arr(
            {
                "C": -2 * (self.omega + np.sqrt(params.zeta)),
                "D": -(2 * self.omega * np.sqrt(params.zeta) + params.zeta),
            }
        )

    @property
    def a_c_2(self):
        return (1 / (2 * self.omega * np.sqrt(self.params.zeta))) * self.to_arr(
            {
                "A": 1,
            }
        )

    @property
    def b_c_2(self):
        params = self.params
        return (1 / (2 * self.omega * np.sqrt(params.zeta)) ** 2) * self.to_arr(
            {
                "A": -2 * (self.omega + np.sqrt(params.zeta)),
                "B": 2 * self.omega * np.sqrt(params.zeta),
            }
        )

    @property
    def c_c_2(self):
        return (1 / (2 * self.omega * np.sqrt(self.params.zeta))) * self.to_arr(
            {
                "A": 1,
            }
        )

    @property
    def d_c_2(self):
        params = self.params
        return (1 / (2 * self.omega * np.sqrt(params.zeta)) ** 2) * self.to_arr(
            {
                "A": 2 * (self.omega - np.sqrt(params.zeta)),
                "B": 2 * self.omega * np.sqrt(params.zeta),
            }
        )

    @property
    def e_c_2(self):
        return (1 / (2 * self.omega * np.sqrt(self.params.zeta))) * self.to_arr(
            {
                "C": -1,
            }
        )

    @property
    def f_c_2(self):
        params = self.params
        return (1 / (2 * self.omega * np.sqrt(params.zeta)) ** 2) * self.to_arr(
            {
                "C": 2 * (self.omega - np.sqrt(params.zeta)),
                "D": -2 * self.omega * np.sqrt(params.zeta),
            }
        )

    @property
    def g_c_2(self):
        return (1 / (2 * self.omega * np.sqrt(self.params.zeta))) * self.to_arr(
            {
                "C": -1,
            }
        )

    @property
    def h_c_2(self):
        params = self.params
        return (1 / (2 * self.omega * np.sqrt(params.zeta)) ** 2) * self.to_arr(
            {
                "C": -2 * (self.omega + np.sqrt(params.zeta)),
                "D": -2 * self.omega * np.sqrt(params.zeta),
            }
        )

    def p_0(self, y):
        eq_1 = self.to_arr(
            {
                "E": np.sinh(self.omega * y),
                "F": np.cosh(self.omega * y),
            }
        )
        eq_2 = self.to_arr(
            {
                "G": np.sinh(np.sqrt(self.params.zeta + self.omega**2) * y),
                "H": np.cosh(np.sqrt(self.params.zeta + self.omega**2) * y),
            }
        )

        return np.array([eq_1, eq_2])

    def p_1(self, y):
        vec = (self.params.Lambda @ np.linalg.inv(self.params.V)) @ np.array(
            [self.leading.A_0, 0]
        )
        eq_1 = np.einsum("ij...,j->i...", self.Lambda_inv, vec)
        f_vec = self.to_arr({"f": 1})

        return -np.einsum("i...,j...->ij...", eq_1[:, -1], f_vec)

    def p_2(self, y):
        params = self.params
        leading = self.leading

        p_1 = self.to_arr(
            {
                "const": -(
                    leading.A_0 / ((params.alpha + params.eta) * self.omega**2)
                    + leading.B_0
                    / (self.omega**2 - params.zeta)
                    * np.cosh(y * np.sqrt(params.zeta))
                ),
            }
        )
        p_2 = self.to_arr(
            {
                "const": -(
                    leading.A_0
                    / ((params.alpha + params.eta) * (self.omega**2 + params.zeta))
                    + leading.B_0 / self.omega**2 * np.cosh(y * np.sqrt(params.zeta))
                ),
            }
        )
        zero = self.to_arr(dict())

        matr = np.array([[p_1, zero], [zero, p_2]])
        vec = params.Lambda @ np.linalg.inv(params.V) @ np.array([-params.eta, 1.0])
        vec_2 = np.einsum("ij...,j->i...", matr, vec)

        f_vec = self.to_arr({"f": 1})

        return np.einsum("i...,j...->ij...", vec_2[:, -1], f_vec)

    def p_3_1(self, y):
        return (
            (self.a_c_1 * y + self.b_c_1)
            * np.exp((self.omega + np.sqrt(self.params.zeta)) * y)
            + (self.c_c_1 * y + self.d_c_1)
            * np.exp((self.omega - np.sqrt(self.params.zeta)) * y)
            + (self.e_c_1 * y + self.f_c_1)
            * np.exp(-(self.omega - np.sqrt(self.params.zeta)) * y)
            + (self.g_c_1 * y + self.h_c_1)
            * np.exp(-(self.omega + np.sqrt(self.params.zeta)) * y)
        )

    def p_3_2(self, y):
        return (
            (self.a_c_2 * y + self.b_c_2)
            * np.exp((self.omega + np.sqrt(self.params.zeta)) * y)
            + (self.c_c_2 * y + self.d_c_2)
            * np.exp((self.omega - np.sqrt(self.params.zeta)) * y)
            + (self.e_c_2 * y + self.f_c_2)
            * np.exp(-(self.omega - np.sqrt(self.params.zeta)) * y)
            + (self.g_c_2 * y + self.h_c_2)
            * np.exp(-(self.omega + np.sqrt(self.params.zeta)) * y)
        )

    def p_3(self, y):
        params = self.params

        zero = self.to_arr(dict())
        matr = np.array([[self.p_3_1(y), zero], [zero, self.p_3_2(y)]])
        vec = np.linalg.inv(params.V) @ params.P @ np.array([-params.eta, 1])

        return (
            -(1.0j * self.omega * self.leading.B_0 * np.sqrt(params.zeta))
            / 2
            * np.einsum("ij...,j->i...", matr, vec)
        )

    def p(self, y):
        return self.p_0(y) + self.p_1(y) + self.p_2(y) + self.p_3(y)

    def d_p_0(self, y):
        params = self.params

        eq_1 = self.omega * self.to_arr(
            {
                "E": np.cosh(self.omega * y),
                "F": np.sinh(self.omega * y),
            }
        )
        eq_2 = np.sqrt(params.zeta + self.omega**2) * self.to_arr(
            {
                "G": np.cosh(np.sqrt(params.zeta + self.omega**2) * y),
                "H": np.sinh(np.sqrt(params.zeta + self.omega**2) * y),
            }
        )

        return np.array([eq_1, eq_2])

    def d_p_2(self, y):
        params = self.params
        leading = self.leading

        d_p_1 = self.to_arr(
            {
                "const": -leading.B_0
                * np.sqrt(params.zeta)
                / (self.omega**2 - params.zeta)
                * np.sinh(y * np.sqrt(params.zeta)),
            }
        )
        d_p_2 = self.to_arr(
            {
                "const": -leading.B_0
                * np.sqrt(params.zeta)
                / self.omega**2
                * np.sinh(y * np.sqrt(params.zeta)),
            }
        )
        zero = self.to_arr(dict())

        matr = np.array([[d_p_1, zero], [zero, d_p_2]])
        vec = params.Lambda @ np.linalg.inv(params.V) @ np.array([-params.eta, 1])
        vec_2 = np.einsum("ij...,j->i...", matr, vec)

        f_vec = self.to_arr({"f": 1})

        return np.einsum("i...,j...->ij...", vec_2[:, -1], f_vec)

    def d_p_3_1(self, y):
        params = self.params

        return (
            self.a_c_1 * np.exp((self.omega + np.sqrt(params.zeta)) * y)
            + self.c_c_1 * np.exp((self.omega - np.sqrt(params.zeta)) * y)
            + self.e_c_1 * np.exp(-(self.omega - np.sqrt(params.zeta)) * y)
            + self.g_c_1 * np.exp(-(self.omega + np.sqrt(params.zeta)) * y)
        ) + (
            (self.omega + np.sqrt(params.zeta))
            * (self.a_c_1 * y + self.b_c_1)
            * np.exp((self.omega + np.sqrt(params.zeta)) * y)
            + (self.omega - np.sqrt(params.zeta))
            * (self.c_c_1 * y + self.d_c_1)
            * np.exp((self.omega - np.sqrt(params.zeta)) * y)
            - (self.omega - np.sqrt(params.zeta))
            * (self.e_c_1 * y + self.f_c_1)
            * np.exp(-(self.omega - np.sqrt(params.zeta)) * y)
            - (self.omega + np.sqrt(params.zeta))
            * (self.g_c_1 * y + self.h_c_1)
            * np.exp(-(self.omega + np.sqrt(params.zeta)) * y)
        )

    def d_p_3_2(self, y):
        params = self.params

        return (
            self.a_c_2 * np.exp((self.omega + np.sqrt(params.zeta)) * y)
            + self.c_c_2 * np.exp((self.omega - np.sqrt(params.zeta)) * y)
            + self.e_c_2 * np.exp(-(self.omega - np.sqrt(params.zeta)) * y)
            + self.g_c_2 * np.exp(-(self.omega + np.sqrt(params.zeta)) * y)
        ) + (
            (self.omega + np.sqrt(params.zeta))
            * (self.a_c_2 * y + self.b_c_2)
            * np.exp((self.omega + np.sqrt(params.zeta)) * y)
            + (self.omega - np.sqrt(params.zeta))
            * (self.c_c_2 * y + self.d_c_2)
            * np.exp((self.omega - np.sqrt(params.zeta)) * y)
            - (self.omega - np.sqrt(params.zeta))
            * (self.e_c_2 * y + self.f_c_2)
            * np.exp(-(self.omega - np.sqrt(params.zeta)) * y)
            - (self.omega + np.sqrt(params.zeta))
            * (self.g_c_2 * y + self.h_c_2)
            * np.exp(-(self.omega + np.sqrt(params.zeta)) * y)
        )

    def d_p_3(self, y):
        params = self.params

        zero = self.to_arr(dict())
        matr = np.array([[self.d_p_3_1(y), zero], [zero, self.d_p_3_2(y)]])
        vec = np.linalg.inv(params.V) @ params.P @ np.array([-params.eta, 1])

        return (
            -(1.0j * self.omega * self.leading.B_0 * np.sqrt(params.zeta))
            / 2
            * np.einsum("ij...,j->i...", matr, vec)
        )

    def d_p(self, y):
        return self.d_p_0(y) + self.d_p_2(y) + self.d_p_3(y)

    def c(self, y):
        return np.einsum("ij,j...->i...", self.params.V, self.p(y))

    def d_c(self, y):
        return np.einsum("ij,j...->i...", self.params.V, self.d_p(y))


class ZeroFourierVariables(FourierVariables):
    """Variables for the zero frequency in Fourier space."""

    unknowns = [
        "A",
        "B",
        "C",
        "A_1",
        "B_1",
        "gamma_tr",
        "gamma_ci",
        "f",
        "J_tr",
        "J_ci",
        "const",
    ]

    def to_arr(self, vals):
        return super().to_arr(vals, [0], self.unknowns)

    def psi(self, y):
        return self.to_arr(
            {
                "A": y**2,
                "B": y,
                "C": 1,
            }
        )

    def d_psi(self, y):
        return self.to_arr(
            {
                "A": 2 * y,
                "B": 1,
            }
        )

    def d2_psi(self, y):
        return self.to_arr(
            {
                "A": 2,
            }
        )

    def c_ci(self, y):
        params = self.params
        return self.to_arr(
            {
                "A_1": 1 / (params.alpha + params.eta),
                "B_1": np.cosh(y * np.sqrt(params.zeta)),
                "f": (self.leading.B_0 * np.sqrt(params.zeta) / 2)
                * y
                * np.sinh(y * np.sqrt(params.zeta)),
            }
        )

    def c_tr(self, y):
        return self.to_arr({"A_1": 1}) - self.params.eta * self.c_ci(y)

    def c(self, y):
        return np.array([self.c_tr(y), self.c_ci(y)])

    def d_c_ci(self, y):
        params = self.params
        return self.to_arr(
            {
                "B_1": np.sqrt(params.zeta) * np.sinh(y * np.sqrt(params.zeta)),
                "f": (self.leading.B_0 * np.sqrt(params.zeta) / 2)
                * (
                    np.sinh(y * np.sqrt(params.zeta))
                    + y * np.sqrt(params.zeta) * np.cosh(y * np.sqrt(params.zeta))
                ),
            }
        )

    def d_c_tr(self, y):
        return -self.params.eta * self.d_c_ci(y)

    def d_c(self, y):
        return np.array([self.d_c_tr(y), self.d_c_ci(y)])

    @property
    def i_c_ci(self):
        params = self.params
        return self.to_arr(
            {
                "A_1": 1 / (params.alpha + params.eta),
                "B_1": np.sinh(np.sqrt(params.zeta)) / np.sqrt(params.zeta),
                "f": (self.leading.B_0 / 2)
                * (
                    np.cosh(np.sqrt(params.zeta))
                    - np.sinh(np.sqrt(params.zeta)) / np.sqrt(params.zeta)
                ),
            }
        )

    @property
    def i_c_tr(self):
        return self.to_arr({"A_1": 1}) - self.params.eta * self.i_c_ci

    @property
    def i_c(self):
        return np.array([self.i_c_tr, self.i_c_ci])


class FourierConditions(ABC):
    """Boundary conditions for the first order solution in Fourier space."""

    def __init__(self, variables: FourierVariables):
        """Initialize the boundary conditions.

        :param variables: :class:`~.fourier.FourierVariables` object containing
            the Fourier variables.
        """
        self.variables = variables

        self.omega = variables.omega
        self.params = variables.params
        self.leading = variables.leading

        self.to_arr = variables.to_arr

    @abstractmethod
    def form_equations(self, f):
        """Form the system of equations for the forward problem."""
        pass

    @abstractmethod
    def form_inverse_equations(self, S):
        """Form the system of equations for the inverse problem."""
        pass


class NonZeroFourierConditions(FourierConditions):
    """Boundary conditions for non-zero frequencies in Fourier space."""

    def __init__(self, variables: NonZeroFourierVariables):
        """Initialize the non-zero boundary conditions.

        :param variables: :class:`~.fourier.NonZeroFourierVariables` object containing
            the Fourier variables.
        """
        super().__init__(variables)

    @property
    def no_slip(self):
        """No slip condition."""
        vars = self.variables
        return np.array([vars.d_psi(0), self.omega * vars.psi(0)])

    @property
    def kinematic(self):
        """Kinematic condition."""
        return np.array([self.variables.psi(1)])

    @property
    def normal_stress(self):
        """Normal stress balance."""
        vars = self.variables
        lhs = vars.d3_psi(1) - 3 * self.omega**2 * vars.d_psi(1)
        rhs = (
            1.0j
            * self.omega**3
            * (1 + self.params.Man * (1 - self.leading.gamma_tot))
            * self.to_arr(
                {
                    "S": 1,
                }
            )
        )

        return np.array([lhs - rhs])

    @property
    def no_flux(self):
        """No flux condition."""
        return self.variables.d_c(0)

    @property
    def kin_fluxes(self):
        """Kinetic fluxes."""
        J_vec = np.array([self.to_arr({"J_tr": 1}), self.to_arr({"J_ci": 1})])
        S_vec = np.tile(self.to_arr({"S": 1}), (2, 1, 1))
        gamma_vec = np.array(
            [self.to_arr({"gamma_tr": 1}), self.to_arr({"gamma_ci": 1})]
        )

        eq_1 = self.variables.c(1) + np.einsum(
            "i...,i->i...", S_vec, self.leading.d_c(1)
        )
        eq_2 = (1 - self.leading.gamma_tot) * np.einsum(
            "ij,j...->i...", self.params.K, eq_1
        )

        eq_3 = np.tile(gamma_vec[0] + gamma_vec[1], (2, 1, 1))
        eq_4 = np.einsum("i,i...->i...", self.leading.c(1), eq_3)
        eq_5 = np.einsum("ij,j...->i...", self.params.K, eq_4)

        rhs = np.einsum("ij,j...->i...", self.params.B, eq_2 - eq_5 - gamma_vec)

        return J_vec - rhs

    @property
    def surf_excess(self):
        """Surface excess concentration equations."""
        gamma_vec = np.array(
            [self.to_arr({"gamma_tr": 1}), self.to_arr({"gamma_ci": 1})]
        )
        J_vec = np.array([self.to_arr({"J_tr": 1}), self.to_arr({"J_ci": 1})])
        f_vec = np.tile(self.to_arr({"f": 1}), (2, 1, 1))

        d_psi_vec = np.tile(self.variables.d_psi(1), (2, 1, 1))
        eq_1 = (
            1.0j
            * self.omega
            * np.einsum("i,i...->i...", self.params.P_s @ self.leading.gamma, d_psi_vec)
        )

        eq_2 = self.omega**2 * gamma_vec

        eq_3 = -np.einsum("ij,j...->i...", self.params.P_s, J_vec)

        eq_4 = gamma_vec + np.einsum("i...,i->i...", f_vec, self.leading.gamma)
        eq_5 = np.einsum("ij,j...->i...", self.params.A_s, eq_4)

        return eq_1 + eq_2 + eq_3 + eq_5

    @property
    def tangential_stress(self):
        """Tangential stress balance."""
        vars = self.variables

        lhs = vars.d2_psi(1) + self.omega**2 * vars.psi(1)
        rhs = (
            -1.0j
            * self.omega
            * self.params.Man
            / (1 - self.leading.gamma_tot)
            * self.to_arr(
                {
                    "gamma_tr": 1,
                    "gamma_ci": 1,
                }
            )
        )

        return np.array([lhs - rhs])

    @property
    def mass_balance(self):
        """Mass balance."""
        J_vec = np.array([self.to_arr({"J_tr": 1}), self.to_arr({"J_ci": 1})])
        S_vec = np.tile(self.to_arr({"S": 1}), (2, 1, 1))

        eq_1 = self.variables.d_c(1) + np.einsum(
            "i...,i->i...", S_vec, self.leading.d2_c(1)
        )
        lhs = self.params.k_tr * self.params.chi_tr * eq_1

        rhs = -np.einsum("ij,j...->i...", self.params.P, J_vec)

        return lhs - rhs

    def light_intensity(self, f):
        """Specify the light intensity."""
        f_vec = self.to_arr({"f": 1})
        const_vec = self.to_arr({"const": 1})

        return np.array([f_vec - f * const_vec])

    def surface(self, S):
        """Specify the surface shape."""
        S_vec = self.to_arr({"S": 1})
        const_vec = self.to_arr({"const": 1})

        return np.array([S_vec - S * const_vec])

    def form_equations(self, f):
        """Form the system of equations for the forward problem."""
        return np.concatenate(
            [
                self.no_slip,
                self.kinematic,
                self.normal_stress,
                self.no_flux,
                self.kin_fluxes,
                self.surf_excess,
                self.tangential_stress,
                self.mass_balance,
                self.light_intensity(f),
            ],
            axis=0,
        )

    def form_inverse_equations(self, S):
        """Form the system of equations for the inverse problem."""
        return np.concatenate(
            [
                self.no_slip,
                self.kinematic,
                self.normal_stress,
                self.no_flux,
                self.kin_fluxes,
                self.surf_excess,
                self.tangential_stress,
                self.mass_balance,
                self.surface(S),
            ],
            axis=0,
        )


class ZeroFourierConditions(FourierConditions):
    """Boundary conditions for the zero frequency in Fourier space."""

    def __init__(self, variables: ZeroFourierVariables):
        """Initialize the zero boundary conditions.

        :param variables: :class:`~.fourier.ZeroFourierVariables` object containing
            the Fourier variables.
        """
        super().__init__(variables)

    @property
    def no_slip(self):
        """No slip condition."""
        return np.array([self.variables.d_psi(0), self.variables.psi(0)])

    @property
    def kin_fluxes(self):
        """Kinetic fluxes."""
        J_vec = np.array([self.to_arr({"J_tr": 1}), self.to_arr({"J_ci": 1})])
        gamma_vec = np.array(
            [self.to_arr({"gamma_tr": 1}), self.to_arr({"gamma_ci": 1})]
        )

        eq_1 = (1 - self.leading.gamma_tot) * np.einsum(
            "ij,j...->i...",
            self.params.B @ self.params.K,
            self.variables.c(1),
        )
        eq_2 = -np.einsum("ij,j...->i...", self.leading.M - self.params.D, gamma_vec)
        rhs = eq_1 + eq_2

        return J_vec - rhs

    @property
    def surf_excess(self):
        """Surface excess concentration equations."""
        J_vec = np.array([self.to_arr({"J_tr": 1}), self.to_arr({"J_ci": 1})])
        gamma_vec = np.array(
            [self.to_arr({"gamma_tr": 1}), self.to_arr({"gamma_ci": 1})]
        )
        f_vec = np.tile(self.to_arr({"f": 1}), (2, 1, 1))

        eq_1 = np.einsum("ij,j...->i...", self.params.D, gamma_vec)
        eq_2 = np.einsum("i...,i->i...", f_vec, self.params.D @ self.leading.gamma)
        rhs = eq_1 + eq_2

        return J_vec - rhs

    @property
    def tangential_stress(self):
        """Tangential stress balance."""
        return np.array([self.variables.d2_psi(1)])

    @property
    def mass_balance(self):
        J_ci_vec = self.to_arr({"J_ci": 1})

        lhs = (
            self.params.k_ci * self.params.chi_ci / self.params.Pen_ci
        ) * self.variables.d_c_ci(1)

        rhs = -J_ci_vec

        return np.array([lhs - rhs])

    @property
    def integral(self):
        """Surfactant conservation condition."""
        gamma_vec = self.to_arr(
            {
                "gamma_tr": 1,
                "gamma_ci": 1,
            }
        )

        return np.array(
            [
                gamma_vec / (self.params.k_tr * self.params.chi_tr)
                + self.variables.i_c_tr
                + self.variables.i_c_ci
            ]
        )

    def light_intensity(self, f):
        """Specify the light intensity."""
        f_vec = self.to_arr({"f": 1})
        const_vec = self.to_arr({"const": 1})

        return np.array([f_vec - f * const_vec])

    def form_equations(self, f):
        return np.concatenate(
            [
                self.no_slip,
                self.kin_fluxes,
                self.surf_excess,
                self.tangential_stress,
                self.mass_balance,
                self.integral,
                self.light_intensity(f),
            ],
            axis=0,
        )

    def form_inverse_equations(self):
        return np.concatenate(
            [
                self.no_slip,
                self.kin_fluxes,
                self.surf_excess,
                self.tangential_stress,
                self.mass_balance,
                self.integral,
                self.light_intensity(1.0),
            ],
            axis=0,
        )[:, :, 0]


class FirstOrder(object):
    """First order solution to the photosurfactant model."""

    def __init__(
        self,
        omega,
        func_coeffs,
        params: Parameters,
        leading: LeadingOrder,
        direction="forward",
    ):
        """Initalise solution to the first order model.

        :param omega: Array of angular frequencies.
        :param func_coeffs: Fourier coefficients of the interpolated function.
        :param params: :class:`~.parameters.Parameters` object containing the
            model parameters.
        :param leading: :class:`~.leadingrder.LeadingOrder` object containing
            the leading order solution.
        """
        self.omega = omega
        self.func_coeffs = func_coeffs
        self.params = params
        self.leading = leading
        self.direction = direction

        self._initialize(direction)

    def _initialize(self, direction):
        self.zero_vars = ZeroFourierVariables(self.omega, self.params, self.leading)
        self.vars = NonZeroFourierVariables(self.omega, self.params, self.leading)

        self.zero_conditions = ZeroFourierConditions(self.zero_vars)
        self.conditions = NonZeroFourierConditions(self.vars)

        if direction == "forward":
            zero_sys, sys = self._initialize_forward()
        elif direction == "inverse":
            zero_sys, sys = self._initialize_inverse()

        zero_forcing = zero_sys[:, -1]
        forcing = sys[:, -1, :]

        self.zero_sols = np.array([np.linalg.solve(zero_sys[:, :-1], -zero_forcing)])
        self.sols = np.array(
            [
                np.linalg.solve(sys[:, :-1, i], -forcing[:, i])
                for i in range(len(self.omega))
            ]
        )

        self._invert_variables()

    def _initialize_forward(self):
        zero_sys = self.zero_conditions.form_equations(self.func_coeffs[0])[:, :, 0]
        sys = self.conditions.form_equations(self.func_coeffs[1:])

        return zero_sys, sys

    def _initialize_inverse(self):
        zero_sys = self.zero_conditions.form_inverse_equations()[:, :, 0]
        sys = self.conditions.form_inverse_equations(self.func_coeffs[1:])

        return zero_sys, sys

    def _invert(self, func):
        def _(x, *args):
            coeffs = func(*args)
            return coeffs[0] + np.sum(
                coeffs[1:, np.newaxis]
                * np.exp(
                    1.0j
                    * self.omega[:, np.newaxis]
                    * (x + self.params.L)[np.newaxis, :]
                ),
                axis=0,
            )

        return _

    def _invert_variables(self):
        self.psi = self._invert(self.psi_f)
        self.u = self._invert(self.u_f)
        self.v = self._invert(self.v_f)
        self.c_tr = self._invert(self.c_tr_f)
        self.c_ci = self._invert(self.c_ci_f)
        self.gamma_tr = self._invert(self.gamma_tr_f)
        self.gamma_ci = self._invert(self.gamma_ci_f)
        self.J_tr = self._invert(self.J_tr_f)
        self.J_ci = self._invert(self.J_ci_f)
        self.S_inv = self._invert(self.S_f)
        self.f_inv = self._invert(self.f_f)

    def psi_f(self, y):
        """Streamfunction in Fourier space."""
        zero_val = self.zero_vars.psi(y)
        val = self.vars.psi(y)
        return np.concatenate(
            [
                np.einsum("ij,ji->i", self.zero_sols, zero_val[:-1]) + zero_val[-1],
                np.einsum("ij,ji->i", self.sols, val[:-1]) + val[-1],
            ]
        )

    def u_f(self, y):
        """Horizontal velocity in Fourier space."""
        zero_val = self.zero_vars.d_psi(y)
        val = self.vars.d_psi(y)
        return np.concatenate(
            [
                np.einsum("ij,ji->i", self.zero_sols, zero_val[:-1]) + zero_val[-1, 0],
                np.einsum("ij,ji->i", self.sols, val[:-1] + val[-1]),
            ]
        )

    def v_f(self, y):
        """Vertical velocity in Fourier space."""
        val = self.vars.psi(y)
        return np.concatenate(
            [
                [0],
                -1.0j
                * self.omega
                * (np.einsum("ij,ji->i", self.sols, val[:-1]) + val[-1]),
            ]
        )

    def c_f(self, y):
        """Concentration in Fourier space."""
        zero_val = self.zero_vars.c(y)
        val = self.vars.c(y)
        return np.concatenate(
            [
                np.einsum("ij,kji->ki", self.zero_sols, zero_val[:, :-1])
                + zero_val[:, -1],
                np.einsum("ij,kji->ki", self.sols, val[:, :-1]) + val[:, -1],
            ],
            axis=1,
        )

    def c_tr_f(self, y):
        return self.c_f(y)[0]

    def c_ci_f(self, y):
        return self.c_f(y)[1]

    def gamma_tr_f(self):
        return np.concatenate(
            [
                self.zero_sols[:, self.zero_vars["gamma_tr"]],
                self.sols[:, self.vars["gamma_tr"]],
            ]
        )

    def gamma_ci_f(self):
        return np.concatenate(
            [
                self.zero_sols[:, self.zero_vars["gamma_ci"]],
                self.sols[:, self.vars["gamma_ci"]],
            ]
        )

    def J_tr_f(self):
        return np.concatenate(
            [
                self.zero_sols[:, self.zero_vars["J_tr"]],
                self.sols[:, self.vars["J_tr"]],
            ]
        )

    def J_ci_f(self):
        return np.concatenate(
            [
                self.zero_sols[:, self.zero_vars["J_ci"]],
                self.sols[:, self.vars["J_ci"]],
            ]
        )

    def S_f(self):
        return np.concatenate(
            [
                [0],
                self.sols[:, self.vars["S"]],
            ]
        )

    def f_f(self):
        return np.concatenate(
            [
                self.zero_sols[:, self.zero_vars["f"]],
                self.sols[:, self.vars["f"]],
            ]
        )
