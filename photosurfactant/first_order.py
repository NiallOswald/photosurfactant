"""First order solution to the photosurfactant model."""

from .parameters import Parameters
from .leading_order import LeadingOrder
from abc import ABC, abstractmethod
import numpy as np
from math import factorial


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

    def _psi(self, y, k):
        return self.to_arr(
            {
                "A": self.omega ** (k - 1)
                * (k + self.omega * y)
                * np.exp(self.omega * y),
                "B": self.omega**k * np.exp(self.omega * y),
                "C": (-self.omega) ** (k - 1)
                * (k - self.omega * y)
                * np.exp(-self.omega * y),
                "D": (-self.omega) ** k * np.exp(-self.omega * y),
            }
        )

    def psi(self, y):
        return self._psi(y, 0)

    def d_psi(self, y):
        return self._psi(y, 1)

    def d2_psi(self, y):
        return self._psi(y, 2)

    def d3_psi(self, y):
        return self._psi(y, 3)

    def d4_psi(self, y):
        return self._psi(y, 4)

    def pressure(self, y):
        return (self.d3_psi(y) - self.omega**2 * self.d_psi(y)) / (1.0j * self.omega)

    def d_pressure(self, y):
        return (self.d4_psi(y) - self.omega**2 * self.d2_psi(y)) / (1.0j * self.omega)

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

    def _p_0(self, y, k):
        eq_1 = self.omega**k * self.to_arr(
            {
                "E": (np.cosh if k % 2 else np.sinh)(self.omega * y),
                "F": (np.sinh if k % 2 else np.cosh)(self.omega * y),
            }
        )
        eq_2 = np.sqrt(self.params.zeta + self.omega**2) ** k * self.to_arr(
            {
                "G": (np.cosh if k % 2 else np.sinh)(
                    np.sqrt(self.params.zeta + self.omega**2) * y
                ),
                "H": (np.sinh if k % 2 else np.cosh)(
                    np.sqrt(self.params.zeta + self.omega**2) * y
                ),
            }
        )

        return np.array([eq_1, eq_2])

    def _p_1(self, y, k):
        if k == 0:
            vec = (self.params.Lambda @ np.linalg.inv(self.params.V)) @ np.array(
                [self.leading.A_0, 0]
            )
            eq_1 = np.einsum("ij...,j->i...", self.Lambda_inv, vec)
            f_vec = self.to_arr({"f": 1})

            return -np.einsum("i...,j...->ij...", eq_1[:, -1], f_vec)

        else:
            return np.array([self.to_arr(dict()), self.to_arr(dict())])

    def _p_2_1(self, y, k):
        return self.to_arr(
            {
                "const": (
                    (
                        -self.leading.A_0
                        / ((self.params.alpha + self.params.eta) * self.omega**2)
                        if k == 0
                        else 0
                    )
                    - self.leading.B_0
                    * np.sqrt(self.params.zeta) ** k
                    / (self.omega**2 - self.params.zeta)
                    * (np.sinh if k % 2 else np.cosh)(y * np.sqrt(self.params.zeta))
                ),
            }
        )

    def _p_2_2(self, y, k):
        return self.to_arr(
            {
                "const": (
                    (
                        -self.leading.A_0
                        / (
                            (self.params.alpha + self.params.eta)
                            * (self.omega**2 + self.params.zeta)
                        )
                        if k == 0
                        else 0
                    )
                    - self.leading.B_0
                    * np.sqrt(self.params.zeta) ** k
                    / self.omega**2
                    * (np.sinh if k % 2 else np.cosh)(y * np.sqrt(self.params.zeta))
                ),
            }
        )

    def _p_2(self, y, k):
        params = self.params

        zero = self.to_arr(dict())
        matr = np.array([[self._p_2_1(y, k), zero], [zero, self._p_2_2(y, k)]])
        vec = params.Lambda @ np.linalg.inv(params.V) @ np.array([-params.eta, 1.0])
        vec_2 = np.einsum("ij...,j->i...", matr, vec)

        f_vec = self.to_arr({"f": 1})

        return np.einsum("i...,j...->ij...", vec_2[:, -1], f_vec)

    def _p_3_const(self, s_1, s_2):
        return s_1 * (self.omega + s_2 * np.sqrt(self.params.zeta))

    def _p_3_gfunc(self, y, a, b, s_1, s_2, k=0):
        return (
            k * self._p_3_const(s_1, s_2) ** (k - 1) * a
            + self._p_3_const(s_1, s_2) ** k * (a * y + b)
        ) * np.exp(self._p_3_const(s_1, s_2) * y)

    def _p_3_1(self, y, k):
        return (
            self._p_3_gfunc(y, self.a_c_1, self.b_c_1, 1, 1, k)
            + self._p_3_gfunc(y, self.c_c_1, self.d_c_1, 1, -1, k)
            + self._p_3_gfunc(y, self.e_c_1, self.f_c_1, -1, -1, k)
            + self._p_3_gfunc(y, self.g_c_1, self.h_c_1, -1, 1, k)
        )

    def _p_3_2(self, y, k):
        return (
            self._p_3_gfunc(y, self.a_c_2, self.b_c_2, 1, 1, k)
            + self._p_3_gfunc(y, self.c_c_2, self.d_c_2, 1, -1, k)
            + self._p_3_gfunc(y, self.e_c_2, self.f_c_2, -1, -1, k)
            + self._p_3_gfunc(y, self.g_c_2, self.h_c_2, -1, 1, k)
        )

    def _p_3(self, y, k):
        params = self.params

        zero = self.to_arr(dict())
        matr = np.array([[self._p_3_1(y, k), zero], [zero, self._p_3_2(y, k)]])
        vec = np.linalg.inv(params.V) @ params.P @ np.array([-params.eta, 1])

        return (
            -(1.0j * self.omega * self.leading.B_0 * np.sqrt(params.zeta))
            / 2
            * np.einsum("ij...,j->i...", matr, vec)
        )

    def _p(self, y, k):
        return self._p_0(y, k) + self._p_1(y, k) + self._p_2(y, k) + self._p_3(y, k)

    def p(self, y):
        return self._p(y, 0)

    def d_p(self, y):
        return self._p(y, 1)

    def d2_p(self, y):
        return self._p(y, 2)

    def _c(self, y, k):
        return np.einsum("ij,j...->i...", self.params.V, self._p(y, k))

    def c(self, y):
        return self._c(y, 0)

    def d_c(self, y):
        return self._c(y, 1)

    def d2_c(self, y):
        return self._c(y, 2)

    def i_p_0(self, y):
        eq_1 = (
            self.to_arr(
                {
                    "E": np.cosh(self.omega * y) - 1,
                    "F": np.sinh(self.omega * y),
                }
            )
            / self.omega
        )

        eq_2 = self.to_arr(
            {
                "G": np.cosh(np.sqrt(self.params.zeta + self.omega**2) * y) - 1,
                "H": np.sinh(np.sqrt(self.params.zeta + self.omega**2) * y),
            }
        ) / np.sqrt(self.params.zeta + self.omega**2)

        return np.array([eq_1, eq_2])

    def i_p_1(self, y):
        return self._p_1(y, 0) * y

    def i_p_2_1(self, y):
        return self.to_arr(
            {
                "const": -y
                * self.leading.A_0
                / ((self.params.alpha + self.params.eta) * self.omega**2)
                - self.leading.B_0
                / (np.sqrt(self.params.zeta) * (self.omega**2 - self.params.zeta))
                * np.sinh(y * np.sqrt(self.params.zeta))
            }
        )

    def i_p_2_2(self, y):
        return self.to_arr(
            {
                "const": -y
                * self.leading.A_0
                / (
                    (self.params.alpha + self.params.eta)
                    * (self.omega**2 + self.params.zeta)
                )
                - self.leading.B_0
                / (np.sqrt(self.params.zeta) * self.omega**2)
                * np.sinh(y * np.sqrt(self.params.zeta))
            }
        )

    def i_p_2(self, y):
        params = self.params

        zero = self.to_arr(dict())
        matr = np.array([[self.i_p_2_1(y), zero], [zero, self.i_p_2_2(y)]])
        vec = params.Lambda @ np.linalg.inv(params.V) @ np.array([-params.eta, 1.0])
        vec_2 = np.einsum("ij...,j->i...", matr, vec)

        f_vec = self.to_arr({"f": 1})

        return np.einsum("i...,j...->ij...", vec_2[:, -1], f_vec)

    def _i_p_3_gfunc(self, y, a, b, s_1, s_2):
        return (
            (a * y + b) / self._p_3_const(s_1, s_2) - a / self._p_3_const(s_1, s_2**2)
        ) * np.exp(self._p_3_const(s_1, s_2) * y)

    def i_p_3_1(self, y):
        return (
            self._i_p_3_gfunc(y, self.a_c_1, self.b_c_1, 1, 1)
            + self._i_p_3_gfunc(y, self.c_c_1, self.d_c_1, 1, -1)
            + self._i_p_3_gfunc(y, self.e_c_1, self.f_c_1, -1, -1)
            + self._i_p_3_gfunc(y, self.g_c_1, self.h_c_1, -1, 1)
        )

    def i_p_3_2(self, y):
        return (
            self._i_p_3_gfunc(y, self.a_c_2, self.b_c_2, 1, 1)
            + self._i_p_3_gfunc(y, self.c_c_2, self.d_c_2, 1, -1)
            + self._i_p_3_gfunc(y, self.e_c_2, self.f_c_2, -1, -1)
            + self._i_p_3_gfunc(y, self.g_c_2, self.h_c_2, -1, 1)
        )

    def i_p_3(self, y):
        params = self.params

        zero = self.to_arr(dict())
        matr = np.array([[self.i_p_3_1(y), zero], [zero, self.i_p_3_2(y)]])
        vec = np.linalg.inv(params.V) @ params.P @ np.array([-params.eta, 1])

        return (
            -(1.0j * self.omega * self.leading.B_0 * np.sqrt(params.zeta))
            / 2
            * np.einsum("ij...,j->i...", matr, vec)
        )

    def i_p(self, y):
        return self.i_p_0(y) + self.i_p_1(y) + self.i_p_2(y) + self.i_p_3(y)

    def i_c(self, y):
        return np.einsum("ij,j...->i...", self.params.V, self.i_p(y))


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

    def _psi_poly(self, y, r, k):
        return factorial(r) / factorial(r - k) * y ** (r - k) if r >= k else 0

    def _psi(self, y, k):
        return self.to_arr(
            {
                "A": self._psi_poly(y, 2, k),
                "B": self._psi_poly(y, 1, k),
                "C": self._psi_poly(y, 0, k),
            }
        )

    def psi(self, y):
        return self._psi(y, 0)

    def d_psi(self, y):
        return self._psi(y, 1)

    def d2_psi(self, y):
        return self._psi(y, 2)

    def d3_psi(self, y):
        return self._psi(y, 3)

    def d4_psi(self, y):
        return self._psi(y, 4)

    def pressure(self, y):
        return self.to_arr(dict())

    def d_pressure(self, y):
        return self.to_arr(dict())

    def _c_ci(self, y, k):
        params = self.params
        return self.to_arr(
            {
                "A_1": 1 / (params.alpha + params.eta) if k == 0 else 0,
                "B_1": np.sqrt(params.zeta) ** k
                * (np.sinh if k % 2 else np.cosh)(y * np.sqrt(params.zeta)),
                "f": (self.leading.B_0 * np.sqrt(params.zeta) / 2)
                * (
                    k
                    * np.sqrt(params.zeta) ** (k - 1)
                    * (np.sinh if k % 2 else np.cosh)(y * np.sqrt(params.zeta))
                    + np.sqrt(params.zeta) ** k
                    * y
                    * (np.cosh if k % 2 else np.sinh)(y * np.sqrt(params.zeta))
                ),
            }
        )

    def _c_tr(self, y, k):
        return self.to_arr({"A_1": int(k == 0)}) - self.params.eta * self._c_ci(y, k)

    def _c(self, y, k):
        return np.array([self._c_tr(y, k), self._c_ci(y, k)])

    def c_ci(self, y):
        return self._c_ci(y, 0)

    def c_tr(self, y):
        return self._c_tr(y, 0)

    def c(self, y):
        return self._c(y, 0)

    def d_c_ci(self, y):
        return self._c_ci(y, 1)

    def d_c_tr(self, y):
        return self._c_tr(y, 1)

    def d_c(self, y):
        return self._c(y, 1)

    def d2_c_ci(self, y):
        return self._c_ci(y, 2)

    def d2_c_tr(self, y):
        return self._c_tr(y, 2)

    def d2_c(self, y):
        return self._c(y, 2)

    def i_c_ci(self, y):
        params = self.params
        return self.to_arr(
            {
                "A_1": y / (params.alpha + params.eta),
                "B_1": np.sinh(y * np.sqrt(params.zeta)) / np.sqrt(params.zeta),
                "f": (self.leading.B_0 / 2)
                * (
                    y * np.cosh(y * np.sqrt(params.zeta))
                    - np.sinh(y * np.sqrt(params.zeta)) / np.sqrt(params.zeta)
                ),
            }
        )

    def i_c_tr(self, y):
        return self.to_arr({"A_1": y}) - self.params.eta * self.i_c_ci(y)

    def i_c(self, y):
        return np.array([self.i_c_tr(y), self.i_c_ci(y)])


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
        lhs = -vars.pressure(1) - 2.0j * self.omega * vars.d_psi(1)
        rhs = self.leading.tension * self.to_arr(
            {
                "S": -self.omega**2,
            }
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
                + self.variables.i_c_tr(1)
                + self.variables.i_c_ci(1)
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
        )[:, :, 0]

    def form_inverse_equations(self):
        return np.concatenate(
            [
                self.no_slip,
                self.kin_fluxes,
                self.surf_excess,
                self.tangential_stress,
                self.mass_balance,
                self.integral,
                self.light_intensity(0.0),
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

        self.omega_zero = np.concatenate([[0], omega])

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
        zero_sys = self.zero_conditions.form_equations(self.func_coeffs[0])
        sys = self.conditions.form_equations(self.func_coeffs[1:])

        return zero_sys, sys

    def _initialize_inverse(self):
        zero_sys = self.zero_conditions.form_inverse_equations()
        sys = self.conditions.form_inverse_equations(self.func_coeffs[1:])

        return zero_sys, sys

    def _invert(self, func):
        def _(x, *args, **kwargs):
            coeffs = func(*args, **kwargs)
            return coeffs[0] + np.sum(
                coeffs[1:, np.newaxis]
                * np.exp(1.0j * self.omega[:, np.newaxis] * x[np.newaxis, :]),
                axis=0,
            )

        return _

    def _invert_variables(self):
        self.psi = self._invert(self.psi_f)
        self.d_psi = self._invert(self.d_psi_f)
        self.d2_psi = self._invert(self.d2_psi_f)
        self.d3_psi = self._invert(self.d3_psi_f)
        self.d4_psi = self._invert(self.d4_psi_f)

        self.u = self._invert(self.u_f)
        self.v = self._invert(self.v_f)

        self.d_u = self._invert(self.d_u_f)
        self.d_v = self._invert(self.d_v_f)

        self.d2_u = self._invert(self.d2_u_f)
        self.d2_v = self._invert(self.d2_v_f)

        self.pressure = self._invert(self.pressure_f)
        self.d_pressure = self._invert(self.d_pressure_f)

        self.c_tr = self._invert(self.c_tr_f)
        self.c_ci = self._invert(self.c_ci_f)

        self.d_c_tr = self._invert(self.d_c_tr_f)
        self.d_c_ci = self._invert(self.d_c_ci_f)

        self.d2_c_tr = self._invert(self.d2_c_tr_f)
        self.d2_c_ci = self._invert(self.d2_c_ci_f)

        self.i_c_tr = self._invert(self.i_c_tr_f)
        self.i_c_ci = self._invert(self.i_c_ci_f)

        self.gamma_tr = self._invert(self.gamma_tr_f)
        self.gamma_ci = self._invert(self.gamma_ci_f)

        self.J_tr = self._invert(self.J_tr_f)
        self.J_ci = self._invert(self.J_ci_f)

        self.S_inv = self._invert(self.S_f)

        self.f_inv = self._invert(self.f_f)

        self.gamma_tot = lambda x, x_order=0: self.gamma_tr(x, x_order) + self.gamma_ci(
            x, x_order
        )
        self.tension = (
            lambda x, x_order=0: -self.params.Man
            * self.gamma_tot(x, x_order)
            / (1 - self.leading.gamma_tot)
        )

    def _gfunc_f(self, zero_val, val, x_order=0):
        return (
            np.concatenate(
                [
                    np.einsum("ij,...ji->...i", self.zero_sols, zero_val[..., :-1, :])
                    + zero_val[..., -1, :],
                    np.einsum("ij,...ji->...i", self.sols, val[..., :-1, :])
                    + val[..., -1, :],
                ],
                axis=-1,
            )
            * (1.0j * self.omega_zero) ** x_order
        )

    def _const_f(self, key, x_order=0):
        return (
            np.concatenate(
                [
                    self.zero_sols[:, self.zero_vars[key]],
                    self.sols[:, self.vars[key]],
                ]
            )
            * (1.0j * self.omega_zero) ** x_order
        )

    def psi_f(self, y, x_order=0):
        """Streamfunction in Fourier space."""
        zero_val = self.zero_vars.psi(y)
        val = self.vars.psi(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d_psi_f(self, y, x_order=0):
        """Derivative of streamfunction in Fourier space."""
        zero_val = self.zero_vars.d_psi(y)
        val = self.vars.d_psi(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d2_psi_f(self, y, x_order=0):
        """Second derivative of streamfunction in Fourier space."""
        zero_val = self.zero_vars.d2_psi(y)
        val = self.vars.d2_psi(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d3_psi_f(self, y, x_order=0):
        """Third derivative of streamfunction in Fourier space."""
        zero_val = self.zero_vars.d3_psi(y)
        val = self.vars.d3_psi(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d4_psi_f(self, y, x_order=0):
        """Fourth derivative of streamfunction in Fourier space."""
        zero_val = self.zero_vars.d4_psi(y)
        val = self.vars.d4_psi(y)
        return self._gfunc_f(zero_val, val, x_order)

    def u_f(self, y, x_order=0):
        """Horizontal velocity in Fourier space."""
        return self.d_psi_f(y, x_order)

    def d_u_f(self, y, x_order=0):
        """Derivative of horizontal velocity in Fourier space."""
        return self.d2_psi_f(y, x_order)

    def d2_u_f(self, y, x_order=0):
        """Second derivative of horizontal velocity in Fourier space."""
        return self.d3_psi_f(y, x_order)

    def v_f(self, y, x_order=0):
        """Vertical velocity in Fourier space."""
        return -self.psi_f(y, x_order + 1)

    def d_v_f(self, y, x_order=0):
        """Derivative of vertical velocity in Fourier space."""
        return -self.d_psi_f(y, x_order + 1)

    def d2_v_f(self, y, x_order=0):
        """Second derivative of vertical velocity in Fourier space."""
        return -self.d2_psi_f(y, x_order + 1)

    def pressure_f(self, y, x_order=0):
        """Pressure in Fourier space."""
        zero_val = self.zero_vars.pressure(y)
        val = self.vars.pressure(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d_pressure_f(self, y, x_order=0):
        """Derivative of pressure in Fourier space."""
        zero_val = self.zero_vars.d_pressure(y)
        val = self.vars.d_pressure(y)
        return self._gfunc_f(zero_val, val, x_order)

    def c_f(self, y, x_order=0):
        """Concentration in Fourier space."""
        zero_val = self.zero_vars.c(y)
        val = self.vars.c(y)
        return self._gfunc_f(zero_val, val, x_order)

    def c_tr_f(self, y, x_order=0):
        return self.c_f(y, x_order)[0]

    def c_ci_f(self, y, x_order=0):
        return self.c_f(y, x_order)[1]

    def d_c_f(self, y, x_order=0):
        """Derivative of concentration in Fourier space."""
        zero_val = self.zero_vars.d_c(y)
        val = self.vars.d_c(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d_c_tr_f(self, y):
        return self.d_c_f(y)[0]

    def d_c_ci_f(self, y):
        return self.d_c_f(y)[1]

    def d2_c_f(self, y, x_order=0):
        """Second derivative of concentration in Fourier space."""
        zero_val = self.zero_vars.d2_c(y)
        val = self.vars.d2_c(y)
        return self._gfunc_f(zero_val, val, x_order)

    def d2_c_tr_f(self, y):
        return self.d2_c_f(y)[0]

    def d2_c_ci_f(self, y):
        return self.d2_c_f(y)[1]

    def i_c_f(self, y):
        """Integral of concentration in Fourier space."""
        zero_val = self.zero_vars.i_c(y)
        val = self.vars.i_c(y)
        return self._gfunc_f(zero_val, val)

    def i_c_tr_f(self, y):
        return self.i_c_f(y)[0]

    def i_c_ci_f(self, y):
        return self.i_c_f(y)[1]

    def gamma_tr_f(self, x_order=0):
        return self._const_f("gamma_tr", x_order)

    def gamma_ci_f(self, x_order=0):
        return self._const_f("gamma_ci", x_order)

    def J_tr_f(self):
        return self._const_f("J_tr")

    def J_ci_f(self):
        return self._const_f("J_ci")

    def S_f(self, x_order=0):
        return (
            np.concatenate(
                [
                    [0],
                    self.sols[:, self.vars["S"]],
                ]
            )
            * (1.0j * self.omega_zero) ** x_order
        )

    def f_f(self):
        return self._const_f("f")
