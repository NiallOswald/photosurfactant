"""Leading order solution to the photosurfactant model."""

from .parameters import Parameters
import numpy as np


class LeadingOrder(object):
    """Leading order solution to the photosurfactant model."""

    def __init__(self, params: Parameters, root_index: int = -1, method="endpoints"):
        """Initalise solution to the leading order model.

        :param params: :class:`~.parameters.Parameters` object containing the
            model parameters.
        :param root_index: Index of the solution branch to use. If set to -1,
            the branch is selected automatically.
        :param method: Method to use to select the solution branch. If set to
            'endpoints', the solution branch is selected by checking the
            concentrations at the endpoints of the domain.
        """
        self.params = params
        self.A_0_vals, self.B_0_vals = self._initialize()
        self.set_root(root_index, method)

    def _initialize(self):
        """Initialize the leading order solution."""
        params = self.params

        # Define determinant coefficients
        a_0 = params.k_tr * ((params.Dam_tr + params.Dam_ci) / params.Bit_ci + 1)
        b_0 = (
            (params.Dam_tr + params.Dam_ci)
            * (params.k_ci / params.Bit_tr - params.eta * params.k_tr / params.Bit_ci)
            + params.k_ci
            - params.eta * params.k_tr
        )
        c_0 = 1 + params.Dam_tr / params.Bit_tr + params.Dam_ci / params.Bit_ci

        # Quadratic coefficients
        a = (
            (a_0 + b_0 / (params.alpha + params.eta))
            * (params.alpha + 1)
            / (params.alpha + params.eta)
        )
        b = b_0 * (params.alpha + 1) / (params.alpha + params.eta) * np.cosh(
            np.sqrt(params.zeta)
        ) + (a_0 + b_0 / (params.alpha + params.eta)) * (1 - params.eta) / np.sqrt(
            params.zeta
        ) * np.sinh(
            np.sqrt(params.zeta)
        )
        c = (
            b_0
            * (1 - params.eta)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
            * np.cosh(np.sqrt(params.zeta))
        )
        d = c_0 * (params.alpha + 1) / (params.alpha + params.eta) + (
            1 / (params.k_tr * params.chi_tr) - 1 / (2 * params.L)
        ) * (a_0 + b_0 / (params.alpha + params.eta))
        e = c_0 * (1 - params.eta) / np.sqrt(params.zeta) * np.sinh(
            np.sqrt(params.zeta)
        ) + b_0 * (1 / (params.k_tr * params.chi_tr) - 1 / (2 * params.L)) * np.cosh(
            np.sqrt(params.zeta)
        )
        f = c_0 * (1 / (params.k_tr * params.chi_tr) - 1 / (2 * params.L)) - (
            1 + params.kappa * (1 + params.alpha * params.beta)
        ) / (params.k_tr * params.chi_tr)

        # Second Quadratic
        p_0 = -params.alpha / (params.alpha + params.eta) * (params.k_tr - params.k_ci)
        q_0 = (
            params.k_tr
            * params.chi_tr
            * (params.alpha + params.eta)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
        )
        r_0 = (params.eta * params.k_tr + params.alpha * params.k_ci) * np.cosh(
            np.sqrt(params.zeta)
        )

        p = b_0 * q_0 * np.cosh(np.sqrt(params.zeta))
        q = (a_0 + b_0 / (params.alpha + params.eta)) * q_0
        r = c_0 * q_0 + r_0
        s = p_0

        # Solve for B_0
        poly = np.poly1d(
            [
                (a * p**2 - b * p * q + c * q**2),
                (
                    2 * a * p * r
                    - b * p * s
                    - b * q * r
                    + 2 * c * q * s
                    - d * p * q
                    + e * q**2
                ),
                (
                    a * r**2
                    - b * r * s
                    + c * s**2
                    - d * p * s
                    - d * q * r
                    + 2 * e * q * s
                    + f * q**2
                ),
                (-d * r * s + e * s**2 + 2 * f * q * s),
                f * s**2,
            ]
        )
        B_0 = poly.roots  # noqa: N806

        # Solve for A_0
        A_0 = -(p * B_0 + r) / (q * B_0 + s) * B_0  # noqa: N806

        return A_0, B_0

    def set_root(self, root_index: int, method: str):
        """Set the solution branch."""
        if root_index == -1:
            self._set_root_auto(method)
        else:
            self.A_0 = self.A_0_vals[root_index]
            self.B_0 = self.B_0_vals[root_index]

    def _set_root_auto(self, method: str):
        """Automatically set the solution branch."""
        if method == "endpoints":
            for A_0, B_0 in zip(self.A_0_vals, self.B_0_vals):
                self.A_0, self.B_0 = A_0, B_0

                if self.c_ci(0) < 0 or self.c_ci(1) < 0:
                    continue
                elif self.c_tr(0) < 0 or self.c_tr(1) < 0:
                    continue
                else:
                    break

            else:
                raise ValueError("No valid solution found.")

    def c_ci(self, y):  # noqa: D102
        params = self.params
        return self.A_0 / (params.alpha + params.eta) + self.B_0 * np.cosh(
            y * np.sqrt(params.zeta)
        )

    def c_tr(self, y):  # noqa: D102
        return self.A_0 - self.params.eta * self.c_ci(y)

    def c(self, y):  # noqa: D102
        return np.array([self.c_tr(y), self.c_ci(y)])

    def d_c_ci(self, y):  # noqa: D102
        params = self.params
        return self.B_0 * np.sqrt(params.zeta) * np.sinh(y * np.sqrt(params.zeta))

    def d_c_tr(self, y):  # noqa: D102
        return -self.params.eta * self.d_c_ci(y)

    def d_c(self, y):  # noqa: D102
        return np.array([self.d_c_tr(y), self.d_c_ci(y)])

    def d2_c_ci(self, y):  # noqa: D102
        params = self.params
        return self.B_0 * params.zeta * np.cosh(y * np.sqrt(params.zeta))

    def d2_c_tr(self, y):  # noqa: D102
        return -self.params.eta * self.d2_c_ci(y)

    def d2_c(self, y):  # noqa: D102
        return np.array([self.d2_c_tr(y), self.d2_c_ci(y)])

    def i_c_ci(self, y, y_s=0):
        return (
            self.A_0 / (self.params.alpha + self.params.eta) * y
            + (self.B_0 / np.sqrt(self.params.zeta))
            * np.sinh(y * np.sqrt(self.params.zeta))
        ) - (self.i_c_ci(y_s) if y_s else 0)

    def i_c_tr(self, y, y_s=0):
        return self.A_0 * (y - y_s) - self.params.eta * self.i_c_ci(y, y_s)

    def i_c(self, y, y_s):
        return np.array([self.i_c_tr(y, y_s), self.i_c_ci(y, y_s)])

    @property
    def gamma_ci(self):  # noqa: D102
        return self.gamma[1]

    @property
    def gamma_tr(self):  # noqa: D102
        return self.gamma[0]

    @property
    def gamma_tot(self):
        return self.gamma_tr + self.gamma_ci

    @property
    def gamma(self):  # noqa: D102
        params = self.params
        return np.linalg.solve(self.M, params.B @ params.K @ self.c(1))

    @property
    def J_tr(self):
        return self.params.Bit_tr * (
            self.params.k_tr * self.c_tr(1) * (1 - self.gamma_tr - self.gamma_ci)
            - self.gamma_tr
        )

    @property
    def J_ci(self):
        return self.params.Bit_ci * (
            self.params.k_ci * self.c_ci(1) * (1 - self.gamma_tr - self.gamma_ci)
            - self.gamma_ci
        )

    @property
    def J(self):
        return np.array([self.J_tr, self.J_ci])

    @property
    def M(self):  # noqa: D102, N802
        params = self.params
        return params.D + params.B @ np.array(
            [
                [
                    params.k_tr * self.c_tr(1) + 1,
                    params.k_tr * self.c_tr(1),
                ],
                [
                    params.k_ci * self.c_ci(1),
                    params.k_ci * self.c_ci(1) + 1,
                ],
            ]
        )
