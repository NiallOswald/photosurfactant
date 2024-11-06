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
        d = (params.alpha + 1) / (params.alpha + params.eta) + (
            1 / (params.k_tr * params.chi_tr) - 1 / (2 * params.L)
        ) * (a_0 + b_0 / (params.alpha + params.eta))
        e = (1 - params.eta) / np.sqrt(params.zeta) * np.sinh(
            np.sqrt(params.zeta)
        ) + b_0 * (1 / (params.k_tr * params.chi_tr) - 1 / (2 * params.L)) * np.cosh(
            np.sqrt(params.zeta)
        )
        f = -1 / (2 * params.L)

        # Second Quadratic
        p = (
            params.k_ci
            * params.chi_ci
            * b_0
            * (params.alpha + params.eta)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
            * np.cosh(np.sqrt(params.zeta))
        )
        q = (
            params.k_ci
            * params.chi_ci
            * (a_0 + b_0 / (params.alpha + params.eta))
            * (params.alpha + params.eta)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
        )
        r = params.k_ci * params.chi_ci * (params.alpha + params.eta) / np.sqrt(
            params.zeta
        ) * np.sinh(np.sqrt(params.zeta)) + (
            params.alpha * params.k_ci + params.eta * params.k_tr
        ) * np.cosh(
            np.sqrt(params.zeta)
        )
        s = params.alpha / (params.alpha + params.eta) * (params.k_ci - params.k_tr)

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
        return self.B_0 * np.sqrt(params.zeta) * np.cosh(y * np.sqrt(params.zeta))

    def d_c_tr(self, y):  # noqa: D102
        return -self.params.eta * self.d_c_ci(y)

    def d_c(self, y):  # noqa: D102
        return np.array([self.d_c_tr(y), self.d_c_ci(y)])

    def d2_c_ci(self, y):  # noqa: D102
        params = self.params
        return self.B_0 * params.zeta * np.sinh(y * np.sqrt(params.zeta))

    def d2_c_tr(self, y):  # noqa: D102
        return -self.params.eta * self.d2_c_ci(y)

    def d2_c(self, y):  # noqa: D102
        return np.array([self.d2_c_tr(y), self.d2_c_ci(y)])

    @property
    def gamma_ci(self):  # noqa: D102
        return self.gamma[1]

    @property
    def gamma_tr(self):  # noqa: D102
        return self.gamma[0]

    @property
    def gamma(self):  # noqa: D102
        params = self.params
        return (1 / self.Delta) * (
            (
                params.k_tr * self.c_tr(1) / params.Bit_ci
                + params.k_ci * self.c_ci(1) / params.Bit_tr
            )
            * np.array([params.Dam_ci, params.Dam_tr])
            + np.array(
                [
                    params.k_tr * self.c_tr(1),
                    params.k_ci * self.c_ci(1),
                ]
            )
        )

    @property
    def Delta(self):  # noqa: D102, N802
        params = self.params
        return (
            (params.Dam_tr + params.Dam_ci)
            * (
                params.k_tr * self.c_tr(1) / params.Bit_ci
                + params.k_ci * self.c_ci(1) / params.Bit_tr
            )
            + params.k_tr * self.c_tr(1)
            + params.k_ci * self.c_ci(1)
            + 1
        )

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
