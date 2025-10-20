"""Leading order solution to the photosurfactant model."""

import numpy as np

from .parameters import Parameters
from .utils import Y, hyperder, polyder


class LeadingOrder(object):
    """Leading order solution to the photosurfactant model."""

    def __init__(self, params: Parameters, root_index: int = -1):
        """Initalise solution to the leading order model.

        :param params: :class:`~.parameters.Parameters` object containing the
            model parameters.
        :param root_index: Index of the solution branch to use. If set to -1,
            the branch is selected automatically.
        """
        self.params = params
        self._initialize()

        if root_index == -1:
            self._trim_roots()
            if len(self.roots) > 1:
                raise ValueError(
                    "Multiple valid solution branches detected. You can override this "
                    "message by explicitly setting a solution branch using root_index."
                )

        (self.A, self.B) = self.roots[root_index]

    def _initialize(self):
        """Initialize the leading order solution."""
        params = self.params

        # Define determinant coefficients
        a_0 = (params.Da_tr + params.Da_ci) * (
            params.alpha * params.Bi_tr * params.k_tr + params.Bi_ci * params.k_ci
        ) + params.Bi_tr * params.Bi_ci * (params.alpha * params.k_tr + params.k_ci)
        b_0 = (params.Da_tr + params.Da_ci) * (
            params.eta * params.Bi_tr * params.k_tr - params.Bi_ci * params.k_ci
        ) + params.Bi_tr * params.Bi_ci * (params.eta * params.k_tr - params.k_ci)
        c_0 = (
            params.Da_tr * params.Bi_ci
            + params.Da_ci * params.Bi_tr
            + params.Bi_tr * params.Bi_ci
        )

        # Quadratic coefficients
        a = a_0 * (params.alpha + 1)
        b = a_0 * (params.eta - 1) / np.sqrt(params.zeta) * np.sinh(
            np.sqrt(params.zeta)
        ) + b_0 * (params.alpha + 1) * np.cosh(np.sqrt(params.zeta))
        c = (
            b_0
            * (params.eta - 1)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
            * np.cosh(np.sqrt(params.zeta))
        )
        d = a_0 * (1 / (params.k_tr * params.chi_tr) - 1) + c_0 * (params.alpha + 1)
        e = b_0 * (1 / (params.k_tr * params.chi_tr) - 1) * np.cosh(
            np.sqrt(params.zeta)
        ) + c_0 * (params.eta - 1) / np.sqrt(params.zeta) * np.sinh(
            np.sqrt(params.zeta)
        )
        f = -c_0

        # Second Quadratic
        p = (
            b_0
            * params.k_tr
            * params.chi_tr
            * (params.alpha + params.eta)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
            * np.cosh(np.sqrt(params.zeta))
        )
        q = (
            a_0
            * params.k_tr
            * params.chi_tr
            * (params.alpha + params.eta)
            / np.sqrt(params.zeta)
            * np.sinh(np.sqrt(params.zeta))
        )
        r = params.alpha * params.Bi_tr * params.Bi_ci * (params.k_tr - params.k_ci)
        s = c_0 * params.k_tr * params.chi_tr * (params.alpha + params.eta) / np.sqrt(
            params.zeta
        ) * np.sinh(np.sqrt(params.zeta)) + params.Bi_tr * params.Bi_ci * (
            params.eta * params.k_tr + params.alpha * params.k_ci
        ) * np.cosh(np.sqrt(params.zeta))

        # Solve for B
        poly = np.poly1d(
            [
                (0.0 if params.eta == 1 else (a * p**2 - b * p * q + c * q**2)),
                (
                    2 * a * p * s
                    - b * p * r
                    - b * q * s
                    + 2 * c * q * r
                    - d * p * q
                    + e * q**2
                ),
                (
                    a * s**2
                    - b * r * s
                    + c * r**2
                    - d * p * r
                    - d * q * s
                    + 2 * e * q * r
                    + f * q**2
                ),
                (-d * r * s + e * r**2 + 2 * f * q * r),
                f * r**2,
            ]
        )
        B = poly.roots  # noqa: N806

        # Solve for A
        A = -(p * B + s) / (q * B + r) * B  # noqa: N806

        self.roots = [(a, b) for a, b in zip(A, B)]

    def _trim_roots(self):
        """Trim any invalid roots."""
        verified_roots = []
        for A_0, B_0 in self.roots:
            if np.isclose(A_0.imag, 0):
                self.A, self.B = A_0.real, B_0.real
            else:
                continue

            # Check if any of the concentrations are negative
            # (it is sufficient to check the endpoints)
            if self.c_ci(0) < 0 or self.c_ci(1) < 0:
                continue
            elif self.c_tr(0) < 0 or self.c_tr(1) < 0:
                continue

            verified_roots.append([self.A, self.B])

        self.roots = verified_roots

    def c_tr(self, z, z_order=0):
        """Concentration of trans surfactant at leading order."""
        params = self.params
        return params.alpha * self.A * polyder(Y**0, z_order)(
            z
        ) + params.eta * self.B * np.sqrt(params.zeta) ** z_order * hyperder(z_order)(
            z * np.sqrt(params.zeta)
        )

    def c_ci(self, z, z_order=0):
        """Concentration of cis surfactant at leading order."""
        params = self.params
        return self.A * polyder(Y**0, z_order)(z) - self.B * np.sqrt(
            params.zeta
        ) ** z_order * hyperder(z_order)(z * np.sqrt(params.zeta))

    def c(self, z, z_order=0):
        """Concentration of surfactant at leading order."""
        return np.array([self.c_tr(z, z_order), self.c_ci(z, z_order)])

    def i_c_tr(self, z, z_s=0):
        """Integral of the trans concentration at leading order."""
        return self.c_tr(z, z_order=-1) - self.c_tr(z_s, z_order=-1)

    def i_c_ci(self, z, z_s=0):
        """Integral of the cis concentration at leading order."""
        return self.c_ci(z, z_order=-1) - self.c_ci(z_s, z_order=-1)

    def i_c(self, z, z_s):
        """Integral of the surfactant concentration at leading order."""
        return np.array([self.i_c_tr(z, z_s), self.i_c_ci(z, z_s)])

    # The following properties should be cached post init
    @property
    def Gamma_ci(self):
        """Surface excess of cis surfactant at leading order."""
        return self.Gamma[1]

    @property
    def Gamma_tr(self):
        """Surface excess of trans surfactant at leading order."""
        return self.Gamma[0]

    @property
    def Gamma(self):
        """Surface excess of surfactant at leading order."""
        params = self.params
        return np.linalg.solve(self.M, params.P @ params.B @ params.K @ self.c(1))

    @property
    def J_ci(self):
        """Kinetic flux of the cis surfactant at leading order."""
        return self.params.Bi_ci * (
            self.params.k_ci * self.c_ci(1) * (1 - self.Gamma_tr - self.Gamma_ci)
            - self.Gamma_ci
        )

    @property
    def J_tr(self):
        """Kinetic flux of the trans surfactant at leading order."""
        return self.params.Bi_tr * (
            self.params.k_tr * self.c_tr(1) * (1 - self.Gamma_tr - self.Gamma_ci)
            - self.Gamma_tr
        )

    @property
    def J(self):
        """Kinetic flux of the surfactant at leading order."""
        return np.array([self.J_tr, self.J_ci])

    @property
    def gamma(self):
        """Surface tension at leading order."""
        return 1 + self.params.Ma * np.log(1 - self.Gamma_tr - self.Gamma_ci)

    @property
    def M(self):  # noqa: N802
        """Matrix M from (5.8)."""
        params = self.params
        return params.A + params.P @ params.B @ np.array(
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
