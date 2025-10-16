"""Limiting solutions for large and small Damkohler numbers."""

import numpy as np

from .parameters import Parameters


class SmallDam:
    """Leading order solution for a uniform intensity at small Damkohler number."""

    def __init__(self, params: Parameters, root_index: int = -1):
        """Initalise solution to the small Damkohler model.

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

        (self._c_tr, self._c_ci) = self.roots[root_index]

    def _initialize(self):
        """Initialize the solution."""
        params = self.params

        # Solve for c_tr
        poly = np.poly1d(
            [
                (params.k_tr * (params.k_ci - params.k_tr)),
                (
                    -params.k_tr * params.k_ci
                    + (
                        1
                        + (1 + params.alpha * (1 - params.k_tr * params.chi_tr))
                        / ((1 + params.alpha) * params.k_tr * params.chi_tr)
                    )
                    * (params.k_ci - params.k_tr)
                ),
                (
                    params.alpha
                    / (1 + params.alpha)
                    * (
                        params.k_tr
                        - (
                            2
                            + (1 - params.k_tr * params.chi_tr)
                            / (params.k_tr * params.chi_tr)
                            * params.k_tr
                        )
                        * params.k_ci
                    )
                ),
                ((params.alpha / (1 + params.alpha)) ** 2 * params.k_ci),
            ]
        )
        c_tr = poly.roots

        # Solve for c_ci
        c_ci = -(
            (1 + params.alpha) * params.chi_tr * params.k_tr * c_tr**2
            + (
                (1 + params.alpha) * (1 + params.chi_tr)
                - params.alpha * params.chi_tr * params.k_tr
            )
            * c_tr
            - params.alpha * params.chi_tr
        ) / (params.chi_tr * params.k_ci * (1 + (1 + params.alpha) * (c_tr - 1)))

        self.roots = [(a, b) for a, b in zip(c_tr, c_ci)]

    def _trim_roots(self):
        """Trim any invalid roots."""
        verified_roots = []
        for c_tr, c_ci in self.roots:
            if np.isclose(c_tr.imag, 0):
                self._c_tr, self._c_ci = c_tr.real, c_ci.real
            else:
                continue

            # Check if any of the concentrations are negative
            if self.c_tr < 0 or self.c_ci < 0:
                continue
            elif self.gamma_tr < 0 or self.gamma_ci < 0:
                continue

            verified_roots.append([self._c_tr, self._c_ci])

        self.roots = verified_roots

    @property
    def c_tr(self):
        """Concentration of trans surfactant at small Damkohler number."""
        return self._c_tr

    @property
    def c_ci(self):
        """Concentration of cis surfactant at small Damkohler number."""
        return self._c_ci

    @property
    def gamma_tr(self):
        """Surface excess of trans surfactant at small Damkohler number."""
        params = self.params
        return (
            params.k_tr
            * self.c_tr
            / (1 + params.k_tr * self.c_tr + params.k_ci * self.c_ci)
        )

    @property
    def gamma_ci(self):
        """Surface excess of cis surfactant at small Damkohler number."""
        params = self.params
        return (
            params.k_ci
            * self.c_ci
            / (1 + params.k_tr * self.c_tr + params.k_ci * self.c_ci)
        )

    @property
    def tension(self):
        """Surface tension at small Damkohler number."""
        return 1 + self.params.Man * np.log(1 - self.gamma_tr - self.gamma_ci)


class LargeDam:
    """Leading order solution for a uniform intensity at large Damkohler number."""

    def __init__(self, params: Parameters, root_index: int = -1):
        """Initalise solution to the large Damkohler model.

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

        self._gamma_ci = self.roots[root_index]

    def _initialize(self):
        """Initialize the solution."""
        params = self.params

        # Define additional parameters
        a = params.alpha * params.Bit_tr * params.k_tr + params.Bit_ci * params.k_ci
        b = params.alpha * params.Bit_tr * (1 + params.k_tr + 1 / params.chi_tr)
        c = params.Bit_ci * (1 + params.k_ci + 1 / params.chi_ci)

        # Solve for gamma_ci
        poly = np.poly1d(
            [
                (1 + params.alpha) / (params.k_tr * params.chi_tr) * a,
                -(b + c),
                a / (1 + params.alpha),
            ]
        )
        self.roots = poly.roots

    def _trim_roots(self):
        """Trim any invalid roots."""
        verified_roots = []
        for root in self.roots:
            if np.isclose(root.imag, 0):
                self._gamma_ci = root.real
            else:
                continue

            # Check if any of the concentrations are negative
            if self.c_ci < 0 or self.c_tr < 0:
                continue
            elif self.gamma_ci < 0:
                continue

            verified_roots.append(self._gamma_ci)

        self.roots = verified_roots

    @property
    def c_tr(self):
        """Concentration of trans surfactant at large Damkohler number."""
        return self.params.alpha * self.c_ci

    @property
    def c_ci(self):
        """Concentration of cis surfactant at large Damkohler number."""
        params = self.params
        return (
            1 / (1 + params.alpha) - 1 / (params.k_tr * params.chi_tr) * self.gamma_ci
        )

    @property
    def gamma_tr(self):
        """Surface excess of trans surfactant at large Damkohler number."""
        return self.params.alpha * self.gamma_ci

    @property
    def gamma_ci(self):
        """Surface excess of cis surfactant at large Damkohler number."""
        return self._gamma_ci

    @property
    def J_tr(self):
        """Kinetic flux of the trans surfactant at large Damkohler number."""
        return self.params.Bit_tr * (
            self.params.k_tr * self.c_tr * (1 - self.gamma_tr - self.gamma_ci)
            - self.gamma_tr
        )

    @property
    def J_ci(self):
        """Kinetic flux of the cis surfactant at large Damkohler number."""
        return self.params.Bit_ci * (
            self.params.k_ci * self.c_ci * (1 - self.gamma_tr - self.gamma_ci)
            - self.gamma_ci
        )

    @property
    def tension(self):
        """Surface tension at large Damkohler number."""
        return 1 + self.params.Man * np.log(1 - self.gamma_tr - self.gamma_ci)
