"""Linear stability of a falling film with photosurfactant."""

from dataclasses import dataclass
from enum import IntEnum, auto

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvals

from photosurfactant.parameters import Parameters
from photosurfactant.semi_analytic import LeadingOrder
from photosurfactant.utils.chebyshev import chebyshev


@dataclass(frozen=True, kw_only=True)
class FallingFilmParameters(Parameters):
    Re: float
    Ca: float
    Ct: float


class BulkVariable(IntEnum):
    v = 0
    c_tr = auto()
    c_ci = auto()


class InterfaceVariable(IntEnum):
    Gamma_tr = 0
    Gamma_ci = auto()
    S = auto()


Variable = BulkVariable | InterfaceVariable

Y = np.polynomial.Polynomial([0, 1], symbol="y")


class FallingFilm:
    u_bar = Y * (2 - Y)

    def __init__(self, params: FallingFilmParameters, n: int):
        self.params = params
        self.n = n

        self.leading = LeadingOrder(params)
        self.D, self.y = chebyshev(self.n)

        # Reshape to match matrices
        self.y = self.y[:, np.newaxis]

    def stability(self, k: float) -> float:
        return np.max(self.eigenvalues(k))

    def eigenvalues(self, k: float, tol: float = 1e2) -> NDArray[np.float64]:
        A, B = self._assemble(k)
        vals = eigvals(A, B).real  # consider only real part
        vals = vals[np.isfinite(vals)]  # discard infinite eigenvalues
        vals = vals[vals < tol]  # discard excessively large values

        return vals

    def inspect(self, k: float) -> None:
        fig, axs = plt.subplots(1, 2)

        A, B = self._assemble(k)
        axs[0].spy(A)
        axs[1].spy(B)

        plt.show()

    def _assemble(
        self, k: float
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        eq_methods = [
            self._momentum_eq,
            self._bulk_concentration_eq,
            self._interfacial_concentration_eq,
            self._kinematic_eq,
        ]

        A, B = [], []
        for method in eq_methods:
            a_eqs, b_eqs = method(k)
            for eq in a_eqs:
                A.append(eq)

            for eq in b_eqs:
                B.append(eq)

        A, B = np.vstack(A), np.vstack(B)

        return self._apply_boundary_conditions(k, A, B)

    def _apply_boundary_conditions(
        self, k: float, A: NDArray[np.complex128], B: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        # Boundary conditions z = 0
        # No-slip
        A[self.n * BulkVariable.v, :] = self.v[0]
        B[self.n * BulkVariable.v, :] = 0

        A[self.n * BulkVariable.v + 1, :] = self.D[0] @ self.v
        B[self.n * BulkVariable.v + 1, :] = 0

        # No-flux
        A[self.n * BulkVariable.c_tr, :] = self.D[0] @ self.c_tr
        B[self.n * BulkVariable.c_tr, :] = 0

        A[self.n * BulkVariable.c_ci, :] = self.D[0] @ self.c_ci
        B[self.n * BulkVariable.c_ci, :] = 0

        # Boundary conditions z = 1
        # Normal stress balance
        a, b = self._normal_stress_eq(k)
        A[self.n * (BulkVariable.v + 1) - 1, :] = a[0]
        B[self.n * (BulkVariable.v + 1) - 1, :] = b[0]

        # Tangential stress balance
        a, b = self._tangential_stress_eq(k)
        A[self.n * (BulkVariable.v + 1) - 2, :] = a[0]
        B[self.n * (BulkVariable.v + 1) - 2, :] = b[0]

        # Mass balance
        a, b = self._mass_balance_eq(k)
        A[self.n * (BulkVariable.c_tr + 1) - 1, :] = a[0]
        B[self.n * (BulkVariable.c_tr + 1) - 1, :] = b[0]

        A[self.n * (BulkVariable.c_ci + 1) - 1, :] = a[1]
        B[self.n * (BulkVariable.c_ci + 1) - 1, :] = b[1]

        return A, B

    def _to_arr(self, var: Variable) -> NDArray[np.complex128]:
        """Convert a symbol to an array."""
        arr_len = self.n * len(BulkVariable) + len(InterfaceVariable)
        match var:
            case BulkVariable():
                # auto() starts indexing at 1
                return np.eye(self.n, arr_len, k=var * self.n, dtype=np.complex128)
            case InterfaceVariable():
                return np.eye(
                    1, arr_len, k=var + self.n * len(BulkVariable), dtype=np.complex128
                )[0]
            case _:
                raise TypeError

    @property
    def p_bar(self) -> np.polynomial.Polynomial:
        return 2 * self.params.Ct * (1 - Y)

    @property
    def v(self) -> NDArray[np.complex128]:
        return self._to_arr(BulkVariable.v)

    @property
    def c_tr(self) -> NDArray[np.complex128]:
        return self._to_arr(BulkVariable.c_tr)

    @property
    def c_ci(self) -> NDArray[np.complex128]:
        return self._to_arr(BulkVariable.c_ci)

    @property
    def Gamma_tr(self) -> NDArray[np.complex128]:
        return self._to_arr(InterfaceVariable.Gamma_tr)

    @property
    def Gamma_ci(self) -> NDArray[np.complex128]:
        return self._to_arr(InterfaceVariable.Gamma_ci)

    @property
    def S(self) -> NDArray[np.complex128]:
        return self._to_arr(InterfaceVariable.S)

    @property
    def J_tr(self) -> NDArray[np.complex128]:
        return self.params.Bi_tr * (
            self.params.k_tr
            * (self.c_tr[self.n - 1] + self.leading.c_tr(1.0, z_order=1) * self.S)
            * (1 - self.leading.Gamma_tr - self.leading.Gamma_ci)
            - self.params.k_tr
            * self.leading.c_tr(1.0)
            * (self.Gamma_tr + self.Gamma_ci)
            - self.Gamma_tr
        )

    @property
    def J_ci(self) -> NDArray[np.complex128]:
        return self.params.Bi_ci * (
            self.params.k_ci
            * (self.c_ci[self.n - 1] + self.leading.c_ci(1.0, z_order=1) * self.S)
            * (1 - self.leading.Gamma_tr - self.leading.Gamma_ci)
            - self.params.k_ci
            * self.leading.c_ci(1.0)
            * (self.Gamma_tr + self.Gamma_ci)
            - self.Gamma_ci
        )

    @property
    def gamma(self) -> NDArray[np.complex128]:
        return (
            -self.params.Ma
            * (self.Gamma_tr + self.Gamma_ci)
            / (1 - self.leading.Gamma_tr - self.leading.Gamma_ci)
        )

    def _momentum_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        D, I = self.D, np.eye(self.n)
        D_2 = D @ D

        A = [
            (D_2 - k**2 * I - 1.0j * k * self.params.Re * self.u_bar(self.y) * I)
            @ (D_2 - k**2 * I)
            @ self.v
            + (1.0j * k * self.params.Re * self.u_bar.deriv(2)(self.y)) * self.v
        ]
        B = [self.params.Re * (D_2 - k**2 * I) @ self.v]

        return A, B

    def _bulk_concentration_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        params, D, I = self.params, self.D, np.eye(self.n)
        D_2 = D @ D

        A = [
            1 / params.Pe_tr * (D_2 - k**2 * I) @ self.c_tr
            - params.Da_tr * self.c_tr
            + params.Da_ci * self.c_ci
            - 1.0j * k * self.u_bar(self.y) * self.c_tr
            - self.leading.c_tr(self.y, z_order=1) * self.v,
            1 / params.Pe_ci * (D_2 - k**2 * I) @ self.c_ci
            + params.Da_tr * self.c_tr
            - params.Da_ci * self.c_ci
            - 1.0j * k * self.u_bar(self.y) * self.c_ci
            - self.leading.c_ci(self.y, z_order=1) * self.v,
        ]
        B = [self.c_tr, self.c_ci]

        return A, B

    def _interfacial_concentration_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        params = self.params

        A = [
            self.leading.Gamma_tr
            * (
                self.D[self.n - 1] @ self.v
                - 1.0j * k * self.u_bar.deriv(1)(1.0) * self.S
            )
            - 1.0j * k * self.u_bar(1.0) * self.Gamma_tr
            - (k**2) / params.Pe_tr * self.Gamma_tr
            + self.J_tr
            - params.Da_tr * self.Gamma_tr
            + params.Da_ci * self.Gamma_ci,
            self.leading.Gamma_ci
            * (
                self.D[self.n - 1] @ self.v
                - 1.0j * k * self.u_bar.deriv(1)(1.0) * self.S
            )
            - 1.0j * k * self.u_bar(1.0) * self.Gamma_ci
            - (k**2) / params.Pe_ci * self.Gamma_ci
            + self.J_ci
            + params.Da_tr * self.Gamma_tr
            - params.Da_ci * self.Gamma_ci,
        ]
        B = [self.Gamma_tr, self.Gamma_ci]

        return A, B

    def _normal_stress_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        D, I, params = self.D, np.eye(self.n), self.params
        D_2 = D @ D

        A = [
            (
                (D_2 - (3 * k**2 + 1.0j * k * params.Re * self.u_bar(1.0)) * I)
                @ D
                @ self.v
            )[self.n - 1]
            + 1.0j * k * params.Re * self.u_bar.deriv(1)(1.0) * self.v[self.n - 1]
            + k**2
            * (
                self.p_bar.deriv(1)(1.0)
                + 2.0j * k * self.u_bar.deriv(1)(1.0)
                - k**2 / params.Ca * self.leading.gamma
            )
            * self.S
        ]
        B = [params.Re * D[self.n - 1] @ self.v]

        return A, B

    def _tangential_stress_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        D_2, I = self.D @ self.D, np.eye(self.n)

        A = [
            (D_2 + k**2 * I)[self.n - 1] @ self.v
            - 1.0j * k * self.u_bar.deriv(2)(1.0) * self.S
            - k**2 / self.params.Ca * self.gamma
        ]
        B = [0.0 * self.S]  # placeholder for zero

        return A, B

    def _mass_balance_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        params = self.params
        A = [
            params.Pe_tr
            / (params.k_tr * params.chi_tr)
            * (
                self.D[self.n - 1] @ self.c_tr
                + self.leading.c_tr(1.0, z_order=2) * self.S
            )
            + self.J_tr,
            params.Pe_ci
            / (params.k_ci * params.chi_ci)
            * (
                self.D[self.n - 1] @ self.c_ci
                + self.leading.c_ci(1.0, z_order=2) * self.S
            )
            + self.J_ci,
        ]
        B = [0.0 * self.S, 0.0 * self.S]  # placeholder for zero

        return A, B

    def _kinematic_eq(
        self, k: float
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        A = [self.v[self.n - 1] - 1.0j * k * self.u_bar(1.0) * self.S]
        B = [self.S]

        return A, B
