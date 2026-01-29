"""Test the solutions to first order problem."""

import numpy as np
from pytest_cases import fixture

from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.intensity_functions import (
    gaussian,
)
from photosurfactant.parameters import Parameters
from photosurfactant.semi_analytic import FirstOrder, LeadingOrder, Variables

N_WAVE = 20


@fixture(scope="module")
def params():
    return Parameters()


@fixture(scope="module")
def leading(params: Parameters):
    return LeadingOrder(params)


@fixture(scope="module")
def first(params: Parameters, leading: LeadingOrder):
    # TODO: parametrize_with_cases is bugged. Cannot pass callable as a case
    wavenumbers, func_coeffs = fourier_series_coeff(gaussian, params.L, N_WAVE)
    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))
    return first


def test_biharmonic(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the biharmonic equations."""
    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq = np.array(
        [
            first.psi(xx, y, x_order=4)
            + 2 * first.psi(xx, y, x_order=2, z_order=2)
            + first.psi(xx, y, z_order=4)
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


def test_navier_stokes(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the Navier-Stokes equations."""
    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq_x = np.array(
        [
            first.p(xx, y, x_order=1)
            - first.u(xx, y, x_order=2)
            - first.u(xx, y, z_order=2)
            for y in yy
        ]
    )
    eq_y = np.array(
        [
            first.p(xx, y, z_order=1)
            - first.w(xx, y, x_order=2)
            - first.w(xx, y, z_order=2)
            for y in yy
        ]
    )

    assert np.allclose(eq_x, 0)
    assert np.allclose(eq_y, 0)


def test_continuity(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the continuity equation."""
    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq = np.array([first.u(xx, y, x_order=1) + first.w(xx, y, z_order=1) for y in yy])

    assert np.allclose(eq, 0)


def test_bulk_concentrations(
    params: Parameters, leading: LeadingOrder, first: FirstOrder
):
    """Test bulk surfactant concentration equations."""
    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq_tr = np.array(
        [
            leading.c_tr(y, z_order=1) * first.w(xx, y)
            - 1
            / params.Pe_tr
            * (first.c_tr(xx, y, x_order=2) + first.c_tr(xx, y, z_order=2))
            + params.Da_tr * (first.c_tr(xx, y) + leading.c_tr(y) * first.f(xx))
            - params.Da_ci * (first.c_ci(xx, y) + leading.c_ci(y) * first.f(xx))
            for y in yy
        ]
    )
    eq_ci = np.array(
        [
            leading.c_ci(y, z_order=1) * first.w(xx, y)
            - 1
            / params.Pe_ci
            * (first.c_ci(xx, y, x_order=2) + first.c_ci(xx, y, z_order=2))
            - params.Da_tr * (first.c_tr(xx, y) + leading.c_tr(y) * first.f(xx))
            + params.Da_ci * (first.c_ci(xx, y) + leading.c_ci(y) * first.f(xx))
            for y in yy
        ]
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_surface_excess(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the surface excess concentration equations."""
    xx = np.linspace(-params.L, params.L, 100)

    eq_tr = (
        leading.Gamma_tr * first.u(xx, 1, x_order=1)
        - 1 / params.Pe_tr_s * first.Gamma_tr(xx, x_order=2)
        - first.J_tr(xx)
        + params.Da_tr * (first.Gamma_tr(xx) + leading.Gamma_tr * first.f(xx))
        - params.Da_ci * (first.Gamma_ci(xx) + leading.Gamma_ci * first.f(xx))
    )
    eq_ci = (
        leading.Gamma_ci * first.u(xx, 1, x_order=1)
        - 1 / params.Pe_ci_s * first.Gamma_ci(xx, x_order=2)
        - first.J_ci(xx)
        - params.Da_tr * (first.Gamma_tr(xx) + leading.Gamma_tr * first.f(xx))
        + params.Da_ci * (first.Gamma_ci(xx) + leading.Gamma_ci * first.f(xx))
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_kinetic_flux(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the kinetic fluxes."""
    xx = np.linspace(-params.L, params.L, 100)

    eq_tr = params.Bi_tr * (
        params.k_tr
        * (first.c_tr(xx, 1) + first.S(xx) * leading.c_tr(1, z_order=1))
        * (1 - leading.Gamma_tr - leading.Gamma_ci)
        - params.k_tr * leading.c_tr(1) * (first.Gamma_tr(xx) + first.Gamma_ci(xx))
        - first.Gamma_tr(xx)
    ) - first.J_tr(xx)
    eq_ci = params.Bi_ci * (
        params.k_ci
        * (first.c_ci(xx, 1) + first.S(xx) * leading.c_ci(1, z_order=1))
        * (1 - leading.Gamma_tr - leading.Gamma_ci)
        - params.k_ci * leading.c_ci(1) * (first.Gamma_tr(xx) + first.Gamma_ci(xx))
        - first.Gamma_ci(xx)
    ) - first.J_ci(xx)

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_normal_stress(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the normal stress balance."""
    xx = np.linspace(-params.L, params.L, 100)

    eq = (
        -first.p(xx, 1)
        + 2 * first.w(xx, 1, z_order=1)
        - leading.gamma * first.S(xx, x_order=2)
    )

    assert np.allclose(eq, 0)


def test_tangential_stress(
    params: Parameters, leading: LeadingOrder, first: FirstOrder
):
    """Test the tangential stress balance."""
    xx = np.linspace(-params.L, params.L, 100)

    eq = (
        first.u(xx, 1, z_order=1)
        + first.w(xx, 1, x_order=1)
        - first.gamma(xx, x_order=1)
    )

    assert np.allclose(eq, 0)


def test_mass_balance(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the mass balances."""
    xx = np.linspace(-params.L, params.L, 100)

    eq_tr = params.k_tr * params.chi_tr / params.Pe_tr * (
        first.c_tr(xx, 1, z_order=1) + first.S(xx) * leading.c_tr(1, z_order=2)
    ) + first.J_tr(xx)
    eq_ci = params.k_ci * params.chi_ci / params.Pe_ci * (
        first.c_ci(xx, 1, z_order=1) + first.S(xx) * leading.c_ci(1, z_order=2)
    ) + first.J_ci(xx)

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_kinematic(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the kinematic condition."""
    xx = np.linspace(-params.L, params.L, 100)
    assert np.allclose(first.w(xx, 1), 0)


def test_mass_cons(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the mass conservation condition."""
    xx = np.linspace(-params.L, params.L, 100)
    eq = np.trapezoid(first.S(xx), xx)
    assert np.allclose(eq, 0)


def test_surf_cons(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test the surfactant conservation condition."""
    xx = np.linspace(-params.L, params.L, 100)

    integrand = (  # noqa: E731
        lambda x: first.S(x) * (leading.c_tr(1) + leading.c_ci(1))
        + 1 / (params.chi_tr * params.k_tr) * (first.Gamma_tr(x) + first.Gamma_ci(x))
        + first.i_c_tr(x, 1)
        + first.i_c_ci(x, 1)
    )
    eq = np.trapezoid(integrand(xx), xx)

    assert np.allclose(eq, 0)


def test_no_slip(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test no-slip condition on the lower wall."""
    xx = np.linspace(-params.L, params.L, 100)
    assert np.allclose(first.u(xx, 0), 0)
    assert np.allclose(first.w(xx, 0), 0)


def test_no_flux(params: Parameters, leading: LeadingOrder, first: FirstOrder):
    """Test no-flux condition on the lower wall."""
    xx = np.linspace(-params.L, params.L, 100)
    assert np.allclose(first.c_tr(xx, 0, z_order=1), 0)
    assert np.allclose(first.c_ci(xx, 0, z_order=1), 0)
