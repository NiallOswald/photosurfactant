"""Test the solutions to first order problem."""

from photosurfactant.parameters import Parameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.first_order import FirstOrder, Variables
from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.functions import gaussian, super_gaussian, smoothed_square
import numpy as np
import pytest

N_WAVE = 20


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_biharmonic(func):
    """Test the biharmonic equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq = np.array(
        [
            first.psi(xx, y, x_order=4)
            + 2 * first.psi(xx, y, x_order=2, y_order=2)
            + first.psi(xx, y, y_order=4)
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_navier_stokes(func):
    """Test the Navier-Stokes equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq_x = np.array(
        [
            first.pressure(xx, y, x_order=1)
            - first.u(xx, y, x_order=2)
            - first.u(xx, y, y_order=2)
            for y in yy
        ]
    )
    eq_y = np.array(
        [
            first.pressure(xx, y, y_order=1)
            - first.v(xx, y, x_order=2)
            - first.v(xx, y, y_order=2)
            for y in yy
        ]
    )

    assert np.allclose(eq_x, 0)
    assert np.allclose(eq_y, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_continuity(func):
    """Test the continuity equation."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq = np.array([first.u(xx, y, x_order=1) + first.v(xx, y, y_order=1) for y in yy])

    assert np.allclose(eq, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_bulk_concentrations(func):
    """Test bulk surfactant concentration equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)
    yy = np.linspace(0, 1, 100)

    eq_tr = np.array(
        [
            leading.c_tr(y, y_order=1) * first.v(xx, y)
            - 1
            / params.Pen_tr
            * (first.c_tr(xx, y, x_order=2) + first.c_tr(xx, y, y_order=2))
            + params.Dam_tr * (first.c_tr(xx, y) + leading.c_tr(y) * first.f(xx))
            - params.Dam_ci * (first.c_ci(xx, y) + leading.c_ci(y) * first.f(xx))
            for y in yy
        ]
    )
    eq_ci = np.array(
        [
            leading.c_ci(y, y_order=1) * first.v(xx, y)
            - 1
            / params.Pen_ci
            * (first.c_ci(xx, y, x_order=2) + first.c_ci(xx, y, y_order=2))
            - params.Dam_tr * (first.c_tr(xx, y) + leading.c_tr(y) * first.f(xx))
            + params.Dam_ci * (first.c_ci(xx, y) + leading.c_ci(y) * first.f(xx))
            for y in yy
        ]
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_surface_excess(func):
    """Test the surface excess concentration equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    eq_tr = (
        leading.gamma_tr * first.u(xx, 1, x_order=1)
        - 1 / params.Pen_tr_s * first.gamma_tr(xx, x_order=2)
        - first.J_tr(xx)
        + params.Dam_tr * (first.gamma_tr(xx) + leading.gamma_tr * first.f(xx))
        - params.Dam_ci * (first.gamma_ci(xx) + leading.gamma_ci * first.f(xx))
    )
    eq_ci = (
        leading.gamma_ci * first.u(xx, 1, x_order=1)
        - 1 / params.Pen_ci_s * first.gamma_ci(xx, x_order=2)
        - first.J_ci(xx)
        - params.Dam_tr * (first.gamma_tr(xx) + leading.gamma_tr * first.f(xx))
        + params.Dam_ci * (first.gamma_ci(xx) + leading.gamma_ci * first.f(xx))
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_kinetic_flux(func):
    """Test the kinetic fluxes."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    eq_tr = params.Bit_tr * (
        params.k_tr
        * (first.c_tr(xx, 1) + first.S(xx) * leading.c_tr(1, y_order=1))
        * (1 - leading.gamma_tr - leading.gamma_ci)
        - params.k_tr * leading.c_tr(1) * (first.gamma_tr(xx) + first.gamma_ci(xx))
        - first.gamma_tr(xx)
    ) - first.J_tr(xx)
    eq_ci = params.Bit_ci * (
        params.k_ci
        * (first.c_ci(xx, 1) + first.S(xx) * leading.c_ci(1, y_order=1))
        * (1 - leading.gamma_tr - leading.gamma_ci)
        - params.k_ci * leading.c_ci(1) * (first.gamma_tr(xx) + first.gamma_ci(xx))
        - first.gamma_ci(xx)
    ) - first.J_ci(xx)

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_normal_stress(func):
    """Test the normal stress balance."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    eq = (
        -first.pressure(xx, 1)
        + 2 * first.v(xx, 1, y_order=1)
        - leading.tension * first.S(xx, x_order=2)
    )

    assert np.allclose(eq, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_tangential_stress(func):
    """Test the tangential stress balance."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    eq = (
        first.u(xx, 1, y_order=1)
        + first.v(xx, 1, x_order=1)
        - first.tension(xx, x_order=1)
    )

    assert np.allclose(eq, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_mass_balance(func):
    """Test the mass balances."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    eq_tr = params.k_tr * params.chi_tr / params.Pen_tr * (
        first.c_tr(xx, 1, y_order=1) + first.S(xx) * leading.c_tr(1, y_order=2)
    ) + first.J_tr(xx)
    eq_ci = params.k_ci * params.chi_ci / params.Pen_ci * (
        first.c_ci(xx, 1, y_order=1) + first.S(xx) * leading.c_ci(1, y_order=2)
    ) + first.J_ci(xx)

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_kinematic(func):
    """Test the kinematic condition."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    assert np.allclose(first.v(xx, 1), 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_mass_cons(func):
    """Test the mass conservation condition."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    eq = np.trapezoid(first.S(xx), xx)

    assert np.allclose(eq, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_surf_cons(func):
    """Test the surfactant conservation condition."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    integrand = (
        lambda x: first.S(x) * (leading.c_tr(1) + leading.c_ci(1))
        + 1 / (params.chi_tr * params.k_tr) * (first.gamma_tr(x) + first.gamma_ci(x))
        + first.i_c_tr(x, 1)
        + first.i_c_ci(x, 1)
    )
    eq = np.trapezoid(integrand(xx), xx)

    assert np.allclose(eq, 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_no_slip(func):
    """Test no-slip condition on the lower wall."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    assert np.allclose(first.u(xx, 0), 0)
    assert np.allclose(first.v(xx, 0), 0)


@pytest.mark.parametrize(
    "func",
    [gaussian, lambda x: super_gaussian(x, 4.0), lambda x: smoothed_square(x, 0.1)],
)
def test_no_flux(func):
    """Test no-flux condition on the lower wall."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, func_coeffs = fourier_series_coeff(func, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    xx = np.linspace(-params.L, params.L, 100)

    assert np.allclose(first.c_tr(xx, 0, y_order=1), 0)
    assert np.allclose(first.c_ci(xx, 0, y_order=1), 0)
