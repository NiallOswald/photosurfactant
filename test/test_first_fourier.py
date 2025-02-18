"""Test the eigenfunctions to first order problem."""

from photosurfactant.parameters import Parameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.first_order import FirstOrder, Variables
from photosurfactant.fourier import fourier_series_coeff
import numpy as np

N_WAVE = 20


def dummy_condition(k):
    """Dummy condition for testing."""  # noqa: D401
    return (Variables.f, 1) if k == 0 else (Variables.f, 0)


def test_streamfunction():
    """Test the streamfunction."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        dummy_condition,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                k**4 * first._psi(k, y)
                - 2 * k**2 * first._psi(k, y, y_order=2)
                + first._psi(k, y, y_order=4)
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


def test_p_0():
    """Test the distinguished complementary eigenfunction p_0."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        dummy_condition,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                first._p_0(k, y, y_order=2)
                - (params.Lambda + k**2 * np.eye(2)) @ first._p_0(k, y)
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


def test_p_1():
    """Test the distinguished eigenfunction component p_1."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        dummy_condition,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                (
                    first._p_1(k, y, y_order=2)
                    - (params.Lambda + k**2 * np.eye(2)) @ first._p_1(k, y)
                )
                - (
                    Variables.f[np.newaxis, :]
                    * leading.B_0
                    * params.zeta
                    * np.cosh(y * np.sqrt(params.zeta))
                    * np.array([0, 1])[:, np.newaxis]
                )
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


def test_p_2():
    """Test the distinguished eigenfunction component p_2."""
    params = Parameters()
    leading = LeadingOrder(params)

    alpha, eta = params.alpha, params.eta

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        dummy_condition,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                (
                    first._p_2(k, y, y_order=2)
                    - (params.Lambda + k**2 * np.eye(2)) @ first._p_2(k, y)
                )
                - (
                    1.0j
                    * k
                    * first._psi(k, y)[np.newaxis, :]
                    * leading.d_c_ci(y)
                    * params.Pen_ci
                    / (alpha + eta)
                    * np.array([eta**2 - eta, eta**2 + alpha])[:, np.newaxis]
                )
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


def test_p():
    """Test the eigenfunction for the concentration in distinguished coordinates."""
    params = Parameters()
    leading = LeadingOrder(params)

    alpha, eta, zeta = params.alpha, params.eta, params.zeta

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        dummy_condition,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                (
                    first._p(k, y, y_order=2)
                    - (params.Lambda + k**2 * np.eye(2)) @ first._p(k, y)
                )
                - (
                    Variables.f[np.newaxis, :]
                    * leading.B_0
                    * zeta
                    * np.cosh(y * np.sqrt(zeta))
                    * np.array([0, 1])[:, np.newaxis]
                )
                - (
                    1.0j
                    * k
                    * first._psi(k, y)[np.newaxis, :]
                    * leading.d_c_ci(y)
                    * params.Pen_ci
                    / (alpha + eta)
                    * np.array([eta**2 - eta, eta**2 + alpha])[:, np.newaxis]
                )
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)


def test_bulk_concentration():
    """Test bulk surfactant concentrations."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        dummy_condition,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                first._c(k, y, y_order=2)
                - (params.A + k**2 * np.eye(2)) @ first._c(k, y)
                - (Variables.f * (params.A @ leading.c(y)[:, np.newaxis]))
                + (
                    1.0j
                    * k
                    * first._psi(k, y)
                    * (params.P @ leading.d_c(y)[:, np.newaxis])
                )
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)
