"""Test the eigenfunctions to first order problem."""

import numpy as np

from photosurfactant.first_order import FirstOrder, Variables
from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.parameters import Parameters

N_WAVE = 20


def test_streamfunction():
    """Test the streamfunction."""
    params = Parameters()
    leading = LeadingOrder(params)

    wavenumbers, _ = fourier_series_coeff(lambda x: 0.0, params.L, N_WAVE)

    first = FirstOrder(
        wavenumbers,
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                k**4 * first._psi(k, y)
                - 2 * k**2 * first._psi(k, y, z_order=2)
                + first._psi(k, y, z_order=4)
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
        params,
        leading,
    )

    zz = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                first._q_0(k, z, z_order=2)
                - (params.Lambda + k**2 * np.eye(2)) @ first._q_0(k, z)
                for k in wavenumbers
            ]
            for z in zz
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
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                (
                    first._q_1(k, y, z_order=2)
                    - (params.Lambda + k**2 * np.eye(2)) @ first._q_1(k, y)
                )
                - (
                    Variables.f[np.newaxis, :]
                    * leading.B
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
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                (
                    first._q_2(k, y, z_order=2)
                    - (params.Lambda + k**2 * np.eye(2)) @ first._q_2(k, y)
                )
                - (
                    1.0j
                    * k
                    * first._psi(k, y)[np.newaxis, :]
                    * leading.c_ci(y, z_order=1)
                    * params.Pe_ci
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
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                (
                    first._q(k, y, z_order=2)
                    - (params.Lambda + k**2 * np.eye(2)) @ first._q(k, y)
                )
                - (
                    Variables.f[np.newaxis, :]
                    * leading.B
                    * zeta
                    * np.cosh(y * np.sqrt(zeta))
                    * np.array([0, 1])[:, np.newaxis]
                )
                - (
                    1.0j
                    * k
                    * first._psi(k, y)[np.newaxis, :]
                    * leading.c_ci(y, z_order=1)
                    * params.Pe_ci
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
        params,
        leading,
    )

    yy = np.linspace(0, 1, 10)

    eq = np.array(
        [
            [
                first._c(k, y, z_order=2)
                - (params.A + k**2 * np.eye(2)) @ first._c(k, y)
                - (Variables.f * (params.A @ leading.c(y)[:, np.newaxis]))
                + (
                    1.0j
                    * k
                    * first._psi(k, y)
                    * (params.P @ leading.c(y, z_order=1)[:, np.newaxis])
                )
                for k in wavenumbers
            ]
            for y in yy
        ]
    )

    assert np.allclose(eq, 0)
