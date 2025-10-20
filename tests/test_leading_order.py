"""Test the solutions to leading order problem."""

import numpy as np

from photosurfactant.leading_order import LeadingOrder
from photosurfactant.parameters import Parameters


def test_bulk_concentrations():
    """Test bulk surfactant concentration equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    yy = np.linspace(0, 1, 100)

    eq_tr = (
        (1 / params.Pe_tr) * leading.c_tr(yy, z_order=2)
        - params.Da_tr * leading.c_tr(yy)
        + params.Da_ci * leading.c_ci(yy)
    )
    eq_ci = (
        (1 / params.Pe_ci) * leading.c_ci(yy, z_order=2)
        + params.Da_tr * leading.c_tr(yy)
        - params.Da_ci * leading.c_ci(yy)
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_surface_excess():
    """Test the surface excess concentration equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    eq_tr = (
        leading.J_tr - params.Da_tr * leading.Gamma_tr + params.Da_ci * leading.Gamma_ci
    )
    eq_ci = (
        leading.J_ci + params.Da_tr * leading.Gamma_tr - params.Da_ci * leading.Gamma_ci
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_kinetic_flux():
    """Test the kinetic fluxes."""
    params = Parameters()
    leading = LeadingOrder(params)

    eq = leading.J_tr + leading.J_ci

    assert np.allclose(eq, 0)


def test_mass_balance():
    """Test the mass balances."""
    params = Parameters()
    leading = LeadingOrder(params)

    eq_tr = (
        params.k_tr * params.chi_tr / params.Pe_tr * leading.c_tr(1, z_order=1)
        + leading.J_tr
    )
    eq_ci = (
        params.k_ci * params.chi_ci / params.Pe_ci * leading.c_ci(1, z_order=1)
        + leading.J_ci
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_surf_cons():
    """Test the surfactant conservation condition."""
    params = Parameters()
    leading = LeadingOrder(params)

    eq = (
        (leading.i_c_tr(1) + leading.i_c_ci(1))
        + 1 / (params.k_tr * params.chi_tr) * (leading.Gamma_tr + leading.Gamma_ci)
        - 1
    )

    assert np.allclose(eq, 0)


def test_no_flux():
    """Test no-flux condition on the lower wall."""
    params = Parameters()
    leading = LeadingOrder(params)

    assert np.allclose(leading.c(0, z_order=1), 0)
