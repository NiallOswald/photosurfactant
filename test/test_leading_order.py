"""Test the solutions to leading order problem."""

from photosurfactant.parameters import Parameters
from photosurfactant.leading_order import LeadingOrder
import numpy as np
import pytest


def test_bulk_concentrations():
    """Test bulk surfactant concentration equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    yy = np.linspace(0, 1, 100)

    eq_tr = (
        (1 / params.Pen_tr) * leading.d2_c_tr(yy)
        - params.Dam_tr * leading.c_tr(yy)
        + params.Dam_ci * leading.c_ci(yy)
    )
    eq_ci = (
        (1 / params.Pen_ci) * leading.d2_c_ci(yy)
        + params.Dam_tr * leading.c_tr(yy)
        - params.Dam_ci * leading.c_ci(yy)
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_surface_excess():
    """Test the surface excess concentration equations."""
    params = Parameters()
    leading = LeadingOrder(params)

    eq_tr = (
        leading.J_tr
        - params.Dam_tr * leading.gamma_tr
        + params.Dam_ci * leading.gamma_ci
    )
    eq_ci = (
        leading.J_ci
        + params.Dam_tr * leading.gamma_tr
        - params.Dam_ci * leading.gamma_ci
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
        params.k_tr * params.chi_tr / params.Pen_tr * leading.d_c_tr(1) + leading.J_tr
    )
    eq_ci = (
        params.k_ci * params.chi_ci / params.Pen_ci * leading.d_c_ci(1) + leading.J_ci
    )

    assert np.allclose(eq_tr, 0)
    assert np.allclose(eq_ci, 0)


def test_surf_cons():
    """Test the surfactant conservation condition."""
    params = Parameters()
    leading = LeadingOrder(params)

    eq = (
        2 * params.L * (leading.i_c_tr(1) + leading.i_c_ci(1))
        + (2 * params.L)
        / (params.k_tr * params.chi_tr)
        * (leading.gamma_tr + leading.gamma_ci)
        - 1
    )

    assert np.allclose(eq, 0)


def test_no_flux():
    """Test no-flux condition on the lower wall."""
    params = Parameters()
    leading = LeadingOrder(params)

    assert np.allclose(leading.d_c(0), 0)
