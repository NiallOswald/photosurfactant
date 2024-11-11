#! /usr/bin/env python
from photosurfactant.parameters import Parameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.first_order import FirstOrder
from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.utils import parameter_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt


N = 100
GRID_N = 10


def plot_error():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the first order surfactant concentrations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    args = parser.parse_args()

    # Extract parameters
    kwargs = vars(args)
    Dam_tr = kwargs["Dam_tr"]
    Dam_ci = kwargs["Dam_ci"]

    params = Parameters.from_dict(vars(args))

    func = lambda x: 1.0  # noqa: E731
    omega, func_coeffs = fourier_series_coeff(func, params.L, N)

    # Solve first order problem
    leading = LeadingOrder(params)
    first = FirstOrder(omega, func_coeffs, params, leading)

    # Solve the linearised system
    yy = np.linspace(0, 1, GRID_N)
    cc_tr_1 = np.array([first.c_tr(np.array([0]), y) for y in yy])[:, 0]
    cc_tr_0 = leading.c_tr(yy)

    # Set delta values
    delta_values = 2.0 ** np.arange(0, -20, -1)
    error = np.zeros((len(delta_values), 2))

    for i, delta in enumerate(delta_values):
        # Update parameters
        kwargs["Dam_tr"] = (1 + delta) * Dam_tr
        kwargs["Dam_ci"] = (1 + delta) * Dam_ci

        params_delta = Parameters.from_dict(kwargs)
        leading_delta = LeadingOrder(params_delta)
        cc_tr_delta = leading_delta.c_tr(yy)

        # Compute error
        error[i, 0] = np.linalg.norm(cc_tr_delta - cc_tr_0)
        error[i, 1] = np.linalg.norm(cc_tr_delta - (cc_tr_0 + delta * cc_tr_1))
        print(f"Error at delta = {delta}. Leading: {error[i, 0]}, first: {error[i, 1]}")

    plt.loglog(delta_values, error[:, 0], "o-", label="Leading order")
    plt.loglog(delta_values, error[:, 1], "o-", label="First order")
    plt.loglog(delta_values, error[0, 0] * delta_values, "k:", label=r"O($\delta$)")
    plt.loglog(
        delta_values, error[0, 1] * delta_values**2, "k--", label=r"O($\delta^2$)"
    )
    plt.xlabel(r"$\delta$")
    plt.ylabel("Error")

    plt.legend()
    plt.show()
