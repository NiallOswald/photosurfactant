#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.parameters import Parameters
from photosurfactant.semi_analytic.first_order import FirstOrder, Variables
from photosurfactant.semi_analytic.leading_order import LeadingOrder
from photosurfactant.utils import parameter_parser
from photosurfactant.utils.arg_parser import parameter_parser

WAVE_N = 5
GRID_N = 100


def plot_error():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the first order surfactant concentrations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    args = parser.parse_args()

    # Extract parameters
    kwargs = vars(args)
    Da_tr = kwargs["Da_tr"]
    Da_ci = kwargs["Da_ci"]

    params = Parameters.from_dict(vars(args))

    wavenumbers, func_coeffs = fourier_series_coeff(lambda x: 1.0, params.L, WAVE_N)

    # Solve first order problem
    leading = LeadingOrder(params)
    first = FirstOrder(wavenumbers, params, leading)
    first.solve(lambda n: (Variables.f, func_coeffs[n]))

    # Solve the linearised system
    yy = np.linspace(0, 1, GRID_N)
    cc_tr_1 = np.array([first.c_tr(np.array([0]), y) for y in yy])[:, 0]
    cc_tr_0 = leading.c_tr(yy)

    # Set delta values
    delta_values = 2.0 ** np.arange(0, -20, -1)
    error = np.zeros((len(delta_values), 2))

    for i, delta in enumerate(delta_values):
        # Update parameters
        kwargs["Da_tr"] = (1 + delta) * Da_tr
        kwargs["Da_ci"] = (1 + delta) * Da_ci

        params_delta = Parameters.from_dict(kwargs)
        leading_delta = LeadingOrder(params_delta)
        cc_tr_delta = leading_delta.c_tr(yy)

        # Compute error
        error[i, 0] = np.linalg.norm(cc_tr_delta - cc_tr_0)
        error[i, 1] = np.linalg.norm(cc_tr_delta - (cc_tr_0 + delta * cc_tr_1))
        print(
            f"Error at delta = {delta:.2e}. Leading: {error[i, 0]:.2e}, "
            f"first: {error[i, 1]:.2e}"
        )

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
