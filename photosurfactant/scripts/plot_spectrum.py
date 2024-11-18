#! /usr/bin/env python
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.first_order import FirstOrder
from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.utils import (
    parameter_parser,
    plot_parser,
    leading_order_parser,
    first_order_parser,
)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np


def plot_spectrum():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the first order surfactant concentrations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    plot_parser(parser)
    leading_order_parser(parser)
    first_order_parser(parser)
    args = parser.parse_args()

    root_index = args.root_index
    func = lambda x: 1.0  # noqa: E731

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))

    omega, func_coeffs = fourier_series_coeff(func, params.L, plot_params.wave_count)
    func_coeffs = np.ones_like(func_coeffs, dtype=np.complex128)

    # Solve leading order problem
    leading = LeadingOrder(params, root_index)

    # Solve first order problem
    first = FirstOrder(omega, func_coeffs, params, leading)

    # Figure setup
    plt = plot_params.plt

    # Plot the spectrum
    plt.figure()
    plt.plot(
        omega[plot_params.wave_count :],
        np.abs(first.S_f()[plot_params.wave_count + 1 :]),
        "k-",
    )
    plt.yscale("log")
    plt.xlabel(r"$k_n$")
    plt.ylabel(r"$|S^{(n)}|$")
    plt.tight_layout()

    if plot_params.save:
        plt.savefig(
            plot_params.path + f"spectrum{plot_params.label}.{plot_params.format}",
        )
    else:
        plt.show()
