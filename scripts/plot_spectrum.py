#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from photosurfactant.semi_analytic.first_order import FirstOrder, Variables
from photosurfactant.fourier import convolution_coeff, fourier_series_coeff
from photosurfactant.intensity_functions import mollifier, square_wave
from photosurfactant.semi_analytic.leading_order import LeadingOrder
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.utils.arg_parser import (
    first_order_parser,
    leading_order_parser,
    parameter_parser,
    plot_parser,
)


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

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))

    wavenumbers, _ = fourier_series_coeff(
        lambda x: 0.0, params.L, plot_params.wave_count
    )

    _, square_wave_coeffs = fourier_series_coeff(
        lambda x: square_wave(x), params.L, plot_params.wave_count
    )

    _, mol_wave_coeffs = convolution_coeff(
        lambda x: square_wave(x),
        mollifier(plot_params.delta),
        params.L,
        plot_params.wave_count,
    )

    # Solve leading order problem
    leading = LeadingOrder(params, root_index)

    # Solve first order problem
    first = FirstOrder(wavenumbers, params, leading)
    first.solve(lambda n: (Variables.f, 1.0))

    S_n = abs(first.solution @ Variables.S)

    # Calculate the slope of the spectrum on a log plot
    index = S_n > 1e-20
    slope = np.polyfit(wavenumbers[index], np.log(S_n[index]), 1)

    # Figure setup
    plt = plot_params.plt

    # Plot the spectrum
    plt.figure()
    plt.plot(wavenumbers[1:], S_n[1:], "k-")
    plt.plot(
        wavenumbers,
        np.exp(slope[1] + slope[0] * wavenumbers),
        "k--",
        label=r"$e^{" f"{slope[0] * np.pi / params.L:.2f}" r"n}$",
    )
    plt.yscale("log")
    plt.xlabel(r"$k_n$")
    plt.ylabel(r"$|\tilde{S}^{(n)}|$")
    plt.legend()
    plt.tight_layout()

    if plot_params.save:
        plt.savefig(
            plot_params.path
            + f"interface_spectrum{plot_params.label}.{plot_params.format}",
            bbox_inches="tight",
        )
    else:
        plt.show()

    # Plot the square wave spectrum
    plt.figure()
    plt.plot(
        wavenumbers,
        square_wave_coeffs.real,
        "k-",
        label="Square wave",
    )
    plt.plot(
        wavenumbers,
        mol_wave_coeffs.real,
        "r--",
        label="Mollified square wave",
    )
    plt.xlabel(r"$k_n$")
    plt.ylabel(r"$S^{(n)}$")
    plt.legend()
    plt.tight_layout()

    if plot_params.save:
        plt.savefig(
            plot_params.path
            + f"square_wave_spectrum{plot_params.label}.{plot_params.format}",
            bbox_inches="tight",
        )
    else:
        plt.show()

    # Plot the square wave
    def invert(fourier_coeffs, x):
        return (
            fourier_coeffs[0].real
            + 2
            * np.sum(
                fourier_coeffs[1:, np.newaxis]
                * np.exp(1j * wavenumbers[1:, np.newaxis] * x[np.newaxis, :]),
                axis=0,
            ).real
        )

    x = np.linspace(-params.L, params.L, plot_params.grid_size)

    plt.figure()
    plt.plot(x, invert(square_wave_coeffs, x), "k-", label="Square wave")
    plt.plot(x, invert(mol_wave_coeffs, x), "r--", label="Mollified square wave")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f_1(x)$")
    plt.legend()
    plt.tight_layout()

    if plot_params.save:
        plt.savefig(
            plot_params.path + f"square_wave{plot_params.label}.{plot_params.format}",
            bbox_inches="tight",
        )
    else:
        plt.show()
