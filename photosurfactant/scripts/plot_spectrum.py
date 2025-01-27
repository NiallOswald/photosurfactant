#! /usr/bin/env python
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.first_order import FirstOrder
from photosurfactant.fourier import fourier_series_coeff, convolution_coeff
from photosurfactant.functions import square_wave, mollifier
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

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))

    omega, func_coeffs = fourier_series_coeff(
        lambda x: 0.0, params.L, plot_params.wave_count
    )
    half_omega = omega[plot_params.wave_count :]

    func_coeffs = np.ones_like(func_coeffs, dtype=np.complex128)

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
    first = FirstOrder(omega, func_coeffs, params, leading)
    S_n = abs(first.S_f())[plot_params.wave_count + 1 :]

    # Calculate the slope of the spectrum on a log plot
    index = S_n > 1e-20
    slope = np.polyfit(half_omega[index], np.log(S_n[index]), 1)

    # Figure setup
    plt = plot_params.plt

    # Plot the spectrum
    plt.figure()
    plt.plot(half_omega, S_n, "k-")
    plt.plot(
        half_omega,
        np.exp(slope[0] * half_omega),
        "k--",
        label=r"$e^{" f"{slope[0]:.2f}" r"k_n}$",
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
        )
    else:
        plt.show()

    # Plot the square wave spectrum
    plt.figure()
    plt.plot(
        half_omega,
        square_wave_coeffs[plot_params.wave_count + 1 :],
        "k-",
        label="Square wave",
    )
    plt.plot(
        half_omega,
        mol_wave_coeffs[plot_params.wave_count + 1 :],
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
        )
    else:
        plt.show()

    # Plot the square wave
    def invert(fourier_coeffs, x):
        return fourier_coeffs[0] + np.sum(
            fourier_coeffs[1:, np.newaxis]
            * np.exp(1j * omega[:, np.newaxis] * x[np.newaxis, :]),
            axis=0,
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
        )
    else:
        plt.show()
