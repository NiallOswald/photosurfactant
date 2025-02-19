#! /usr/bin/env python
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.utils import parameter_parser, plot_parser, leading_order_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np


def plot_leading_order():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the leading order surfactant concentrations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    plot_parser(parser)
    leading_order_parser(parser)
    args = parser.parse_args()

    root_index = args.root_index

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))

    # Solve leading order problem
    leading = LeadingOrder(params, root_index)

    # Print surface excess concentrations
    print("Gamma_tr:", leading.gamma_tr)
    print("Gamma_ci:", leading.gamma_ci)

    # Figure setup
    plt = plot_params.plt
    yy = np.linspace(0, 1, plot_params.grid_size)

    # Plot concentration profiles
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(yy, leading.c_tr(yy), "k-")
    ax[0].set_ylabel(r"$c_{\mathrm{tr}, 0}$")
    # ax[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax[1].plot(yy, leading.c_ci(yy), "k-")
    ax[1].set_xlabel(r"$y$")
    ax[1].set_ylabel(r"$c_{\mathrm{ci}, 0}$")
    # ax[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()

    if plot_params.save:
        plt.savefig(
            plot_params.path
            + f"leading_concentrations{plot_params.label}.{plot_params.format}",
        )
    else:
        plt.show()
