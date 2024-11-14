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

    # Plot bulk concentrations
    yy = np.linspace(0, 1, plot_params.grid_size)

    plt.figure(figsize=(8, 6))
    plt.plot(yy, leading.c_tr(yy), "k-", label=r"$c_{\mathrm{tr}, 0}$")
    plt.plot(yy, leading.c_ci(yy), "k--", label=r"$c_{\mathrm{ci}, 0}$")
    plt.xlabel(r"$y$")
    plt.ylabel("Concentration")
    plt.legend()
    plt.tight_layout()

    if plot_params.save:
        plt.savefig(
            plot_params.path + f"leading_bulk_concentrations{plot_params.label}.png",
            dpi=300,
        )
    else:
        plt.show()
