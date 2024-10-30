#! /usr/bin/env python
from photosurfactant.parameters import Parameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.utils import parameter_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt


def plot_leading_order():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the leading order surfactant concentrations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root_index", type=int, default=2, help="Index of the solution branch to use."
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save the figures to disk."
    )
    parser.add_argument(
        "--path", type=str, default="./", help="Path to save the figures to."
    )
    parser.add_argument(
        "--label", type=str, help="Label to append to the figure filenames."
    )

    parameter_parser(parser)
    args = parser.parse_args()
    root_index = args.root_index
    save = args.save
    path = args.path
    label = "_" + args.label if args.label else ""

    params = Parameters.from_dict(vars(args))

    # Solve leading order problem
    leading = LeadingOrder(params, root_index)

    # Print surface excess concentrations
    print("Gamma_tr:", leading.gamma_tr)
    print("Gamma_ci:", leading.gamma_ci)

    # Plot bulk concentrations
    yy = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))
    plt.plot(yy, leading.c_tr(yy), "k-", label=r"$c_{\mathrm{tr}, 0}$")
    plt.plot(yy, leading.c_ci(yy), "k--", label=r"$c_{\mathrm{ci}, 0}$")
    plt.xlabel(r"$y$")
    plt.ylabel("Concentration")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(path + f"leading_bulk_concentrations{label}.png", dpi=300)
    else:
        plt.show()
