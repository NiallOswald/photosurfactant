#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from matplotlib import colors

from photosurfactant.semi_analytic.leading_order import LeadingOrder
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.semi_analytic.limits import HighIntensity, LowIntensity
from photosurfactant.utils.arg_parser import (
    leading_order_parser,
    parameter_parser,
    plot_parser,
)

# TODO: Reduced figures at 3 magnitudes?


def plot_bulk_concentration(
    intensities, default_params, plot_params, root_index, log=False
):
    """Plot the bulk concentrations for varying intensity."""

    default_intensity = np.sqrt(default_params.Da_tr**2 + default_params.Da_ci**2)

    # Figure setup
    plt = plot_params.plt
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)

    cmap = plt.cm.get_cmap("coolwarm")

    if log:
        norm = colors.LogNorm(
            vmin=intensities[0] / np.sqrt(10), vmax=intensities[-1] * np.sqrt(10)
        )
    else:
        norm = plt.Normalize(vmin=intensities[0], vmax=intensities[-1])

    yy = np.linspace(0, 1, plot_params.grid_size)

    # Plot concentrations for varying intensity
    for intensity in intensities:
        params = default_params.update(
            Da_tr=default_params.Da_tr * intensity / default_intensity,
            Da_ci=default_params.Da_ci * intensity / default_intensity,
        )

        # Solve leading order problem
        leading = LeadingOrder(params, root_index)

        # Plot bulk concentrations
        axs[0].plot(yy, leading.c_tr(yy), color=cmap(norm(intensity)))
        axs[1].plot(yy, leading.c_ci(yy), color=cmap(norm(intensity)))

    # Highlight curves for unit intensity
    params = default_params.copy()
    leading = LeadingOrder(params, root_index)
    axs[0].plot(yy, leading.c_tr(yy), color="k", linestyle=(0, (5, 10)))
    axs[1].plot(yy, leading.c_ci(yy), color="k", linestyle=(0, (5, 10)))

    # Tidy up figure
    axs[0].set_xticklabels([])
    axs[0].set_ylabel(r"$c_{\mathrm{tr}, 0}$")
    axs[0].ticklabel_format(style="sci", axis="z", scilimits=(0, 0))

    axs[1].set_xlabel(r"$z$")
    axs[1].set_ylabel(r"$c_{\mathrm{ci}, 0}$")
    axs[1].ticklabel_format(style="sci", axis="z", scilimits=(0, 0))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), label=r"Intensity ($Da$)")

    # Annotate colorbar
    cbar.ax.scatter(
        np.mean(cbar.ax.set_xlim()) * np.ones_like(intensities),
        intensities,
        s=50,
        facecolors="none",
        edgecolors="k",
    )

    cbar.ax.scatter(
        [np.mean(cbar.ax.set_xlim())],
        [default_intensity],
        s=60,
        facecolors="k",
        edgecolors="k",
        marker="D",
    )

    if plot_params.save:
        plt.savefig(
            plot_params.path
            + f"bulk_concentration{plot_params.label}.{plot_params.format}",
            bbox_inches="tight",
        )
    else:
        plt.show()


def plot_interfacial_concentration(
    intensities, default_params, plot_params, root_index, log=False, limits=False
):
    """Plot surface excess for varying intensity."""
    # Collect interfacial values for varying intensity
    gamma_tr, gamma_ci = [], []
    tension = []
    default_intensity = np.sqrt(default_params.Da_tr**2 + default_params.Da_ci**2)

    for intensity in intensities:
        params = default_params.update(
            Da_tr=default_params.Da_tr * intensity / default_intensity,
            Da_ci=default_params.Da_ci * intensity / default_intensity,
        )

        # Solve leading order problem
        leading = LeadingOrder(params, root_index)

        # Store values
        gamma_tr.append(leading.Gamma_tr)
        gamma_ci.append(leading.Gamma_ci)

        tension.append(leading.gamma)

    gamma_tr, gamma_ci = np.array(gamma_tr), np.array(gamma_ci)
    tension = np.array(tension)

    plt = plot_params.plt
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot surface excess
    gamma_tr_plt = ax1.plot(
        intensities, gamma_tr, "r--", label=r"$\Gamma_{\mathrm{tr}, 0}$"
    )
    gamma_ci_plt = ax1.plot(
        intensities, gamma_ci, "b-.", label=r"$\Gamma_{\mathrm{ci}, 0}$"
    )

    # Plot limiting cases
    if limits:
        tan_length = int((3 / 8) * len(intensities))

        small_dam = LowIntensity(params)
        ax1.plot(
            intensities[:tan_length],
            small_dam.gamma_tr * np.ones_like(intensities[:tan_length]),
            "r:",
        )
        ax1.plot(
            intensities[:tan_length],
            small_dam.gamma_ci * np.ones_like(intensities[:tan_length]),
            "b:",
        )

        large_dam = HighIntensity(params)
        ax1.plot(
            intensities[-tan_length:],
            large_dam.gamma_tr * np.ones_like(intensities[-tan_length:]),
            "r:",
        )
        ax1.plot(
            intensities[-tan_length:],
            large_dam.gamma_ci * np.ones_like(intensities[-tan_length:]),
            "b:",
        )

    # Set xscale
    if log:
        ax1.set_xscale("log")

    ax1.set_xlabel(r"Intensity ($Da$)")
    ax1.set_ylabel(
        r"Surface Excess ($\Gamma_{\mathrm{tr}, 0}, \Gamma_{\mathrm{ci}, 0}$)"
    )
    ax1.ticklabel_format(style="sci", axis="z", scilimits=(0, 0))
    ax1.grid()

    ax2 = ax1.twinx()

    # Plot surface tension
    tension_plt = ax2.plot(
        intensities,
        tension,
        "k-",
        label=r"$\gamma_0$",
    )
    ax2.set_ylabel(r"$\gamma_0$")
    ax2.ticklabel_format(style="sci", axis="z", scilimits=(0, 0))

    # Annotate points at unit intensity
    leading = LeadingOrder(default_params, root_index)
    ax1.plot([default_intensity], [leading.Gamma_tr], color="r", marker="D")
    ax1.plot([default_intensity], [leading.Gamma_ci], color="b", marker="D")
    ax2.plot([default_intensity], [leading.gamma], color="k", marker="D")

    # Merge legends
    plots = tension_plt + gamma_tr_plt + gamma_ci_plt
    labels = [plot.get_label() for plot in plots]
    ax2.legend(plots, labels, loc="upper right")

    if plot_params.save:
        plt.savefig(
            plot_params.path
            + f"interfacial_concentration{plot_params.label}.{plot_params.format}",
            bbox_inches="tight",
        )
    else:
        plt.show()


def plot_leading_order():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the leading order surfactant concentrations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    plot_parser(parser)
    leading_order_parser(parser)
    parser.add_argument(
        "--intensities",
        type=float,
        nargs="*",
        default=[1.0],
        help="Intensity factors to plot the concentration for. If count is greater "
        "than one, plots for a range instead.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="The number of values to plot if using an intensity range.",
    )
    parser.add_argument(
        "--interface",
        action="store_true",
        help="Plot the interfacial values.",
    )
    parser.add_argument(
        "--limits",
        action="store_true",
        help="Plot the equilibrium states for small and large intensity.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a log scale.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="The stretching rate when using a range.",
    )
    args = parser.parse_args()

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))
    root_index = args.root_index
    intensities = args.intensities

    if args.count > 1:
        if len(intensities) != 2:
            raise ValueError("intensities must be of length 2 to plot a range.")

        grid = abs(np.linspace(-1.0, 1.0, args.count)) ** args.gamma
        grid[: args.count // 2] *= -1
        intensities = grid * (intensities[1] - intensities[0]) / 2 + np.mean(
            intensities
        )

    if args.log:
        intensities = 10.0 ** np.array(intensities)

    if args.interface:
        plot_interfacial_concentration(
            intensities,
            params,
            plot_params,
            root_index,
            log=args.log,
            limits=args.limits,
        )
    else:
        plot_bulk_concentration(
            intensities, params, plot_params, root_index, log=args.log
        )
