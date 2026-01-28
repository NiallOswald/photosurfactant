#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from photosurfactant.semi_analytic.first_order import FirstOrder, Variables
from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.semi_analytic.leading_order import LeadingOrder
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.utils.arg_parser import (
    first_order_parser,
    parameter_parser,
    plot_parser,
)
from photosurfactant.utils.func_parser import parse_func

from alive_progress import alive_it
from scipy.optimize import minimize_scalar

from typing import Callable
from matplotlib import colors


def find_max(func: Callable[[float], float], bounds: list[float]) -> float:
    """Find the maximum of a functional over a given range."""
    res = minimize_scalar(lambda x: -func(x), bounds=bounds)
    assert res.success, "maximum velocity not found"

    return -res.fun


def plot_multi_sweep(
    intensity: Callable[[float], float],
    x_func: Callable[[Parameters], float],
    k_tr_vals: list[float],
    params_list: list[Parameters],
    plot_params: PlottingParameters,
    xlabel: str,
    tol: float = 1e-3,
    label: str = "",
):
    """Sweep maximum velocity for set of k_tr values."""

    if label:
        label += "_"

    # Figure setup
    plt = plot_params.plt
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)  # (9, 6)

    cmap = plt.get_cmap("viridis")

    norm = colors.LogNorm(
        vmin=min(k_tr_vals) / np.sqrt(10), vmax=max(k_tr_vals) * np.sqrt(10)
    )

    # Plot profiles over multile values of k_tr
    for k_tr in k_tr_vals:
        data = []
        for params in alive_it(params_list, title=f"Sweeping at k_tr = {k_tr}"):
            # Preserve the ratio in k parameters
            params = params.update(k_tr=k_tr, k_ci=params.k_ci * (k_tr / params.k_tr))

            # Find Fourier coeffients of the intensity function
            wavenumbers, func_coeffs = fourier_series_coeff(
                lambda x: intensity(x, params), params.L, plot_params.wave_count
            )

            # Solve leading order problem
            leading = LeadingOrder(params)

            if leading.gamma < tol:
                # TODO: Temp fix for Marangoni sweep
                if label == "ma_":
                    data.pop()

                break

            # Solve first order problem
            first = FirstOrder(wavenumbers, params, leading)
            first.solve(lambda n: (Variables.f, func_coeffs[n]))

            # Reduce using the functionals
            data.append(
                (
                    x_func(params),
                    find_max(lambda x: abs(first.u(x, y=1.0)), [-params.L, params.L]),
                )
            )

        if data:
            x_data, y_data = zip(*data)
            ax.plot(x_data, y_data, color=cmap(norm(k_tr)))

    # Tidy up figure
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\max_{x}\lvert{u_1}\rvert_{y=1}$")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, label=r"$k_\mathrm{tr}$")

    # Annotate colorbar
    cbar.ax.scatter(
        np.mean(cbar.ax.set_xlim()) * np.ones_like(k_tr_vals),
        k_tr_vals,
        s=50,
        facecolors="none",
        edgecolors="k",
    )

    if plot_params.save:
        plt.savefig(
            plot_params.path + f"{label}sweep.{plot_params.format}",
            bbox_inches="tight",
        )
    else:
        plt.show()


def bi_sweep(
    intensity,
    start: float,
    stop: float,
    k_tr_vals: list[float],
    default_params: Parameters,
    plot_params: PlottingParameters,
):
    """Sweep Biot number."""
    biot_tr_vals = 10 ** np.linspace(start, stop, plot_params.grid_size)
    params_list = [
        default_params.update(
            Bi_tr=Bi_tr,
            Bi_ci=default_params.Bi_ci * (Bi_tr / default_params.Bi_tr),
        )
        for Bi_tr in biot_tr_vals
    ]

    print("Sweeping Biot number...")
    plot_multi_sweep(
        intensity,
        lambda p: p.Bi_tr,
        k_tr_vals,
        params_list,
        plot_params,
        xlabel=r"$Bi_{\mathrm{tr}}$",
        label="bi",
    )


def da_sweep(
    intensity,
    start: float,
    stop: float,
    k_tr_vals: list[float],
    default_params: Parameters,
    plot_params: PlottingParameters,
):
    """Sweep Damkohler number."""
    damkohler_tr_vals = 10 ** np.linspace(start, stop, plot_params.grid_size)
    params_list = [
        default_params.update(
            Da_tr=Da_tr,
            Da_ci=default_params.Da_ci * (Da_tr / default_params.Da_tr),
        )
        for Da_tr in damkohler_tr_vals
    ]

    print("Sweeping Damkohler number...")
    plot_multi_sweep(
        intensity,
        lambda p: p.Da_tr,
        k_tr_vals,
        params_list,
        plot_params,
        xlabel=r"$Da_{\mathrm{tr}}$",
        label="da",
    )


def ma_sweep(
    intensity,
    start: float,
    stop: float,
    k_tr_vals: list[float],
    default_params: Parameters,
    plot_params: PlottingParameters,
):
    """Sweep Marangoni number."""
    marangoni_vals = 10 ** np.linspace(start, stop, plot_params.grid_size)
    params_list = [default_params.update(Ma=Ma) for Ma in marangoni_vals]

    print("Sweeping Marangoni number...")
    plot_multi_sweep(
        intensity,
        lambda p: p.Ma,
        k_tr_vals,
        params_list,
        plot_params,
        xlabel="$Ma$",
        label="ma",
    )


def main():
    # Parse parameters
    parser = ArgumentParser(
        description="Plot the maximum interfacial velocity.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    plot_parser(parser)
    first_order_parser(parser)
    args = parser.parse_args()

    default_params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))
    intensity = parse_func(args.func)

    # Set k values
    k_tr_vals = 10.0 ** np.arange(-4, 1)

    # Sweep Biot number
    bi_sweep(intensity, -4, 4, k_tr_vals, default_params, plot_params)

    # Sweep Damkohler number
    da_sweep(intensity, -6, 2, k_tr_vals, default_params, plot_params)

    # Sweep Marangoni number
    ma_sweep(intensity, -4, 6, k_tr_vals, default_params, plot_params)


if __name__ == "__main__":
    main()
