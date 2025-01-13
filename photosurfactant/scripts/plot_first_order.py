#! /usr/bin/env python
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.leading_order import LeadingOrder
from photosurfactant.first_order import FirstOrder
from photosurfactant.fourier import fourier_series_coeff, convolution_coeff
from photosurfactant.functions import *  # noqa: F401, F403
from photosurfactant.utils import (
    parameter_parser,
    plot_parser,
    leading_order_parser,
    first_order_parser,
)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from math import *  # noqa: F401, F403
from matplotlib import colors


class Figures:
    """Figures for the first order problem."""

    def __init__(self, first: FirstOrder, plot_params: PlottingParameters):
        """Initialize the figures class."""
        self.first = first
        self.plot_params = plot_params

        self.params = first.params
        self.leading = first.leading

        self.plt = plot_params.plt

        self._initialize()

    def _initialize(self):
        self.xx = np.linspace(
            -self.first.params.L, self.first.params.L, self.plot_params.grid_size
        )
        self.yy = np.linspace(0, 1, self.plot_params.grid_size)

        self.ggamma_tr = self.first.gamma_tr(self.xx)
        self.ggamma_ci = self.first.gamma_ci(self.xx)
        self.ttension = (
            -self.params.Man
            * (self.ggamma_tr + self.ggamma_ci)
            / (1 - self.leading.gamma_tot)
        )
        self.JJ_tr = self.first.J_tr(self.xx)
        self.JJ_ci = self.first.J_ci(self.xx)
        self.SS_inv = self.first.S_inv(self.xx)
        self.ff_inv = self.first.f_inv(self.xx)

        self.psii = np.array([self.first.psi(self.xx, y) for y in self.yy])
        self.uu = np.array([self.first.u(self.xx, y) for y in self.yy])
        self.vv = np.array([self.first.v(self.xx, y) for y in self.yy])
        self.cc_tr = np.array([self.first.c_tr(self.xx, y) for y in self.yy])
        self.cc_ci = np.array([self.first.c_ci(self.xx, y) for y in self.yy])

        self.direction = self.first.direction
        self.label = self.plot_params.label
        self.format = self.plot_params.format

    def export_data(self, path: str):
        """Export data to a .csv file."""
        np.savetxt(path + "xx.csv", self.xx.real, delimiter=",")
        np.savetxt(path + "yy.csv", self.yy.real, delimiter=",")
        np.savetxt(path + "ggamma_tr.csv", self.ggamma_tr.real, delimiter=",")
        np.savetxt(path + "ggamma_ci.csv", self.ggamma_ci.real, delimiter=",")
        np.savetxt(path + "ttension.csv", self.ttension.real, delimiter=",")
        np.savetxt(path + "JJ_tr.csv", self.JJ_tr.real, delimiter=",")
        np.savetxt(path + "JJ_ci.csv", self.JJ_ci.real, delimiter=",")
        np.savetxt(path + "SS_inv.csv", self.SS_inv.real, delimiter=",")
        np.savetxt(path + "ff_inv.csv", self.ff_inv.real, delimiter=",")
        np.savetxt(path + "psii.csv", self.psii, delimiter=",")
        np.savetxt(path + "uu.csv", self.uu.real, delimiter=",")
        np.savetxt(path + "vv.csv", self.vv.real, delimiter=",")
        np.savetxt(path + "cc_tr.csv", self.cc_tr.real, delimiter=",")
        np.savetxt(path + "cc_ci.csv", self.cc_ci.real, delimiter=",")

    def plot_interfacial_velocity(self):
        """Plot the interfacial velocity."""
        self.plt.figure()
        self.plt.plot(self.xx, self.uu[-1, :].real, "k-", label=r"$u_1$")
        self.plt.plot(self.xx, self.vv[-1, :].real, "k--", label=r"$v_1$")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel("Interfacial Velocity")
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.legend(loc="upper right")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_interfacial_velocity{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_streamlines(self):
        """Plot the streamlines."""
        self.plt.figure(figsize=(12, 4))
        self.plt.contour(self.xx, self.yy, self.psii.real, levels=15, colors="black")
        self.plt.title("Streamlines")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$y$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_streamlines{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_velocity(self):
        """Plot the velocity field."""
        step = self.plot_params.grid_size // 20

        self.plt.figure(figsize=(12, 4))
        self.plt.quiver(
            self.xx[::step],
            self.yy[::step],
            self.uu[::step, ::step].real,
            self.vv[::step, ::step].real,
        )
        self.plt.imshow(
            np.sqrt(self.uu[::-1, :].real ** 2 + self.vv[::-1, :].real ** 2),
            extent=[-self.params.L, self.params.L, 0, 1],
            aspect="auto",
            cmap="viridis",
        )
        self.plt.colorbar()
        self.plt.title("Velocity")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$y$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_velocity{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_concentration_tr(self):
        """Plot the concentration field of the trans surfactant."""
        self.plt.figure(figsize=(12, 4))
        self.plt.imshow(
            self.cc_tr[::-1, :].real,
            extent=[-self.params.L, self.params.L, 0, 1],
            aspect="auto",
            cmap="coolwarm",
            norm=colors.CenteredNorm(),
        )
        self.plt.colorbar()
        self.plt.title(r"$c_{\mathrm{tr}, 1}$")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$y$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_concentration_tr{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_concentration_ci(self):
        """Plot the concentration field of the cis surfactant."""
        self.plt.figure(figsize=(12, 4))
        self.plt.imshow(
            self.cc_ci[::-1, :].real,
            extent=[-self.params.L, self.params.L, 0, 1],
            aspect="auto",
            cmap="coolwarm",
            norm=colors.CenteredNorm(),
        )
        self.plt.colorbar()
        self.plt.title(r"$c_{\mathrm{ci}, 1}$")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$y$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_concentration_ci{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_concentration_tot(self):
        """Plot the total surfactant concentration field."""
        self.plt.figure(figsize=(12, 4))
        self.plt.imshow(
            self.cc_ci[::-1, :].real + self.cc_tr[::-1, :].real,
            extent=[-self.params.L, self.params.L, 0, 1],
            aspect="auto",
            cmap="coolwarm",
            norm=colors.CenteredNorm(),
        )
        self.plt.colorbar()
        self.plt.title(r"$c_{\mathrm{tot}, 1}$")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$y$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_concentration_tot{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_surface_excess(self):
        """Plot the surface excess concentrations."""
        fig, ax1 = self.plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(self.xx, self.ff_inv.real, "k-", label=r"$f_1$")

        ax2.plot(
            self.xx, self.ggamma_tr.real, "k--", label=r"$\Gamma_{\mathrm{tr}, 1}$"
        )
        ax2.plot(
            self.xx, self.ggamma_ci.real, "k-.", label=r"$\Gamma_{\mathrm{ci}, 1}$"
        )

        packing = 1.1
        ylim_1 = max(abs(self.ff_inv)) * packing

        ax1.set_ylim(-ylim_1, ylim_1)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"Light intensity ($f_1$)")

        ylim_2 = max(max(abs(self.ggamma_tr)), max(abs(self.ggamma_ci))) * packing

        ax2.set_ylim(-ylim_2, ylim_2)
        ax2.set_ylabel(
            r"Surface excess concentrations ($\Gamma_{\mathrm{tr}, 1}$, "
            r"$\Gamma_{\mathrm{ci}, 1}$)"
        )

        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        fig.legend(loc="upper left", bbox_to_anchor=(0.79, 0.94))
        fig.tight_layout()

        if self.plot_params.save:
            fig.savefig(
                self.plot_params.path
                + f"{self.direction}_surface_excess{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_interface(self):
        """Plot the surface shape."""
        fig, ax1 = self.plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(self.xx, self.ff_inv.real, "k-", label=r"$f_1$")
        ax2.plot(self.xx, self.SS_inv.real, "k:", label=r"$S_1$")

        packing = 1.1
        ylim_1 = max(abs(self.ff_inv)) * packing

        ax1.set_ylim(-ylim_1, ylim_1)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"Light intensity ($f_1$)")

        ylim_2 = max(abs(self.SS_inv)) * packing

        ax2.set_ylim(-ylim_2, ylim_2)
        ax2.set_ylabel(r"Surface shape ($S_1$)")

        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        fig.legend(loc="upper left", bbox_to_anchor=(0.82, 0.93))
        fig.tight_layout()

        if self.plot_params.save:
            fig.savefig(
                self.plot_params.path
                + f"{self.direction}_interface{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_surface_tension(self):
        """Plot the surface tension."""
        self.plt.figure()
        self.plt.plot(self.xx, self.ttension.real, "k-")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$\gamma_1$")
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_tension{self.label}.{self.format}",
            )
        else:
            self.plt.show()

    def plot_fluxes(self):
        """Plot the fluxes."""
        self.plt.figure()
        self.plt.plot(self.xx, self.JJ_tr.real, "k-", label=r"$J_{\mathrm{tr}, 1}$")
        self.plt.plot(self.xx, self.JJ_ci.real, "k--", label=r"$J_{\mathrm{ci}, 1}$")
        self.plt.plot(
            self.xx,
            self.JJ_tr.real + self.JJ_ci.real,
            "k:",
            label=r"$J_{\mathrm{tr}, 1} + J_{\mathrm{ci}, 1}$",
        )
        self.plt.xlabel(r"$x$")
        self.plt.ylabel("Kinetic Flux")
        self.plt.legend(loc="upper right")
        self.plt.grid()
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"{self.direction}_flux{self.label}.{self.format}",
            )
        else:
            self.plt.show()


def plot_first_order():  # noqa: D103
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
    func = eval("lambda x, L: " + args.func)
    problem = args.problem

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))

    # Calculate Fourier series coefficients
    if args.mollify:
        mol = mollifier(delta=args.delta)  # noqa: F405
        omega, func_coeffs = convolution_coeff(
            lambda x: func(x, params.L),
            mol,
            params.L,
            plot_params.wave_count,
        )
    else:
        omega, func_coeffs = fourier_series_coeff(
            lambda x: func(x, params.L), params.L, plot_params.wave_count
        )

    # Solve leading order problem
    leading = LeadingOrder(params, root_index)

    # Solve first order problem
    first = FirstOrder(omega, func_coeffs, params, leading, direction=problem)

    # Plot figures
    figures = Figures(first, plot_params)

    figures.plot_streamlines()
    figures.plot_velocity()
    figures.plot_concentration_tr()
    figures.plot_concentration_ci()
    figures.plot_concentration_tot()
    figures.plot_interface()
    figures.plot_surface_excess()
    figures.plot_surface_tension()
    figures.plot_fluxes()
    figures.plot_interfacial_velocity()
