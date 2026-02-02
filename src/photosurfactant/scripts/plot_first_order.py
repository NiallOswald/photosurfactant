#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from matplotlib import colors

from photosurfactant.fourier import convolution_coeff, fourier_series_coeff
from photosurfactant.intensity_functions import *  # noqa: F401, F403
from photosurfactant.intensity_functions import mollifier
from photosurfactant.parameters import Parameters, PlottingParameters
from photosurfactant.semi_analytic.first_order import FirstOrder, Variables
from photosurfactant.semi_analytic.leading_order import LeadingOrder
from photosurfactant.utils.arg_parser import (
    first_order_parser,
    leading_order_parser,
    parameter_parser,
    plot_parser,
)
from photosurfactant.utils.func_parser import parse_func


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

        self.ggamma_tr = self.first.Gamma_tr(self.xx)
        self.ggamma_ci = self.first.Gamma_ci(self.xx)
        self.ttension = (
            -self.params.Ma
            * (self.ggamma_tr + self.ggamma_ci)
            / (1 - self.leading.Gamma_tr - self.leading.Gamma_ci)
        )
        self.JJ_tr = self.first.J_tr(self.xx)
        self.JJ_ci = self.first.J_ci(self.xx)
        self.SS = self.first.S(self.xx)
        self.ff = self.first.f(self.xx)

        self.psii = np.array([self.first.psi(self.xx, y) for y in self.yy])
        self.ppressure = np.array([self.first.p(self.xx, y) for y in self.yy])
        self.uu = np.array([self.first.u(self.xx, y) for y in self.yy])
        self.vv = np.array([self.first.w(self.xx, y) for y in self.yy])
        self.cc_tr = np.array([self.first.c_tr(self.xx, y) for y in self.yy])
        self.cc_ci = np.array([self.first.c_ci(self.xx, y) for y in self.yy])

        self.label = self.plot_params.label
        self.format = self.plot_params.format

    def export_data(self, path: str):
        """Export data to a .csv file."""
        np.savetxt(path + "xx.csv", self.xx, delimiter=",")
        np.savetxt(path + "yy.csv", self.yy, delimiter=",")
        np.savetxt(path + "ggamma_tr.csv", self.ggamma_tr, delimiter=",")
        np.savetxt(path + "ggamma_ci.csv", self.ggamma_ci, delimiter=",")
        np.savetxt(path + "ttension.csv", self.ttension, delimiter=",")
        np.savetxt(path + "JJ_tr.csv", self.JJ_tr, delimiter=",")
        np.savetxt(path + "JJ_ci.csv", self.JJ_ci, delimiter=",")
        np.savetxt(path + "SS.csv", self.SS, delimiter=",")
        np.savetxt(path + "ff.csv", self.ff, delimiter=",")
        np.savetxt(path + "psii.csv", self.psii, delimiter=",")
        np.savetxt(path + "ppressure.csv", self.ppressure, delimiter=",")
        np.savetxt(path + "uu.csv", self.uu, delimiter=",")
        np.savetxt(path + "vv.csv", self.vv, delimiter=",")
        np.savetxt(path + "cc_tr.csv", self.cc_tr, delimiter=",")
        np.savetxt(path + "cc_ci.csv", self.cc_ci, delimiter=",")

    def plot_interfacial_velocity(self):
        """Plot the interfacial velocity."""
        self.plt.figure(figsize=(6, 5))
        self.plt.plot(self.xx, self.uu[-1, :], "k-")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"Interfacial Velocity ($u_1$)")
        self.plt.grid()
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"interfacial_velocity{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_streamplot(self):
        """Plot the streamlines and velocity field."""
        self.plt.figure(figsize=(12, 4))
        self.plt.streamplot(self.xx, self.yy, self.uu, self.vv, color="black")
        self.plt.imshow(
            np.sqrt(self.uu**2 + self.vv**2),
            extent=[-self.params.L, self.params.L, 0, 1],
            origin="lower",
            aspect="auto",
            cmap="Reds",
        )
        cbar = self.plt.colorbar(label=r"$\lVert \mathbf{u}_1 \rVert$")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$z$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"streamplot{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_streamlines(self):
        """Plot the streamlines and velocity field."""
        self.plt.figure(figsize=(12, 4))
        self.plt.contour(self.xx, self.yy, self.psii, colors="black")
        self.plt.imshow(
            np.sqrt(self.uu**2 + self.vv**2),
            extent=[-self.params.L, self.params.L, 0, 1],
            origin="lower",
            aspect="auto",
            cmap="Reds",
        )
        cbar = self.plt.colorbar(label=r"$\lVert \mathbf{u}_1 \rVert$")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$z$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"streamlines{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_concentration_crop(self, lims):
        """Plot the concentration field."""
        _, axs = self.plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        conc_fields = [self.cc_tr, self.cc_ci, self.cc_tr + self.cc_ci]
        ax_labels = [
            r"$c_{\mathrm{tr}, 1}$",
            r"$c_{\mathrm{ci}, 1}$",
            r"$c_{\mathrm{tot}, 1}$",
        ]

        conc_lim = np.max(np.abs(conc_fields))
        if self.plot_params.norm_scale == "linear":
            norm = colors.CenteredNorm(vcenter=0.0, halfrange=conc_lim)
        elif self.plot_params.norm_scale == "log":
            norm = colors.SymLogNorm(
                linthresh=1e-2 * conc_lim,
                vmin=-conc_lim,
                vmax=conc_lim,
            )
        else:
            raise ValueError("Invalid norm scale.")

        images = []
        for ax, field, label in zip(axs.flat, conc_fields, ax_labels):
            images.append(
                ax.imshow(
                    field,
                    extent=[-self.params.L, self.params.L, 0, 1],
                    origin="lower",
                    aspect="auto",
                    cmap="coolwarm",
                    norm=norm,
                )
            )

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$z$")
            ax.set_title(label)
            ax.set_xlim(lims)

        # Hide duplicate labels
        for ax in axs.flat:
            ax.label_outer()

        cbar = self.plt.colorbar(
            images[0], label="Concentration", ax=axs.ravel().tolist()
        )

        if self.plot_params.norm_scale == "linear":
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"concentration{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def _plot_field(self, field, label, prefix=""):
        """Plot the field of a given scalar."""
        if self.plot_params.norm_scale == "linear":
            norm = colors.CenteredNorm()
        elif self.plot_params.norm_scale == "log":
            conc_lim = np.max(np.abs(field))
            norm = colors.SymLogNorm(
                linthresh=1e-2 * conc_lim,
                vmin=-conc_lim,
                vmax=conc_lim,
            )
        else:
            raise ValueError("Invalid norm scale.")

        self.plt.figure(figsize=(12, 4))
        self.plt.imshow(
            field,
            extent=[-self.params.L, self.params.L, 0, 1],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            norm=norm,
        )

        cbar = self.plt.colorbar(label=label)

        if self.plot_params.norm_scale == "linear":
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)

        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$z$")
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + prefix + f"{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_pressure(self):
        """Plot the pressure field."""
        self._plot_field(self.ppressure, r"$p_1$", "pressure")

    def plot_concentration_tr(self):
        """Plot the concentration field of the trans surfactant."""
        self._plot_field(self.cc_tr, r"$c_{\mathrm{tr}, 1}$", "concentration_tr")

    def plot_concentration_ci(self):
        """Plot the concentration field of the cis surfactant."""
        self._plot_field(self.cc_ci, r"$c_{\mathrm{ci}, 1}$", "concentration_ci")

    def plot_concentration_tot(self):
        """Plot the total surfactant concentration field."""
        self._plot_field(
            self.cc_tr + self.cc_ci,
            r"$c_{\mathrm{tot}, 1}$",
            "concentration_tot",
        )

    def plot_surface_excess(self):
        """Plot the surface excess concentrations."""
        self.plt.figure(figsize=(6, 5))

        self.plt.plot(
            self.xx,
            self.ggamma_tr + self.ggamma_ci,
            "k-",
            label=r"$\Gamma_{\mathrm{tr}, 1} + \Gamma_{\mathrm{ci}, 1}$",
        )
        self.plt.plot(
            self.xx, self.ggamma_tr, "r--", label=r"$\Gamma_{\mathrm{tr}, 1}$"
        )
        self.plt.plot(
            self.xx, self.ggamma_ci, "b-.", label=r"$\Gamma_{\mathrm{ci}, 1}$"
        )

        self.plt.xlabel(r"$x$")
        self.plt.ylabel("Surface Excess")
        self.plt.legend(loc="lower right")
        self.plt.grid()
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"surface_excess{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_intensity(self):
        """Plot the light intensity."""
        self.plt.figure(figsize=(6, 5))

        self.plt.plot(self.xx, self.ff, "k-", label="Approx")

        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$f_1$")
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"intensity{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_intensity_slip_tension(self):
        """Plot the light intensity, interfacial slip velocity, and surface tension."""
        _, ax1 = self.plt.subplots(figsize=(10, 4.5))

        slip_plt = ax1.plot(self.xx, self.uu[-1, :], "k-", label=r"$u_1$")
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$u_1$")
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        ax2 = ax1.twinx()

        tension_plt = ax2.plot(self.xx, self.ttension, "k--", label=r"$\gamma_1$")
        ax2.set_ylabel(r"$\gamma_1$")
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # Align and stretch axes (must occur before intensity plot)
        offset, scale = 2, 0.5  # Relative location/scale of intensity
        _, limit = ax1.get_ylim()
        ax1.set_ylim(-limit, offset * limit)
        _, limit = ax2.get_ylim()
        ax2.set_ylim(-limit, offset * limit)

        # Plot intensity on first axes scaled by slip velocity
        max_u = np.max(self.uu[-1, :])
        base, max_height = offset * max_u, scale * max_u
        arrow_xx, arrow_ff = self.xx[::20], self.ff[::20]
        ax1.quiver(
            arrow_xx,
            [base + 0.5 * max_height],  # base of arrow
            np.zeros_like(arrow_ff),
            -(0.75 + arrow_ff),  # arrow length (scaled by max_height)
            scale=1 / max_height,
            width=0.005,
            scale_units="y",
            color="b",
        )

        # Fix for having a single legend over multiple axes
        plots = slip_plt + tension_plt
        labels = [plot.get_label() for plot in plots]
        ax1.legend(plots, labels, loc="lower right")

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path
                + f"intensity_slip_tension{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_interface(self):
        """Plot the surface shape."""
        self.plt.figure(figsize=(6, 5))

        self.plt.plot(self.xx, self.SS, "k-", label="Approx")

        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"$S_1$")
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"interface{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_surface_tension(self):
        """Plot the surface tension."""
        self.plt.figure(figsize=(6, 5))
        self.plt.plot(self.xx, self.ttension, "k-")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel(r"Surface Tension ($\gamma_1$)")
        self.plt.grid()
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"tension{self.label}.{self.format}",
                bbox_inches="tight",
            )
        else:
            self.plt.show()

    def plot_fluxes(self):
        """Plot the fluxes."""
        self.plt.figure(figsize=(6, 5))
        self.plt.plot(
            self.xx,
            self.JJ_tr + self.JJ_ci,
            "k-",
            label=r"$J_{\mathrm{tr}, 1} + J_{\mathrm{ci}, 1}$",
        )
        self.plt.plot(self.xx, self.JJ_tr, "r--", label=r"$J_{\mathrm{tr}, 1}$")
        self.plt.plot(self.xx, self.JJ_ci, "b-.", label=r"$J_{\mathrm{ci}, 1}$")
        self.plt.xlabel(r"$x$")
        self.plt.ylabel("Kinetic Flux")
        self.plt.legend(loc="lower right")
        self.plt.grid()
        self.plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.plt.tight_layout()

        if self.plot_params.save:
            self.plt.savefig(
                self.plot_params.path + f"flux{self.label}.{self.format}",
                bbox_inches="tight",
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

    params = Parameters.from_dict(vars(args))
    plot_params = PlottingParameters.from_dict(vars(args))

    root_index = args.root_index
    func = parse_func(args.func)
    problem = args.problem

    # Calculate Fourier series coefficients
    if args.mollify:
        wavenumbers, func_coeffs = convolution_coeff(
            lambda x: func(x, params),
            mollifier(delta=args.delta),  # noqa: F405
            params.L,
            plot_params.wave_count,
        )
    else:
        wavenumbers, func_coeffs = fourier_series_coeff(
            lambda x: func(x, params), params.L, plot_params.wave_count
        )

    # Solve leading order problem
    leading = LeadingOrder(params, root_index)

    # Solve first order problem
    if problem == "forward":
        constraint = lambda n: (  # noqa: E731
            Variables.f,
            func_coeffs[n],
        )
    elif problem == "inverse":
        constraint = lambda n: (  # noqa: E731
            (Variables.f, 0.0) if n == 0 else (Variables.S, func_coeffs[n])
        )

    first = FirstOrder(wavenumbers, params, leading)
    first.solve(constraint)

    # Plot figures
    figures = Figures(first, plot_params)

    figures.plot_streamplot()
    figures.plot_streamlines()
    figures.plot_pressure()
    figures.plot_concentration_crop(lims=[-3.0, 3.0])
    figures.plot_concentration_tr()
    figures.plot_concentration_ci()
    figures.plot_concentration_tot()
    figures.plot_surface_excess()
    figures.plot_surface_tension()
    figures.plot_fluxes()
    figures.plot_interfacial_velocity()
    figures.plot_intensity()
    figures.plot_intensity_slip_tension()
    figures.plot_interface()
