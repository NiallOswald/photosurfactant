"""A module for the Parameters class."""

import inspect
from copy import copy
from dataclasses import dataclass, asdict

import numpy as np

from typing import Any


@dataclass(frozen=True)
class Parameters:
    """The Parameters for the model.

    :param L: The aspect ratio of the domain.
    :param Da_tr: The Damkohler number for the trans surfactant.
    :param Da_ci: The Damkohler number for the cis surfactant.
    :param Pe_tr: The Peclet number for the trans surfactant.
    :param Pe_ci: The Peclet number for the cis surfactant.
    :param Pe_tr_s: The Peclet number for the trans surfactant on the
        interface.
    :param Pe_ci_s: The Peclet number for the cis surfactant on the interface.
    :param Bi_tr: The Biot number for the trans surfactant.
    :param Bi_ci: The Biot number for the cis surfactant.
    :param Ma: The Marangoni number.
    :param k_tr: The adsorption rate for the trans surfactant.
    :param k_ci: The adsorption rate for the cis surfactant.
    :param chi_tr: The desorption rate for the trans surfactant.
    :param chi_ci: The desorption rate for the cis surfactant.
    """

    # Aspect ratio
    L: float = 10.0

    # Reynolds numbers
    Re: float = 0.0  # TODO: Deprecate this parameter

    # Damkohler numbers
    Da_tr: float = 1.0
    Da_ci: float = 2.0

    # Peclet numbers
    Pe_tr: float = 10.0
    Pe_ci: float = 10.0

    # Interfacial Peclet numbers
    Pe_tr_s: float = 10.0
    Pe_ci_s: float = 10.0

    # Biot numbers
    Bi_tr: float = 1 / 300
    Bi_ci: float = 1.0

    # Marangoni number
    Ma: float = 2.0

    # Adsorption and desorption rates
    k_tr: float = 1.0
    k_ci: float = 1 / 30

    chi_tr: float = 100 / 30
    chi_ci: float = 100.0

    def __post_init__(self):  # noqa: D105
        if not np.isclose(self.k_tr * self.chi_tr, self.k_ci * self.chi_ci):
            raise ValueError(
                "Adsorption rates do not satisfy the condition k * chi = const."
            )

    def from_dict(kwargs: dict[str, float]) -> "Parameters":
        """Load parameters from a dictionary."""
        return Parameters(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(Parameters).parameters
            }
        )

    def copy(self) -> "Parameters":
        """Return a copy of the `Parameters` object."""
        return copy(self)

    def update(self, **new_kwargs) -> "Parameters":
        """Return a new `Parameter` object with missing values derived."""
        derived_kwargs = asdict(self)
        # new_kwargs overwrite derived_kwargs
        return Parameters(**(derived_kwargs | new_kwargs))

    @property
    def alpha(self) -> float:
        return self.Da_ci / self.Da_tr

    @property
    def eta(self) -> float:
        return self.Pe_tr / self.Pe_ci

    @property
    def zeta(self) -> float:
        return self.Pe_tr * self.Da_tr + self.Pe_ci * self.Da_ci

    @property
    def P(self):
        return np.array([[self.Pe_tr, 0.0], [0.0, self.Pe_ci]])

    @property
    def P_s(self):
        return np.array([[self.Pe_tr_s, 0.0], [0.0, self.Pe_ci_s]])

    @property
    def B(self):
        return np.array([[self.Bi_tr, 0.0], [0.0, self.Bi_ci]])

    @property
    def K(self):
        return np.array([[self.k_tr, 0.0], [0.0, self.k_ci]])

    @property
    def A(self):
        return self.P @ self._D

    @property
    def A_s(self):
        return self.P_s @ self._D

    @property
    def V(self):
        return np.array([[self.alpha, self.eta], [1.0, -1.0]])

    @property
    def Lambda(self):
        return np.array([[0.0, 0.0], [0.0, self.zeta]])

    @property
    def _D(self):
        return np.array([[self.Da_tr, -self.Da_ci], [-self.Da_tr, self.Da_ci]])


@dataclass
class PlottingParameters:
    """Additional parameters for plotting.

    :param wave_count: The number of wave numbers to use.
    :param grid_size: The number of grid points to evaluate the solution on.
    :param mollify: A flag to mollify the input function.
    :param delta: The mollification parameter.
    :param norm_scale: Normalization type. Either "linear" or "log".
    :param save: A flag to save the figures to disk.
    :param path: The path to save the figures to.
    :param label: A label to append to the figure filenames.
    :param format: The format to save the figures in.
    """

    wave_count: int = 100
    grid_size: int = 1000
    mollify: bool = False
    delta: float = 0.5
    norm_scale: str = "linear"
    save: bool = False
    path: str = "./"
    label: str = ""
    usetex: bool = False
    format: str = "png"

    def __post_init__(self):  # noqa: D105
        self.label = "_" + self.label if self.label else ""
        self.plot_setup()

    def from_dict(kwargs: dict[str, Any]) -> "PlottingParameters":
        """Load parameters from a dictionary."""
        return PlottingParameters(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(PlottingParameters).parameters
            }
        )

    def copy(self):
        """Return a copy of the class."""
        raise NotImplementedError
        # TODO: Deprecate this method
        return copy(self)

    def plot_setup(self):
        """Set up the matplotlib rcParams."""
        import matplotlib.pyplot as plt

        rcparams = {
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.formatter.useoffset": True,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.figsize": [7, 6],
            "figure.dpi": 100,
            "figure.autolayout": True,
            "savefig.dpi": 300,
        }

        if self.usetex:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    "axes.formatter.use_mathtext": True,
                    "text.latex.preamble": r"\usepackage{amsmath}",
                }
                | rcparams
            )

        else:
            plt.rcParams.update(rcparams)

        plt.close("all")
        self.plt = plt
