"""A module for the Parameters class."""

from dataclasses import dataclass
import inspect
import numpy as np


@dataclass
class Parameters:
    """The Parameters for the model.

    :param L: The aspect ratio of the domain.
    :param Dam_tr: The Damkohler number for the trans surfactant.
    :param Dam_ci: The Damkohler number for the cis surfactant.
    :param Pen_tr: The Peclet number for the trans surfactant.
    :param Pen_ci: The Peclet number for the cis surfactant.
    :param Pen_tr_s: The Peclet number for the trans surfactant on the
        interface.
    :param Pen_ci_s: The Peclet number for the cis surfactant on the interface.
    :param Bit_tr: The Biot number for the trans surfactant.
    :param Bit_ci: The Biot number for the cis surfactant.
    :param Man: The Marangoni number.
    :param k_tr: The adsorption rate for the trans surfactant.
    :param k_ci: The adsorption rate for the cis surfactant.
    :param chi_tr: The desorption rate for the trans surfactant.
    :param chi_ci: The desorption rate for the cis surfactant.
    """

    # Aspect ratio
    L: float = 10.0

    # Damkohler numbers
    Dam_tr: float = 1.0
    Dam_ci: float = 2.0

    # Peclet numbers
    Pen_tr: float = 10.0
    Pen_ci: float = 10.0

    # Interfacial Peclet numbers
    Pen_tr_s: float = 10.0
    Pen_ci_s: float = 10.0

    # Biot numbers
    Bit_tr: float = 1 / 300
    Bit_ci: float = 1.0

    # Marangoni number
    Man: float = 2.0

    # Adsorption and desorption rates
    k_tr: float = 1.0
    k_ci: float = 1 / 30

    chi_tr: float = 100 / 30
    chi_ci: float = 100.0

    def __post_init__(self):  # noqa: D105
        if self.k_tr * self.chi_tr != self.k_ci * self.chi_ci:
            raise ValueError(
                "Adsorption rates do not satisfy the condition " "k * chi = const."
            )

        self.alpha = self.Dam_ci / self.Dam_tr
        self.eta = self.Pen_tr / self.Pen_ci
        self.zeta = self.Pen_tr * self.Dam_tr + self.Pen_ci * self.Dam_ci

        self.beta = self.Bit_tr / self.Bit_ci
        self.kappa = self.Dam_tr / self.Bit_tr
        self.xi = self.Dam_tr * self.Bit_ci + self.Dam_ci * self.Bit_tr

        self.D = np.array([[self.Dam_tr, -self.Dam_ci], [-self.Dam_tr, self.Dam_ci]])
        self.P = np.array([[self.Pen_tr, 0.0], [0.0, self.Pen_ci]])
        self.P_s = np.array([[self.Pen_tr_s, 0.0], [0.0, self.Pen_ci_s]])
        self.B = np.array([[self.Bit_tr, 0.0], [0.0, self.Bit_ci]])
        self.K = np.array([[self.k_tr, 0.0], [0.0, self.k_ci]])

        self.A = self.P @ self.D
        self.A_s = self.P_s @ self.D

        self.V = np.array([[self.alpha, self.eta], [1.0, -1.0]])
        self.Lambda = np.array([[0.0, 0.0], [0.0, self.zeta]])

    @classmethod
    def from_dict(cls, kwargs):
        return cls(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class PlottingParameters:
    """Additional parameters for plotting.

    :param wave_count: The number of wave numbers to use.
    :param grid_size: The number of grid points to evaluate the solution on.
    :param mollify: A flag to mollify the input function.
    :param delta: The mollification parameter.
    :param save: A flag to save the figures to disk.
    :param path: The path to save the figures to.
    :param label: A label to append to the figure filenames.
    :param format: The format to save the figures in.
    """

    wave_count: int = 100
    grid_size: int = 1000
    mollify: bool = False
    delta: float = 0.5
    save: bool = False
    path: str = "./"
    label: str = ""
    usetex: bool = False
    format: str = "png"

    def __post_init__(self):  # noqa: D105
        self.label = "_" + self.label if self.label else ""
        self.plot_setup()

    @classmethod
    def from_dict(cls, kwargs):
        return cls(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def plot_setup(self):
        """Set up the matplotlib rcParams."""
        import matplotlib.pyplot as plt

        rcparams = {
            "font.size": 16,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": [8, 6],
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
                }
                | rcparams
            )

        else:
            plt.rcParams.update(rcparams)

        plt.close("all")
        self.plt = plt
