"""Utility functions for the photosurfactant model."""

from argparse import ArgumentParser

import numpy as np

Y = np.poly1d([1, 0])  # Polynomial for differentiation


def hyperder(n):
    """Return the n-th derivative of cosh."""
    return np.sinh if n % 2 else np.cosh


def cosh(x, x_order=0):
    return hyperder(x_order)(x)


def sinh(x, x_order=0):
    return hyperder(x_order + 1)(x)


def polyder(p: np.poly1d, n: int):
    """Wrap np.polyder and np.polyint."""
    return p.deriv(n) if n > 0 else p.integ(-n)


def to_arr(vals, unknowns):
    """Convert a dictionary of values to an array."""
    arr = np.zeros(len(unknowns), dtype=complex)
    for key in vals.keys():
        if key not in unknowns:
            raise ValueError(f"Unknown key: {key}")

    for i, key in enumerate(unknowns):
        try:
            arr[i] = vals[key]
        except KeyError:
            pass

    return arr


def parameter_parser(parser: ArgumentParser):
    """Add model parameters to the parser."""
    parser.add_argument(
        "--L",
        type=float,
        default=10.0,
        help="The aspect ratio of the domain.",
    )
    parser.add_argument(
        "--Da_tr",
        type=float,
        default=1.0,
        help="The Damkohler number for the trans surfactant.",
    )
    parser.add_argument(
        "--Da_ci",
        type=float,
        default=2.0,
        help="The Damkohler number for the cis surfactant.",
    )
    parser.add_argument(
        "--Pe_tr",
        type=float,
        default=10.0,
        help="The Peclet number for the trans surfactant.",
    )
    parser.add_argument(
        "--Pe_ci",
        type=float,
        default=10.0,
        help="The Peclet number for the cis surfactant.",
    )
    parser.add_argument(
        "--Pe_tr_s",
        type=float,
        default=10.0,
        help="The Peclet number for the trans surfactant on the interface.",
    )
    parser.add_argument(
        "--Pe_ci_s",
        type=float,
        default=10.0,
        help="The Peclet number for the cis surfactant on the interface.",
    )
    parser.add_argument(
        "--Bi_tr",
        type=float,
        default=1 / 300,
        help="The Biot number for the trans surfactant.",
    )
    parser.add_argument(
        "--Bi_ci",
        type=float,
        default=1.0,
        help="The Biot number for the cis surfactant.",
    )
    parser.add_argument("--Ma", type=float, default=2.0, help="The Marangoni number.")
    parser.add_argument(
        "--k_tr",
        type=float,
        default=1.0,
        help="The adsorption rate for the trans surfactant.",
    )
    parser.add_argument(
        "--k_ci",
        type=float,
        default=1 / 30,
        help="The adsorption rate for the cis surfactant.",
    )
    parser.add_argument(
        "--chi_tr",
        type=float,
        default=100 / 30,
        help="The desorption rate for the trans surfactant.",
    )
    parser.add_argument(
        "--chi_ci",
        type=float,
        default=100.0,
        help="The desorption rate for the cis surfactant.",
    )


def plot_parser(parser: ArgumentParser):
    """Add plotting parameters to the parser."""
    parser.add_argument(
        "--wave_count", type=int, default=100, help="Number of wavenumbers to use."
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=1000,
        help="Number of grid points to evaluate the solution on.",
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
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for rendering text."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Format to save the figures in.",
    )


def leading_order_parser(parser: ArgumentParser):
    """Add leading order parameters to the parser."""
    parser.add_argument(
        "--root_index",
        type=int,
        default=-1,
        help="The index of solution branch for the leading order problem. If set to "
        "-1, the branch is selected automatically.",
    )


def first_order_parser(parser: ArgumentParser):
    """Add first order parameters to the parser."""
    parser.add_argument(
        "--func",
        type=str,
        default="super_gaussian(x, 4.0, 1.0)",
        help="An expression in the coordinate x for the light intensity/interface "
        'perturbation. The function should be a quoted string. E.g. "sin(x)". The '
        "function must be 2L-periodic and always return a float.",
    )
    parser.add_argument(
        "--problem",
        choices=["forward", "inverse"],
        default="forward",
        help="The type of problem to solve.",
    )
    parser.add_argument(
        "--mollify",
        action="store_true",
        help="Apply mollification to the light intensity/interface perturbation.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="The mollification parameter for the light intensity/interface "
        "perturbation.",
    )
    parser.add_argument(
        "--norm_scale",
        choices=["linear", "log"],
        default="linear",
        help="The normalization type.",
    )
