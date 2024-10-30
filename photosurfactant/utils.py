"""Utility functions for the photosurfactant model."""

from argparse import ArgumentParser


def parameter_parser(parser: ArgumentParser):
    """Add model parameters to the parser."""
    parser.add_argument(
        "--L",
        type=float,
        default=10.0,
        help="The aspect ratio of the domain." "domain.",
    )
    parser.add_argument(
        "--Dam_tr",
        type=float,
        default=1.0,
        help="The Damkohler number for the trans surfactant.",
    )
    parser.add_argument(
        "--Dam_ci",
        type=float,
        default=2.0,
        help="The Damkohler number for the cis surfactant.",
    )
    parser.add_argument(
        "--Pen_tr",
        type=float,
        default=10.0,
        help="The Peclet number for the trans surfactant.",
    )
    parser.add_argument(
        "--Pen_ci",
        type=float,
        default=10.0,
        help="The Peclet number for the cis surfactant.",
    )
    parser.add_argument(
        "--Pen_tr_s",
        type=float,
        default=10.0,
        help="The Peclet number for the trans surfactant on the interface.",
    )
    parser.add_argument(
        "--Pen_ci_s",
        type=float,
        default=10.0,
        help="The Peclet number for the cis surfactant on the interface.",
    )
    parser.add_argument(
        "--Bit_tr",
        type=float,
        default=3.33,
        help="The Biot number for the trans surfactant.",
    )
    parser.add_argument(
        "--Bit_ci",
        type=float,
        default=1.0e3,
        help="The Biot number for the cis surfactant.",
    )
    parser.add_argument("--Man", type=float, default=2.0, help="The Marangoni number.")
    parser.add_argument(
        "--k_tr",
        type=float,
        default=30.0,
        help="The adsorption rate for the trans surfactant.",
    )
    parser.add_argument(
        "--k_ci",
        type=float,
        default=1.0,
        help="The adsorption rate for the cis surfactant.",
    )
    parser.add_argument(
        "--chi_tr",
        type=float,
        default=1.0,
        help="The desorption rate for the trans surfactant.",
    )
    parser.add_argument(
        "--chi_ci",
        type=float,
        default=30.0,
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


def leading_order_parser(parser: ArgumentParser):
    """Add leading order parameters to the parser."""
    parser.add_argument(
        "--root_index",
        type=int,
        default=2,
        help="The index of the root for the leading order problem.",
    )


def first_order_parser(parser: ArgumentParser):
    """Add first order parameters to the parser."""
    parser.add_argument(
        "--func",
        type=str,
        default="laser_pointer(delta=0.5)(x)",
        help="An expression in the coordinate x for the light intensity/interface "
        'perturbation. The function should be a quoted string. E.g. "sin(x)". The '
        "function must be L-periodic and always return a float.",
    )
    parser.add_argument(
        "--problem",
        choices=["forward", "inverse"],
        default="forward",
        help="The type of problem to solve.",
    )