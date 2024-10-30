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
