"""Module containing the system for the leading order model."""

import numpy as np
from linear.parameters import *

# Determinant coefficients
a_0 = k_tr * ((Dam_tr + Dam_ci) / Bit_ci + 1)

b_0 = (Dam_tr + Dam_ci) * (k_ci / Bit_tr - eta * k_tr / Bit_ci) + k_ci - eta * k_tr

# Quadratic coefficients
# First Quadratic
a = (a_0 + b_0 / (alpha + eta)) * (alpha + 1) / (alpha + eta)

b = b_0 * (alpha + 1) / (alpha + eta) * np.cosh(np.sqrt(zeta)) + (a_0 + b_0 / (alpha + eta)) * (1 - eta) / np.sqrt(zeta) * np.sinh(np.sqrt(zeta))

c = b_0 * (1 - eta) / np.sqrt(zeta) * np.sinh(np.sqrt(zeta)) * np.cosh(np.sqrt(zeta))

d = (alpha + 1) / (alpha + eta) + (1 / (k_tr * chi_tr) - 1 / (2 * L)) * (a_0 + b_0 / (alpha + eta))

e = (1 - eta) / np.sqrt(zeta) * np.sinh(np.sqrt(zeta)) + b_0 * (1 / (k_tr * chi_tr) - 1 / (2 * L)) * np.cosh(np.sqrt(zeta))

f = -1 / (2 * L)

# Second Quadratic
p = k_ci * chi_ci * b_0 * (alpha + eta) / np.sqrt(zeta) * np.sinh(np.sqrt(zeta)) * np.cosh(np.sqrt(zeta))

q = k_ci * chi_ci * (a_0 + b_0 / (alpha + eta)) * (alpha + eta) / np.sqrt(zeta) * np.sinh(np.sqrt(zeta))

r = k_ci * chi_ci * (alpha + eta) / np.sqrt(zeta) * np.sinh(np.sqrt(zeta)) + (alpha * k_ci + eta * k_tr) * np.cosh(np.sqrt(zeta))

s = alpha / (alpha + eta) * (k_ci - k_tr)

# Solve for B_0
poly = np.poly1d([
    a * p**2 - b * p * q + c * q**2,
    2 * a * p * r - b * p * s - b * q * r + 2 * c * q * s - d * p * q + e * q**2,
    a * r**2 - b * r * s + c * s**2 - d * p * s - d * q * r + 2 * e * q * s  + f * q**2,
    -d * r * s + e * s**2 + 2 * f * q * s,
    f * s**2
])
roots = poly.roots

B_0 = roots[2]  # Select the solution branch as needed

# Solve for A_0
A_0 = -(p * B_0 + r) / (q * B_0 + s) * B_0

# Bulk concentrations
def c_ci_0(y):
    return A_0 / (alpha + eta) + B_0 * np.cosh(y * np.sqrt(zeta))

def c_tr_0(y):
    return A_0 - eta * c_ci_0(y)

# Surface concentrations
Delta = (Dam_tr + Dam_ci) * (k_tr * c_tr_0(1) / Bit_ci + k_ci * c_ci_0(1) / Bit_tr) + k_tr * c_tr_0(1) + k_ci * c_ci_0(1) + 1

gamma_0 = (
    (1 / Delta) * ((k_tr * c_tr_0(1) / Bit_ci + k_ci * c_ci_0(1) / Bit_tr) * np.array([Dam_ci, Dam_tr])
    + np.array([k_tr * c_tr_0(1), k_ci * c_ci_0(1)]))
)

gamma_tr_0 = gamma_0[0]
gamma_ci_0 = gamma_0[1]

# Useful functions for first order
def c_0(y):
    return np.array([c_tr_0(y), c_ci_0(y)])

def d_c_ci_0(y):
    return B_0 * np.sqrt(zeta) * np.cosh(y * np.sqrt(zeta))

def d_c_tr_0(y):
    return -eta * d_c_ci_0(y)

def d_c_0(y):
    return np.array([d_c_tr_0(y), d_c_ci_0(y)])

def d2_c_ci_0(y):
    return B_0 * zeta * np.sinh(y * np.sqrt(zeta))

def d2_c_tr_0(y):
    return -eta * d2_c_ci_0(y)

def d2_c_0(y):
    return np.array([d2_c_tr_0(y), d2_c_ci_0(y)])
