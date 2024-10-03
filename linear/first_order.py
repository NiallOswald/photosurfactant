"""Module containing the system for the first order model."""

import numpy as np
from linear.parameters import (L, Dam_tr, Dam_ci, Pen_tr, Pen_ci, Pen_tr_s, Pen_ci_s, Bit_tr, Bit_ci, Man, k_tr, k_ci, chi_tr, chi_ci, alpha, eta, zeta)
from linear.leading_order import (c_tr_0, c_ci_0, c_0, d_c_0, d2_c_0, gamma_0, A_0, B_0, Delta)

# Define matrices
D = np.array([
    [Dam_tr, -Dam_ci],
    [-Dam_tr, Dam_ci]
])

P = np.array([
    [Pen_tr, 0],
    [0, Pen_ci]
])

P_s = np.array([
    [Pen_tr_s, 0],
    [0, Pen_ci_s]
])

B = np.array([
    [Bit_tr, 0],
    [0, Bit_ci]
])

K = np.array([
    [k_tr, 0],
    [0, k_ci]
])

V = np.array([
    [alpha, eta],
    [1, -1]
])

Lambda = np.array([
    [0, 0],
    [0, zeta]
])

A = P @ D

M = D + B @ np.array([
    [k_tr * c_tr_0(1) + 1, k_tr * c_tr_0(1)],
    [k_ci * c_ci_0(1), k_ci * c_ci_0(1) + 1]
])

A_s = P_s @ D

M_s = P_s @ M

z = np.ones(2).reshape(1, 2)

I = np.eye(2)

# Non-constant mode
## Define embeddings
unknowns = ["A_h", "B_h", "C_h", "D_h", "E_h", "F_h", "G_h", "H_h", "gamma_tr_h", "gamma_ci_h", "f_h", "S_h", "J_tr_h", "J_ci_h", "const"]

def to_arr(vals, omega):
    """Converts a dictionary of values to an array."""
    arr = np.zeros((len(unknowns), len(omega)), dtype=complex)
    for key in vals.keys():
        if key not in unknowns:
            raise ValueError(f"Unknown key: {key}")

    for i, key in enumerate(unknowns):
        try:
            arr[i, :] = vals[key]
        except KeyError:
            pass

    return arr


## Inverse of Lambda matrix
def Lambda_inv_h(omega):
    const = to_arr({"const": 1}, omega)

    return np.array([
        [1 / omega**2 * const, 0 * const],
        [0 * const, 1 / (omega**2 + zeta) * const]
    ])

## Streamfunction
def psi_h(omega, y):
    return to_arr({
        "A_h": y * np.exp(omega * y),
        "B_h": np.exp(omega * y),
        "C_h": y * np.exp(-omega * y),
        "D_h": np.exp(-omega * y),
    }, omega)

def d_psi_h(omega, y):
    return to_arr({
        "A_h": (1 + omega * y) * np.exp(omega * y),
        "B_h": omega * np.exp(omega * y),
        "C_h": (1 - omega * y) * np.exp(-omega * y),
        "D_h": -omega * np.exp(-omega * y),
    }, omega)

def d2_psi_h(omega, y):
    return to_arr({
        "A_h": omega * (2 + omega * y) * np.exp(omega * y),
        "B_h": omega**2 * np.exp(omega * y),
        "C_h": -omega * (2 - omega * y) * np.exp(-omega * y),
        "D_h": omega**2 * np.exp(-omega * y),
    }, omega)

def d3_psi_h(omega, y):
    return to_arr({
        "A_h": omega**2 * (3 + omega * y) * np.exp(omega * y),
        "B_h": omega**3 * np.exp(omega * y),
        "C_h": omega**2 * (3 - omega * y) * np.exp(-omega * y),
        "D_h": -omega**3 * np.exp(-omega * y),
    }, omega)

## Define coefficients for bulk concentration
def a_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) + zeta)) * to_arr({
        "A_h": 1,
    }, omega)

def b_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) + zeta)**2) * to_arr({
        "A_h": -2 * (omega + np.sqrt(zeta)),
        "B_h": 2 * omega * np.sqrt(zeta) + zeta,
    }, omega)

def c_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) - zeta)) * to_arr({
        "A_h": 1,
    }, omega)

def d_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) - zeta)**2) * to_arr({
        "A_h": 2 * (omega - np.sqrt(zeta)),
        "B_h": 2 * omega * np.sqrt(zeta) - zeta,
    }, omega)

def e_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) - zeta)) * to_arr({
        "C_h": -1,
    }, omega)

def f_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) - zeta)**2) * to_arr({
        "C_h": 2 * (omega - np.sqrt(zeta)),
        "D_h": -(2 * omega * np.sqrt(zeta) - zeta),
    }, omega)

def g_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) + zeta)) * to_arr({
        "C_h": -1,
    }, omega)

def h_p_1(omega):
    return (1 / (2 * omega * np.sqrt(zeta) + zeta)**2) * to_arr({
        "C_h": -2 * (omega + np.sqrt(zeta)),
        "D_h": -(2 * omega * np.sqrt(zeta) + zeta),
    }, omega)

## Define coefficients for bulk concentration
def a_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))) * to_arr({
        "A_h": 1,
    }, omega)

def b_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))**2) * to_arr({
        "A_h": -2 * (omega + np.sqrt(zeta)),
        "B_h": 2 * omega * np.sqrt(zeta),
    }, omega)

def c_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))) * to_arr({
        "A_h": 1,
    }, omega)

def d_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))**2) * to_arr({
        "A_h": 2 * (omega - np.sqrt(zeta)),
        "B_h": 2 * omega * np.sqrt(zeta),
    }, omega)

def e_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))) * to_arr({
        "C_h": -1,
    }, omega)

def f_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))**2) * to_arr({
        "C_h": 2 * (omega - np.sqrt(zeta)),
        "D_h": -2 * omega * np.sqrt(zeta),
    }, omega)

def g_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))) * to_arr({
        "C_h": -1,
    }, omega)

def h_p_2(omega):
    return (1 / (2 * omega * np.sqrt(zeta))**2) * to_arr({
        "C_h": -2 * (omega + np.sqrt(zeta)),
        "D_h": -2 * omega * np.sqrt(zeta),
    }, omega)

## Bulk concentration
def p_h_0(omega, y):
    eq_1 = to_arr({
        "E_h": np.sinh(omega * y),
        "F_h": np.cosh(omega * y),
    }, omega)
    eq_2 = to_arr({
        "G_h": np.sinh(np.sqrt(zeta + omega**2) * y),
        "H_h": np.cosh(np.sqrt(zeta + omega**2) * y),
    }, omega)

    return np.array([eq_1, eq_2])

def p_h_1(omega, y):
    vec = (Lambda @ np.linalg.inv(V)) @ np.array([A_0, 0]) 
    eq_1 = np.einsum("ij...,j->i...", Lambda_inv_h(omega), vec)
    f_vec = to_arr({"f_h": 1}, omega)

    return -np.einsum("i...,j...->ij...", eq_1[:, -1], f_vec)

def p_h_2(omega, y):
    p_1 = to_arr({
        "const": -(
            A_0 / ((alpha + eta) * omega**2)
            + B_0 / (omega**2 - zeta) * np.cosh(y * np.sqrt(zeta))
        ),
    }, omega)
    p_2 = to_arr({
        "const": -(
            A_0 / ((alpha + eta) * (omega**2 + zeta))
            + B_0 / omega**2 * np.cosh(y * np.sqrt(zeta))
        ),
    }, omega)
    zero = to_arr(dict(), omega)

    matr = np.array([
        [p_1, zero],
        [zero, p_2]
    ])
    vec = Lambda @ np.linalg.inv(V) @ np.array([-eta, 1])
    vec_2 = np.einsum("ij...,j->i...", matr, vec)

    f_vec = to_arr({"f_h": 1}, omega)

    return np.einsum("i...,j...->ij...", vec_2[:, -1], f_vec)

def p_h_3_1(omega, y):
    return (
        (a_p_1(omega) * y + b_p_1(omega)) * np.exp((omega + np.sqrt(zeta)) * y)
        + (c_p_1(omega) * y + d_p_1(omega)) * np.exp((omega - np.sqrt(zeta)) * y)
        + (e_p_1(omega) * y + f_p_1(omega)) * np.exp(-(omega - np.sqrt(zeta)) * y)
        + (g_p_1(omega) * y + h_p_1(omega)) * np.exp(-(omega + np.sqrt(zeta)) * y)
    )

def p_h_3_2(omega, y):
    return (
        (a_p_2(omega) * y + b_p_2(omega)) * np.exp((omega + np.sqrt(zeta)) * y)
        + (c_p_2(omega) * y + d_p_2(omega)) * np.exp((omega - np.sqrt(zeta)) * y)
        + (e_p_2(omega) * y + f_p_2(omega)) * np.exp(-(omega - np.sqrt(zeta)) * y)
        + (g_p_2(omega) * y + h_p_2(omega)) * np.exp(-(omega + np.sqrt(zeta)) * y)
    )

def p_h_3(omega, y):
    zero = to_arr(dict(), omega)

    matr = np.array([
        [p_h_3_1(omega, y), zero],
        [zero, p_h_3_2(omega, y)]
    ])
    vec = np.linalg.inv(V) @ P @ np.array([-eta, 1])

    return -(1.j * omega * B_0 * np.sqrt(zeta)) / 2 * np.einsum("ij...,j->i...", matr, vec)


def p_h(omega, y):
    return p_h_0(omega, y) + p_h_1(omega, y) + p_h_2(omega, y) + p_h_3(omega, y)

## Bulk concentration derivatives
def d_p_h_0(omega, y):
    eq_1 = omega * to_arr({
        "E_h": np.cosh(omega * y),
        "F_h": np.sinh(omega * y),
    }, omega)
    eq_2 = np.sqrt(zeta + omega**2) * to_arr({
        "G_h": np.cosh(np.sqrt(zeta + omega**2) * y),
        "H_h": np.sinh(np.sqrt(zeta + omega**2) * y),
    }, omega)

    return np.array([eq_1, eq_2])

def d_p_h_2(omega, y):
    d_p_1 = to_arr({
        "const": -B_0 * np.sqrt(zeta) / (omega**2 - zeta) * np.sinh(y * np.sqrt(zeta)),
    }, omega)
    d_p_2 = to_arr({
        "const": -B_0 * np.sqrt(zeta) / omega**2 * np.sinh(y * np.sqrt(zeta)),
    }, omega)
    zero = to_arr(dict(), omega)

    matr = np.array([
        [d_p_1, zero],
        [zero, d_p_2]
    ])
    vec = Lambda @ np.linalg.inv(V) @ np.array([-eta, 1])
    vec_2 = np.einsum("ij...,j->i...", matr, vec)

    f_vec = to_arr({"f_h": 1}, omega)

    return np.einsum("i...,j...->ij...", vec_2[:, -1], f_vec)

def d_p_h_3_1(omega, y):
    return (
        a_p_1(omega) * np.exp((omega + np.sqrt(zeta)) * y)
        + c_p_1(omega) * np.exp((omega - np.sqrt(zeta)) * y)
        + e_p_1(omega) * np.exp(-(omega - np.sqrt(zeta)) * y)
        + g_p_1(omega) * np.exp(-(omega + np.sqrt(zeta)) * y)
    ) + (
        (omega + np.sqrt(zeta)) * (a_p_1(omega) * y + b_p_1(omega)) * np.exp((omega + np.sqrt(zeta)) * y)
        + (omega - np.sqrt(zeta)) * (c_p_1(omega) * y + d_p_1(omega)) * np.exp((omega - np.sqrt(zeta)) * y)
        - (omega - np.sqrt(zeta)) * (e_p_1(omega) * y + f_p_1(omega)) * np.exp(-(omega - np.sqrt(zeta)) * y)
        - (omega + np.sqrt(zeta)) * (g_p_1(omega) * y + h_p_1(omega)) * np.exp(-(omega + np.sqrt(zeta)) * y)
    )

def d_p_h_3_2(omega, y):
    return (
        a_p_2(omega) * np.exp((omega + np.sqrt(zeta)) * y)
        + c_p_2(omega) * np.exp((omega - np.sqrt(zeta)) * y)
        + e_p_2(omega) * np.exp(-(omega - np.sqrt(zeta)) * y)
        + g_p_2(omega) * np.exp(-(omega + np.sqrt(zeta)) * y)
    ) + (
        (omega + np.sqrt(zeta)) * (a_p_2(omega) * y + b_p_2(omega)) * np.exp((omega + np.sqrt(zeta)) * y)
        + (omega - np.sqrt(zeta)) * (c_p_2(omega) * y + d_p_2(omega)) * np.exp((omega - np.sqrt(zeta)) * y)
        - (omega - np.sqrt(zeta)) * (e_p_2(omega) * y + f_p_2(omega)) * np.exp(-(omega - np.sqrt(zeta)) * y)
        - (omega + np.sqrt(zeta)) * (g_p_2(omega) * y + h_p_2(omega)) * np.exp(-(omega + np.sqrt(zeta)) * y)
    )

def d_p_h_3(omega, y):
    zero = to_arr(dict(), omega)

    matr = np.array([
        [d_p_h_3_1(omega, y), zero],
        [zero, d_p_h_3_2(omega, y)]
    ])
    vec = np.linalg.inv(V) @ P @ np.array([-eta, 1])

    return -(1.j * omega * B_0 * np.sqrt(zeta)) / 2 * np.einsum("ij...,j->i...", matr, vec)


def d_p_h(omega, y):
    return d_p_h_0(omega, y) + d_p_h_2(omega, y) + d_p_h_3(omega, y)

## Revert to c
def c_h(omega, y):
    return np.einsum("ij,j...->i...", V, p_h(omega, y))

def d_c_h(omega, y):
    return np.einsum("ij,j...->i...", V, d_p_h(omega, y))

## Boundary conditions
### No-slip, no-penetration conditions
def no_slip(omega):
    return np.array([d_psi_h(omega, 0), omega * psi_h(omega, 0)])

### Kinematic condition
def kinematic(omega):
    return np.array([psi_h(omega, 1)])

### Normal stress balance
def normal_stress(omega):
    lhs = d3_psi_h(omega, 1) - 3 * omega**2 * d_psi_h(omega, 1)
    rhs = 1.j * omega**3 * (1 - Man * np.log(Delta)) * to_arr({
        "S_h": 1,
    }, omega)

    return np.array([lhs - rhs])

### No-flux condition
def no_flux(omega):
    return d_c_h(omega, 0)

### Kinentic fluxes
def kin_fluxes(omega):
    J_vec = np.array([to_arr({"J_tr_h": 1}, omega), to_arr({"J_ci_h": 1}, omega)])
    S_vec = np.tile(to_arr({"S_h": 1}, omega), (2, 1, 1))
    gamma_vec = np.array([to_arr({"gamma_tr_h": 1}, omega), to_arr({"gamma_ci_h": 1}, omega)])

    eq_1 = c_h(omega, 1) + np.einsum("i...,i->i...", S_vec, d_c_0(1))
    eq_2 = 1 / Delta * np.einsum("ij,j...->i...", K, eq_1)

    eq_3 = np.tile(gamma_vec[0] + gamma_vec[1], (2, 1, 1))
    eq_4 = np.einsum("i,i...->i...", c_0(1), eq_3)
    eq_5 = np.einsum("ij,j...->i...", K, eq_4)

    rhs = np.einsum("ij,j...->i...", B, eq_2 - eq_5 - gamma_vec)

    return J_vec - rhs

### Surface excess concentration equations
def surf_excess(omega):
    gamma_vec = np.array([to_arr({"gamma_tr_h": 1}, omega), to_arr({"gamma_ci_h": 1}, omega)])
    J_vec = np.array([to_arr({"J_tr_h": 1}, omega), to_arr({"J_ci_h": 1}, omega)])
    f_vec = np.tile(to_arr({"f_h": 1}, omega), (2, 1, 1))

    d_psi_vec = np.tile(d_psi_h(omega, 1), (2, 1, 1))
    eq_1 = 1.j * omega * np.einsum("i,i...->i...", P_s @ gamma_0, d_psi_vec)

    eq_2 = omega**2 * gamma_vec

    eq_3 = -np.einsum("ij,j...->i...", P_s, J_vec)

    eq_4 = gamma_vec + np.einsum("i...,i->i...", f_vec, gamma_0)
    eq_5 = np.einsum("ij,j...->i...", A_s, eq_4)

    return eq_1 + eq_2 + eq_3 + eq_5

### Tangential stress balance
def tangential_stress(omega):
    lhs = d2_psi_h(omega, 1) + omega**2 * psi_h(omega, 1)
    rhs = -1.j * omega * Man * Delta * to_arr({
        "gamma_tr_h": 1,
        "gamma_ci_h": 1,
    }, omega)

    return np.array([lhs - rhs])

### Mass balance
def mass_balance(omega):
    J_vec = np.array([to_arr({"J_tr_h": 1}, omega), to_arr({"J_ci_h": 1}, omega)])
    S_vec = np.tile(to_arr({"S_h": 1}, omega), (2, 1, 1))

    eq_1 = d_c_h(omega, 1) + np.einsum("i...,i->i...", S_vec, d2_c_0(1))
    lhs = k_tr * chi_tr * eq_1

    rhs = -np.einsum("ij,j...->i...", P, J_vec)

    return lhs - rhs

### Specify light intensity
def light_intensity(omega, f_h):
    f_vec = to_arr({"f_h": 1}, omega)
    const_vec = to_arr({"const": 1}, omega)

    return np.array([f_vec - f_h(omega) * const_vec])

### Specify surface
def surface(omega, S_h):
    S_vec = to_arr({"S_h": 1}, omega)
    const_vec = to_arr({"const": 1}, omega)

    return np.array([S_vec - S_h(omega) * const_vec])

## Collate equations
### Forward problem
def form_equations(omega, f_h):
    return np.concatenate([
        no_slip(omega),
        kinematic(omega),
        normal_stress(omega),
        no_flux(omega),
        kin_fluxes(omega),
        surf_excess(omega),
        tangential_stress(omega),
        mass_balance(omega),
        light_intensity(omega, f_h),
    ], axis=0)

### Inverse problem
def form_inverse_equations(omega, S_h):
    return np.concatenate([
        no_slip(omega),
        kinematic(omega),
        normal_stress(omega),
        no_flux(omega),
        kin_fluxes(omega),
        surf_excess(omega),
        tangential_stress(omega),
        mass_balance(omega),
        surface(omega, S_h),
    ], axis=0)

# Constant mode
## Define embeddings
unknowns_o = ["A_h", "B_h", "C_h", "A_1", "B_1", "gamma_tr_h", "gamma_ci_h", "f_h", "J_tr_h", "J_ci_h", "const"]

def to_arr_o(vals):
    """Converts a dictionary of values to an array."""
    arr = np.zeros((len(unknowns_o), 1), dtype=complex)
    for key in vals.keys():
        if key not in unknowns_o:
            raise ValueError(f"Unknown key: {key}")

    for i, key in enumerate(unknowns_o):
        try:
            arr[i, :] = vals[key]
        except KeyError:
            pass

    return arr

## Streamfunction
def psi_o(y):
    return to_arr_o({
        "A_h": y**2,
        "B_h": y,
        "C_h": 1,
    })

def d_psi_o(y):
    return to_arr_o({
        "A_h": 2 * y,
        "B_h": 1,
    })

def d2_psi_o(y):
    return to_arr_o({
        "A_h": 2,
    })

## Bulk concentration
def c_ci_o(y):
    return to_arr_o({
        "A_1": 1 / (alpha + eta),
        "B_1": np.cosh(y * np.sqrt(zeta)),
        "f_h": -B_0 * np.cosh(y * np.sqrt(zeta))
    })

def c_tr_o(y):
    return to_arr_o({"A_1": 1}) - eta * c_ci_o(y)

def c_o(y):
    return np.array([c_tr_o(y), c_ci_o(y)])

def d_c_ci_o(y):
    return to_arr_o({
        "B_1": np.sqrt(zeta) * np.sinh(y * np.sqrt(zeta)),
        "f_h": -B_0 * np.sqrt(zeta) * np.sinh(y * np.sqrt(zeta))
    })

def d_c_tr_o(y):
    return -eta * d_c_ci_o(y)

def d_c_o(y):
    return np.array([d_c_tr_o(y), d_c_ci_o(y)])

def i_c_ci_o():
    return to_arr_o({
        "A_1": 1 / (alpha + eta),
        "B_1": np.sinh(np.sqrt(zeta)) / np.sqrt(zeta),
        "f_h": -B_0 * np.sinh(np.sqrt(zeta)) / np.sqrt(zeta)
    })

def i_c_tr_o():
    return to_arr_o({"A_1": 1}) - eta * i_c_ci_o()

def i_c_o():
    return np.array([i_c_tr_o(), i_c_ci_o()])

## Boundary conditions
### No-slip, no-penetration conditions
def no_slip_o():
    return np.array([d_psi_o(0), psi_o(0)])

### Kinentic fluxes
def kin_fluxes_o():
    J_vec = np.array([
        to_arr_o({"J_tr_h": 1}), to_arr_o({"J_ci_h": 1})
    ])
    gamma_vec = np.array([
        to_arr_o({"gamma_tr_h": 1}), to_arr_o({"gamma_ci_h": 1})
    ])

    eq_1 = 1 / Delta * np.einsum("ij,j...->i...", B @ K, c_o(1))
    eq_2 = -np.einsum("ij,j...->i...", M - D, gamma_vec)
    rhs = eq_1 + eq_2

    return J_vec - rhs

### Surface excess concentration equations
def surf_excess_o():
    J_vec = np.array([
        to_arr_o({"J_tr_h": 1}), to_arr_o({"J_ci_h": 1})
    ])
    gamma_vec = np.array([
        to_arr_o({"gamma_tr_h": 1}), to_arr_o({"gamma_ci_h": 1})
    ])
    f_vec = np.tile(to_arr_o({"f_h": 1}), (2, 1, 1))

    eq_1 = np.einsum("ij,j...->i...", D, gamma_vec)
    eq_2 = np.einsum("i...,i->i...", f_vec, D @ gamma_0)
    rhs = eq_1 + eq_2

    return J_vec - rhs

### Tangential stress balance
def tangential_stress_o():
    return np.array([d2_psi_o(1)])

### Mass balance
def mass_balance_o():
    J_ci_vec = to_arr_o({"J_ci_h": 1})

    lhs = (k_ci * chi_ci / Pen_ci) * d_c_ci_o(1)

    rhs = -J_ci_vec

    return np.array([lhs - rhs])

### Integral conditions
def integral_o():
    gamma_vec = to_arr_o({
        "gamma_tr_h": 1,
        "gamma_ci_h": 1,
    })

    return np.array([gamma_vec / (k_tr * chi_tr) + i_c_tr_o() + i_c_ci_o()])

### Light intensity
def light_intensity_o(f_o):
    f_vec = to_arr_o({"f_h": 1})
    const_vec = to_arr_o({"const": 1})

    return np.array([f_vec - f_o * const_vec])

### Collate equations
def form_equations_o(f_o):
    return np.concatenate([
        no_slip_o(),
        kin_fluxes_o(),
        surf_excess_o(),
        tangential_stress_o(),
        mass_balance_o(),
        integral_o(),
        light_intensity_o(f_o),
    ], axis=0)

def form_inverse_equations_o():
    return np.concatenate([
        no_slip_o(),
        kin_fluxes_o(),
        surf_excess_o(),
        tangential_stress_o(),
        mass_balance_o(),
        integral_o(),
        light_intensity_o(1)
    ], axis=0)[:, :, 0]
