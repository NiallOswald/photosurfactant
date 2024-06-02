"""Parameters for the model."""

# Non-dimensional parameters
# Cell length
L = 10.0

# Damkohler numbers
Dam_tr = 1.0
Dam_ci = 1.4

# Peclet numbers
Pen_tr = 10.0
Pen_ci = 15.0

Pen_tr_s = 10.0
Pen_ci_s = 15.0

# Biot numbers
Bit_tr = 1.0e3
Bit_ci = 3.33

# Marangoni numbers
Man = 1.03

# Switching rate
k_tr = 1.0
k_ci = 30 * k_tr

chi_tr = 30.0
chi_ci = 1.0

assert k_tr * chi_tr == k_ci * chi_ci

# Beam width
delta = 1.0

# DO NOT MODIFY
# Setup required parameters
alpha = Dam_ci / Dam_tr
eta = Pen_tr / Pen_ci
zeta = Pen_tr * Dam_tr + Pen_ci * Dam_ci
