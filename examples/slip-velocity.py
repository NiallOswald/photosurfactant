import matplotlib.pyplot as plt
import numpy as np

from photosurfactant import Parameters
from photosurfactant.fourier import fourier_series_coeff
from photosurfactant.semi_analytic import (
    FirstOrder,
    LeadingOrder,
    Variables,
)

params = Parameters()
leading = LeadingOrder(params)

# Find the Fourier series of the slip velocity
wavenumbers, func_coeffs = fourier_series_coeff(
    lambda x: 1e-3 * np.sin(2 * np.pi * x / params.L), params.L, 10
)

# Solve the first-order problem by fixing u(x, 1) = f(x)
first = FirstOrder(wavenumbers, params, leading)
first.solve(
    lambda n: (
        (first._psi(wavenumbers[n], 1, z_order=1), func_coeffs[n])
        if n > 0
        else (Variables.f, 0)
    )  # There is no flow at n = 0, so we fix the light intensity instead
)

# Evaluate and plot the light intensity profile
xx = np.linspace(-params.L, params.L, 100)

plt.plot(xx, first.f(xx))
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.show()
