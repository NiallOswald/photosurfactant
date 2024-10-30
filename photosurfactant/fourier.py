"""Module for Fourier series calculations."""

import numpy as np


def fourier_series_coeff(func, L, N):
    """Calculate the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(x) with period 2L, this function returns the
    coefficients {c0,c1,c2,...} such that:

    f(x) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*x/L)

    where we define c_{-n} = complex_conjugate(c_{n}).

    :param func: The periodic function, a callable like f(x).
    :param L: The period of the function f, so that f(0)==f(L).
    :param N: The function will return the first N + 1 Fourier coeff.
    """
    xx = np.linspace(-L, L, 2 * N, endpoint=False)
    f_coeffs = np.fft.rfft(np.array([func(x) for x in xx])) / len(xx)
    f_full = np.concatenate([[f_coeffs[0]], f_coeffs[1:][::-1].conj(), f_coeffs[1:]])

    omega_full = np.arange(-N, N + 1) * np.pi / L
    # Excluding the zero frequency
    omega = np.concatenate([omega_full[:N], omega_full[N + 1 :]])

    return omega, f_full
