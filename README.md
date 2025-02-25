***The paper is currently a work-in-progress. Further references will be provided upon submission.***

# Liquid mixing and sculpting using light-actuated photosurfactants

## Installation
This package can be installed using `pip` through PyPI (Python Package Index). Simply run:
```bash
$ python -m pip install <not-yet-available>
```
This will install the `photosurfactant` Python package and any necessary dependencies.

## Usage
This package includes shell scripts which feature enough functionality for uses to experiment with the package without any knowledge of Python. The scripts are separated into the leading-order and first-order problems and can be called with a wide variety of arguments as detailed in the documentation by calling:
```bash
$ plot_first_order --help
```
```
usage: plot_first_order [-h] [--L L] [--Dam_tr DAM_TR] [--Dam_ci DAM_CI] [--Pen_tr PEN_TR]
                        [--Pen_ci PEN_CI] [--Pen_tr_s PEN_TR_S] [--Pen_ci_s PEN_CI_S]
                        [--Bit_tr BIT_TR] [--Bit_ci BIT_CI] [--Man MAN] [--k_tr K_TR]
                        [--k_ci K_CI] [--chi_tr CHI_TR] [--chi_ci CHI_CI]
                        [--wave_count WAVE_COUNT] [--grid_size GRID_SIZE] [-s] [--path PATH]
                        [--label LABEL] [--usetex] [--format FORMAT]
                        [--root_index ROOT_INDEX] [--func FUNC]
                        [--problem {forward,inverse}] [--mollify] [--delta DELTA]

Plot the first order surfactant concentrations.

options:
  -h, --help            show this help message and exit
  --L L                 The aspect ratio of the domain. (default: 10.0)
  --Dam_tr DAM_TR       The Damkohler number for the trans surfactant. (default: 1.0)
  --Dam_ci DAM_CI       The Damkohler number for the cis surfactant. (default: 2.0)
  --Pen_tr PEN_TR       The Peclet number for the trans surfactant. (default: 10.0)
  --Pen_ci PEN_CI       The Peclet number for the cis surfactant. (default: 10.0)
  --Pen_tr_s PEN_TR_S   The Peclet number for the trans surfactant on the interface.
                        (default: 10.0)
  --Pen_ci_s PEN_CI_S   The Peclet number for the cis surfactant on the interface. (default:
                        10.0)
  --Bit_tr BIT_TR       The Biot number for the trans surfactant. (default:
                        0.0033333333333333335)
  --Bit_ci BIT_CI       The Biot number for the cis surfactant. (default: 1.0)
  --Man MAN             The Marangoni number. (default: 2.0)
  --k_tr K_TR           The adsorption rate for the trans surfactant. (default: 1.0)
  --k_ci K_CI           The adsorption rate for the cis surfactant. (default:
                        0.03333333333333333)
  --chi_tr CHI_TR       The desorption rate for the trans surfactant. (default:
                        3.3333333333333335)
  --chi_ci CHI_CI       The desorption rate for the cis surfactant. (default: 100.0)
  --wave_count WAVE_COUNT
                        Number of wavenumbers to use. (default: 100)
  --grid_size GRID_SIZE
                        Number of grid points to evaluate the solution on. (default: 1000)
  -s, --save            Save the figures to disk. (default: False)
  --path PATH           Path to save the figures to. (default: ./)
  --label LABEL         Label to append to the figure filenames. (default: None)
  --usetex              Use LaTeX for rendering text. (default: False)
  --format FORMAT       Format to save the figures in. (default: png)
  --root_index ROOT_INDEX
                        The index of solution branch for the leading order problem. If set
                        to -1, the branch is selected automatically. (default: -1)
  --func FUNC           An expression in the coordinate x for the light intensity/interface
                        perturbation. The function should be a quoted string. E.g. "sin(x)".
                        The function must be 2L-periodic and always return a float.
                        (default: super_gaussian(x, 4.0, 1.0))
  --problem {forward,inverse}
                        The type of problem to solve. (default: forward)
  --mollify             Apply mollification to the light intensity/interface perturbation.
                        (default: False)
  --delta DELTA         The mollification parameter for the light intensity/interface
                        perturbation. (default: 0.5)
```

### Leading-Order
For the leading-order problem the user may specify the non-dimensional parameters as well as the solution branch to use. In most circumstances `--root_index` can be left to automatic. Do note that for the leading-order problem the default parameters will exactly replicate the figures produced in the paper.

For example if a user wanted to plot the concentrations of the leading-order system but for different Damköhler numbers you can run:
```bash
$ plot_leading_order --Dam_tr 2 --Dam_ci 1
```

### First-Order
For the first-order correction, all of the parameters of the leading-order remain, with the addition of the light-intensity/interface profile and the type of problem to solve (forward/inverse).

The profile can be specified with the `--func` argument and must be $2L$-periodic. Parameters specified elsewhere can also be accessed through the `params` object and access to all numpy functions is possible through `np`. So if a user wished to solve an inverse problem with a sinusoidal surface you may run:
```bash
$ plot_first_order --func "np.sin(2 * np.pi * x / params.L)" --problem inverse
```

### Advanced Usage
For users who want more control and to access a broader range of inverse problems, the code used in the above scripts can also be imported into your own Python code. There are additional modules containing helper-functions which are intended to be used together to keep your code simple. See the examples below for some possibilities provided by the current package.

## Examples

### Solving an inverse problem by specifying the slip velocity

```python
from photosurfactant import Parameters, LeadingOrder, FirstOrder, Variables
from photosurfactant.fourier import fourier_series_coeff

import numpy as np
import matplotlib.pyplot as plt

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
        (first._psi(wavenumbers[n], 1, y_order=1), func_coeffs[n])
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
```
