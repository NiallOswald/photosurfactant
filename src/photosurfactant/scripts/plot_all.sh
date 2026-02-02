#! /bin/bash
echo "Plotting leading order..."
plot_leading_order --intensities -6.0 2.0 --log --count 13 -s --path "./figures/leading/" --usetex --format "eps"
plot_leading_order --interface --limits --intensities -6.0 2.0 --log --count 1000 -s --path "./figures/leading/" --usetex --format "eps"

echo "Plotting first order: laser pointer..."
plot_first_order -s --path "./figures/laser-point/" --usetex --format "eps"
echo "Plotting first order: mixing..."
plot_first_order --func "np.cos(2 * np.pi * x / params.L)" -s --path "./figures/mixing/" --usetex --format "eps"
echo "Plotting first order: inverse problem..."
plot_first_order --func "np.cos(np.pi * x / params.L) + np.cos(2 * np.pi * x / params.L) / 2 - np.cos(3 * np.pi * x / params.L) / 3" --problem inverse --wave_count 4 -s --path "./figures/inverse/" --usetex --format "eps"
echo "Plotting first order: inverse problem..."
plot_first_order --func "(np.sinh(1) / (np.cosh(1) - np.cos(np.pi * x / params.L)) - 1) / 2" --problem inverse --wave_count 30 -s --path "./figures/inverse-adv/" --usetex --format "eps"

echo "Plotting first order: sweeping parameters..."
plot_sweep -s --path "./figures/sweep/" --usetex --format "eps"

echo "Plotting convergence...:"
plot_error -s --path "" --usetex --format "eps"
