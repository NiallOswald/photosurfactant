#! /bin/bash
echo "Plotting leading order..."
plot_leading_order -s --path "./figures/" --grid_size 1000 --usetex --format "pdf"

echo "Plotting first order: laser pointer..."
plot_first_order -s --path "./figures/laser-point/" --grid_size 1000 --usetex --format "pdf"
echo "Plotting first order: mixing..."
plot_first_order --func "np.cos(2 * np.pi * x / L)" -s --path "./figures/mixing/" --grid_size 1000 --usetex --format "pdf"
echo "Plotting first order: inverse problem..."
plot_first_order --func "square_wave(x)" --mollify --problem inverse --wave_count 30 -s --path "./figures/inverse/" --grid_size 1000 --usetex --format "pdf"
echo "Plotting first order: power spectrum..."
plot_spectrum -s --path "./figures/" --grid_size 1000 --usetex --format "pdf"

echo "Plotting convergence...:"
plot_error
