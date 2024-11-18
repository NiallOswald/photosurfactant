#! /bin/bash
echo "Plotting leading order..."
plot_leading_order -s --path "./figures/" --label "blue" --grid_size 1000 --usetex --format "pdf"

echo "Plotting first order: laser pointer..."
plot_first_order -s --path "./figures/laser-point/" --label "laser" --grid_size 1000 --usetex --format "pdf"
echo "Plotting first order: mixing..."
plot_first_order --func "np.cos(2 * np.pi * x / L)" -s --path "./figures/mixing/" --label "mixing" --grid_size 1000 --usetex --format "pdf"
echo "Plotting first order: inverse problem..."
plot_first_order --func "1e-4 * (np.cos(np.pi * x / L) + 2 * np.sin(2 * np.pi * x / L)) " --problem inverse --wave_count 10 -s --path "./figures/inverse/" --label "inverse" --grid_size 1000 --usetex --format "pdf"
echo "Plotting first order: power spectrum..."
plot_spectrum -s --path "./figures/" --grid_size 1000 --usetex --format "pdf"

echo "Plotting convergence...:"
plot_error
