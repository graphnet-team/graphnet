"""Example of plotting contours from PISA fit.

Here we would like to plot two contours in one figure; one for our GNN and one
for RETRO. We build a dictionary for each contour. Each dictionary much contain
"path" and "model". "path" is the path to the .csv file containing the fit
result. "model" is the name of the model in the .csv file - some fits have more
than 1 model! The plotting script returns the figure object - remember to save
it yourself!
"""

from graphnet.pisa.plotting import plot_2D_contour, plot_1D_contour


def example_plot_2d_contour() -> None:
    """Plot 2D contour from PISA fit."""
    contour_data_2D = [
        {
            "path": "/mnt/scratch/rasmus_orsoe/oscillation/30x30_std_config_final_num_bins_15_lbe_0.4_hbe_0.8/merged_results.csv",
            "model": "dynedge",
            "label": "dynedge",
            "color": "tab:blue",
        },
        {
            "path": "/mnt/scratch/rasmus_orsoe/oscillation/30x30_oscNext_config_final_num_bins_15_lbe_0.5_hbe_0.85/merged_results.csv",
            "model": "retro",
            "label": "retro",
            "color": "tab:orange",
        },
    ]

    figure = plot_2D_contour(contour_data_2D, width=6.3, height=2.3 * 2)
    figure.savefig(
        "/home/iwsatlas1/oersoe/phd/oscillations/plots/2d_contour_test.pdf"
    )


def example_plot_1d_contour() -> None:
    """Plot 1D contour from PISA fit."""
    contour_data_1D = [
        {
            "path": "/home/iwsatlas1/oersoe/phd/oscillations/sensitivities/100x_bfgs_pid_bin_05_8by8_bins_fix_all_True_philipp_idea/merged_results.csv",
            "color": "tab:orange",
            "model": "retro",
            "label": "retro - vanilla bin",
            "ls": "--",
        },
        {
            "path": "/home/iwsatlas1/oersoe/phd/oscillations/sensitivities/100x_bfgs_pid_bin_05_8by8_bins_fix_all_True_philipp_idea/merged_results.csv",
            "color": "tab:blue",
            "model": "dynedge",
            "label": "dynedge - vanilla bin",
            "ls": "--",
        },
    ]

    figure = plot_1D_contour(contour_data_1D)
    figure.savefig(
        "/home/iwsatlas1/oersoe/phd/oscillations/plots/1d_contour_test.pdf"
    )


if __name__ == "__main__":
    example_plot_2d_contour()
    example_plot_1d_contour()
