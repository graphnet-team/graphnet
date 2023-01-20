"""Example of fitting oscillation parameter contours using PISA."""

from graphnet.pisa.fitting import ContourFitter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_pisa_package
from graphnet.utilities.logging import get_logger

logger = get_logger()


def main() -> None:
    """Run example."""
    # This configuration dictionary overwrites our PISA standard with your
    # preferences.
    # @NOTE: num_bins should not be higer than 25 for reconstructions.
    config_dict = {
        "reco_energy": {"num_bins": 10},
        "reco_coszen": {"num_bins": 10},
        "pid": {"bin_edges": [0, 0.50, 1]},
        "true_energy": {"num_bins": 200},
        "true_coszen": {"num_bins": 200},
    }

    # Where you want the .csv-file with the results.
    outdir = "/home/iwsatlas1/oersoe/phd/oscillations/sensitivities"  # @TEMP

    # What you call your run.
    run_name = "this_is_a_test_run"

    pipeline_path = (
        "/mnt/scratch/rasmus_orsoe/databases/oscillations/"
        "dev_lvl7_robustness_muon_neutrino_0000/pipelines/"
        "pipeline_oscillation_final/pipeline_oscillation_final.db"
    )

    fitter = ContourFitter(
        outdir=outdir,
        pipeline_path=pipeline_path,
        post_fix="_pred",
        model_name="dynedge",
        include_retro=True,
        statistical_fit=True,
    )

    # Fits 1D contours of dm31 and theta23 individually
    fitter.fit_1d_contour(
        run_name=run_name + "_1d",
        config_dict=config_dict,
        grid_size=5,
        n_workers=30,
    )

    # Fits 2D contours of dm31 and theta23 together
    fitter.fit_2d_contour(
        run_name=run_name + "_2d",
        config_dict=config_dict,
        grid_size=5,
        n_workers=30,
    )


if __name__ == "__main__":
    if not has_pisa_package():
        logger.error(
            "This example requries PISA to be installed, which doesn't seem "
            "to be the case. Please install PISA or run an example scripts in "
            "one of the other folders:"
            "\n * examples/01_icetray/"
            "\n * examples/02_data/"
            "\n * examples/03_weights/"
            "\n * examples/04_training/"
            "\nExiting."
        )

    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Fit oscillation parameter contours using PISA.
"""
        )

        args = parser.parse_args()

        main()
