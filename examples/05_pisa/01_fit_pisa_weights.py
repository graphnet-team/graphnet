"""Example of using the PISA WeightFitter class."""

from graphnet.pisa.fitting import WeightFitter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_pisa_package
from graphnet.utilities.logging import Logger

from _common_pisa import ERROR_MESSAGE_MISSING_PISA


def main() -> None:
    """Run example."""
    # Configuration
    outdir = "/mnt/scratch/rasmus_orsoe/weight_test"  # @TEMP
    database_path = (  # @TEMP
        "/mnt/scratch/rasmus_orsoe/databases/dev_lvl3_genie_burnsample_v5/data"
        "/dev_lvl3_genie_burnsample_v5.db"
    )
    fitter = WeightFitter(database_path=database_path)

    pisa_config_dict = {
        "reco_energy": {"num_bins": 8},
        "reco_coszen": {"num_bins": 8},
        "pid": {"bin_edges": [0, 0.5, 1]},
        "true_energy": {"num_bins": 200},
        "true_coszen": {"num_bins": 200},
        "livetime": 10 * 0.01,  # 1% of 10 years, like oscNext burn sample.
    }

    # By calling `fitter.fit_weights`` we get the weights returned per event.
    # If `add_to_database = True`, a table will be added to the database.
    weights = fitter.fit_weights(
        outdir,
        add_to_database=True,
        weight_name="weight_livetime10_1percent",
        pisa_config_dict=pisa_config_dict,
    )

    print(weights)


if __name__ == "__main__":

    if not has_pisa_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_PISA)

    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Use the PISA WeightFitter class.
"""
        )

        args = parser.parse_args()

        main()
