"""Example on how to use GraphNeTI3Deployer with GraphNeTI3Modules."""

from glob import glob
from os.path import join
from typing import TYPE_CHECKING

from graphnet.constants import (
    TEST_DATA_DIR,
    EXAMPLE_OUTPUT_DIR,
)
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors.i3featureextractor import (
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

if has_icecube_package() or TYPE_CHECKING:
    from graphnet.deployment.i3modules import (
        GraphNeTI3Deployer,
        I3InferenceModule,
    )

from _common_icetray import ERROR_MESSAGE_MISSING_ICETRAY

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE


def main() -> None:
    """GraphNeTI3Deployer Example."""
    # configure input files, output folders and pulsemap
    pulsemap = "SplitInIcePulses"
    input_folders = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    mock_model_path = f"{TEST_DATA_DIR}/models/mock_energy_model.pth"
    output_folder = f"{EXAMPLE_OUTPUT_DIR}/i3_deployment/upgrade_03_04"
    gcd_file = f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V0.i3.bz2"
    input_files = []
    for folder in input_folders:
        input_files.extend(glob(join(folder, "*.i3.gz")))

    # Configure Deployment module
    deployment_module = I3InferenceModule(
        pulsemap=pulsemap,
        features=features,
        pulsemap_extractor=I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap),
        model=mock_model_path,
        gcd_file=gcd_file,
        prediction_columns=["energy"],
        model_name="graphnet_deployment_example",
    )

    # Construct I3 deployer
    deployer = GraphNeTI3Deployer(
        graphnet_modules=[deployment_module],
        n_workers=1,
        gcd_file=gcd_file,
    )

    # Start deployment - files will be written to output_folder
    deployer.run(
        input_files=input_files,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)

    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Use GraphNeTI3Modules to deploy trained model with GraphNeTI3Deployer.
"""
        )

        args = parser.parse_args()

        # Run example script
        main()
