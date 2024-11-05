"""Example on how to use GraphNeTI3Deployer with GraphNeTI3Modules."""

from glob import glob
from os.path import join
from typing import TYPE_CHECKING

from graphnet.constants import (
    TEST_DATA_DIR,
    EXAMPLE_OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
)
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors.icecube import (
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

ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE


def main() -> None:
    """GraphNeTI3Deployer Example."""
    # configure input files, output folders and pulsemap
    pulsemap = "SplitInIcePulses"
    input_folders = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    base_path = f"{PRETRAINED_MODEL_DIR}/icecube/upgrade/QUESO"
    model_name = "total_neutrino_energy"
    model_config = f"{base_path}/{model_name}/{model_name}_config.yml"
    state_dict = f"{base_path}/{model_name}/{model_name}_state_dict.pth"
    output_folder = f"{EXAMPLE_OUTPUT_DIR}/i3_deployment/upgrade_03_04"
    gcd_file = f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V0.i3.bz2"  # noqa: E501
    input_files = []
    for folder in input_folders:
        input_files.extend(glob(join(folder, "*.i3.gz")))

    # Configure Deployment module
    deployment_module = I3InferenceModule(
        pulsemap=pulsemap,
        features=features,
        pulsemap_extractor=I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap),
        model_config=model_config,
        state_dict=state_dict,
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

        args, unknown = parser.parse_known_args()

        # Run example script
        main()
