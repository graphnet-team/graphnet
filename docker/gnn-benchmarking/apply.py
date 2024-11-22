"""Script for applying GraphNeTModule in IceTray chain."""

import argparse
from glob import glob
from os import makedirs
from os.path import join, dirname
from typing import List

from icecube.icetray import I3Tray  # pyright: reportMissingImports=false

from graphnet.deployment.i3modules import I3InferenceModule
from graphnet.data.extractors.i3featureextractor import (
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.data.constants import FEATURES
from graphnet.constants import PRETRAINED_MODEL_DIR


# Constants
MODEL_NAME = "total_neutrino_energy"
BASE_PATH = f"{PRETRAINED_MODEL_DIR}/icecube/upgrade/QUESO"
MODEL_CONFIG = f"{BASE_PATH}/{MODEL_NAME}/{MODEL_NAME}_config.yml"
STATE_DICT = f"{BASE_PATH}/{MODEL_NAME}/{MODEL_NAME}_state_dict.pth"


def main(
    input_files: List[str],
    output_file: str,
    pulsemap: str,
    events_max: int,
) -> None:
    """Apply GraphNeTModule in I3Tray."""
    # Make sure output directory exists
    makedirs(dirname(output_file), exist_ok=True)

    # Get GCD file
    gcd_pattern = "GeoCalibDetector"
    gcd_candidates = [p for p in input_files if gcd_pattern in p]
    assert len(gcd_candidates) == 1, "Did not get exactly one GCD-file "
    gcd_file = gcd_candidates[0]

    # Get all input I3-files
    input_files = [p for p in input_files if gcd_pattern not in p]

    # Construct I3InferenceModule(s)
    extractor = I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap)

    deployment_modules = [
        I3InferenceModule(
            pulsemap=pulsemap,
            features=FEATURES.DEEPCORE,
            pulsemap_extractor=extractor,
            model_config=MODEL_CONFIG,
            state_dict=STATE_DICT,
            gcd_file=gcd_file,
            prediction_columns=["energy_pred"],
            model_name="graphnet_dynedge_energy_reconstruction",
        ),
    ]

    # Run GNN module in tray
    tray = I3Tray()
    tray.Add("I3Reader", filenamelist=input_files)
    for deployment_module in deployment_modules:
        tray.AddModule(deployment_module)
    tray.Add("I3Writer", filename=output_file)
    if events_max > 0:
        tray.Execute(events_max)
    else:
        tray.Execute()


# Main function call
if __name__ == "__main__":
    """The main function must get an input folder and output folder!

    Args:
        input_folder (str): The input folder where i3 files of a
                            given dataset are located.
        output_folder (str): The output folder where processed i3
                            files will be saved.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("pulsemap", nargs="?", default="SplitInIcePulses")
    parser.add_argument("events_max", nargs="?", type=int, default=0)

    args = parser.parse_args()

    input_files = glob(join(args.input_folder, "*.i3*"))
    output_file = join(args.output_folder, "output.i3")

    input_files.sort(key=str.lower)

    main(input_files, output_file, args.key, args.events_max)
