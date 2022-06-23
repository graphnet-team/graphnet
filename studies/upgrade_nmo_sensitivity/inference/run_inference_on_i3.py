"""Running inference on GraphSAGE-cleaned pulses in IceCube-Upgrade."""

import argparse
from typing import List
import dill
from glob import glob
import logging
from os import makedirs
from os.path import join, dirname

import torch

from I3Tray import I3Tray  # pyright: reportMissingImports=false

from graphnet.deployment.i3modules import GraphNeTModuleIceCubeUpgrade
from graphnet.utilities.logging import get_logger

logger = get_logger(logging.DEBUG)

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")


# Main function definition
def main(
    input_files: List[str],
    output_file: str,
    model_path: str,
    keys,
    events_max: int,
):
    """Runs inference on `input_files` using the model in `model_path` and produces `output_file`."""

    # Make sure output directory exists
    makedirs(dirname(output_file), exist_ok=True)

    # Get GCD file
    gcd_pattern = "GeoCalibDetector"
    gcd_candidates = [p for p in input_files if gcd_pattern in p]
    assert (
        len(gcd_candidates) == 1
    ), f"Did not get exactly one GCD-file candidate in `{dirname(input_files[0])}: {gcd_candidates}"
    gcd_file = gcd_candidates[0]

    # Get all input I3-files
    input_files = [p for p in input_files if gcd_pattern not in p]

    # Load model
    model = torch.load(
        model_path, map_location=torch.device("cpu"), pickle_module=dill
    )

    # Run GNN module in tray
    tray = I3Tray()
    tray.Add("I3Reader", filenamelist=input_files)
    tray.Add(
        lambda frame: frame["SplitInIcePulses_GraphSage_Pulses"].sum() > 1
    )
    tray.Add(
        GraphNeTModuleIceCubeUpgrade,
        keys=keys,
        model=model,
        gcd_file=gcd_file,
        pulsemaps=["SplitInIcePulses_GraphSage_Pulses"],
    )

    tray.Add("I3Writer", filename=output_file)
    if events_max > 0:
        tray.Execute(events_max)
    else:
        tray.Execute()


# Main function call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("model_path")
    parser.add_argument("--keys", nargs="+", default="gnn_zenith")
    parser.add_argument("--events_max", type=int, default=0)

    args = parser.parse_args()

    input_files = glob(join(args.input_folder, "*.i3*"))
    output_file = join(args.output_folder, "output.i3")

    input_files.sort(key=str.lower)

    main(input_files, output_file, args.model_path, args.keys, args.events_max)
