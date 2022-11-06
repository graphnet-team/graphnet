"""Script for applying GraphNeTModule in IceTray chain."""

import argparse
from glob import glob
from os import makedirs
from os.path import join, dirname
from typing import List

from I3Tray import I3Tray  # pyright: reportMissingImports=false

from graphnet.deployment.i3modules import GraphNeTModuleIceCube86


# Constants (from Dockerfile)
MODEL_PATH = "model.pth"


# Main function definition
def main(
    input_files: List[str], output_file: str, key: str, events_max: int
) -> None:
    """Apply GraphNeTModule in I3Tray."""
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

    # Run GNN module in tray
    tray = I3Tray()
    tray.Add("I3Reader", filenamelist=input_files)
    tray.Add(
        GraphNeTModuleIceCube86,
        key=key,
        model_path=MODEL_PATH,
        gcd_file=gcd_file,
    )
    tray.Add("I3Writer", filename=output_file)
    if events_max > 0:
        tray.Execute(events_max)
    else:
        tray.Execute()


# Main function call
if __name__ == "__main__":
    """The main function must get an input folder and output folder!

    Args:
        input_folder (str): The input folder where i3 files of a given dataset are located.
        output_folder (str): The output folder where processed i3 files will be saved.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("key", nargs="?", default="gnn_zenith")
    parser.add_argument("events_max", nargs="?", type=int, default=0)

    args = parser.parse_args()

    input_files = glob(join(args.input_folder, "*.i3*"))
    output_file = join(args.output_folder, "output.i3")

    input_files.sort(key=str.lower)

    main(input_files, output_file, args.key, args.events_max)
