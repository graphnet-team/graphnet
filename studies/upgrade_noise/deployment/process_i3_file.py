import argparse
from glob import glob
from os import makedirs
from os.path import join, dirname

from I3Tray import I3Tray
from graphnet.data.utils import is_gcd_file
from icecube import dataclasses

import torch
from torch.optim.adam import Adam

from graphnet.modules import GraphNeTModuleIceCubeUpgrade

# Constants
BASE_DIR = "/groups/icecube/asogaard/gnn/upgrade_sensitivity"
RUN_NAME = "dev_step4_numu_140021_second_run"
MODEL_NAME = "upgrade_energy_regression_45e_GraphSagePulses"
MODEL_PATH = (
    f"{BASE_DIR}/results/{RUN_NAME}/{MODEL_NAME}/{MODEL_NAME}_state_dict.pth"
)


# Main function definition
def main(input_files, output_file, key, pulsemaps, gcd_file, events_max):
    """Run minimal icetray chain with GraphNeT module."""

    # Make sure output directory exists
    makedirs(dirname(output_file), exist_ok=True)

    # Get GCD file
    if gcd_file is None:
        gcd_candidates = [p for p in input_files if is_gcd_file(p)]
        assert (
            len(gcd_candidates) == 1
        ), f"Did not get exactly one GCD-file candidate in `{dirname(input_files[0])}: {gcd_candidates}"
        gcd_file = gcd_candidates[0]

    # Get all input I3-files
    input_files = [p for p in input_files if not is_gcd_file(p)]

    # Run graphnet module in tray
    min_hit_oms = 10

    tray = I3Tray()
    tray.Add("I3Reader", filenamelist=input_files)
    tray.Add(
        lambda frame: len(
            dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulsemaps[0])
        )
        >= min_hit_oms,
    )
    tray.Add(
        GraphNeTModuleIceCubeUpgrade,
        keys=key,
        model=MODEL_PATH,
        pulsemaps=pulsemaps,
        gcd_file=gcd_file,
    )
    tray.Add("I3Writer", filename=output_file)

    if events_max > 0:
        tray.Execute(events_max)
    else:
        tray.Execute()


# Main function call
if __name__ == "__main__":

    # Configure arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_folder",
        help="The input folder where i3 files of a given dataset are located.",
    )
    parser.add_argument(
        "output_folder",
        help="The output folder where processed i3 files will be saved.",
    )
    parser.add_argument("--key", nargs="?", default="graphnet_energy")
    parser.add_argument(
        "--pulsemaps", nargs="+", default=["SplitInIcePulses_GraphSage_Pulses"]
    )
    parser.add_argument("--gcd_file", nargs="?", default=None)
    parser.add_argument("--events_max", nargs="?", type=int, default=0)

    # Parse commmand-line arguments
    args = parser.parse_args()

    # Get input and output files
    input_files = glob(join(args.input_folder, "*.i3*"))
    input_files.sort(key=str.lower)

    output_file = join(args.output_folder, "output.i3.zst")

    # Run main function
    main(
        input_files,
        output_file,
        args.key,
        args.pulsemaps,
        args.gcd_file,
        args.events_max,
    )
