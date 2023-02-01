"""Contains the graphnet i3 deployment module."""
import os.path
import os
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, List, Optional
import time
import numpy as np
import torch

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
    from I3Tray import I3Tray


class GraphNeTI3Deployer:
    """Deploys graphnet i3 modules to i3 files."""

    def __init__(
        self,
        graphnet_modules: List[Any],
        gcd_file: str,
        n_workers: int = 1,
    ) -> None:
        """Construts the I3 Deployment class."""
        # This makes sure that one worker cannot access more than 1 core's worth of compute.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Check
        if isinstance(graphnet_modules, list):
            self._modules = graphnet_modules
        else:
            self._modules = [graphnet_modules]
        self._gcd_file = gcd_file
        self._n_workers = n_workers

    def _setup_settings(self, input_files: str, output_folder: str) -> List:
        os.makedirs(output_folder)
        settings = []
        if self._n_workers > len(input_files):
            self._n_workers = len(input_files)
        file_batches = np.array_split(input_files, self._n_workers)

        for i in range(self._n_workers):
            settings.append([file_batches[i], self._gcd_file, output_folder])
        return settings

    def _launch_jobs(self, settings: List) -> None:
        if self._n_workers > 1:
            p = Pool(processes=len(settings))
            # process_files(settings[0])
            _ = p.map_async(self._process_files, settings)
            p.close()
            p.join()
            print("done")
        else:
            self._process_files(settings)

    def _process_files(self, settings: List) -> None:
        files, gcd_file, output_folder = settings
        for i in range(0, len(files)):
            INFILE = files[i]
            tray = I3Tray()
            tray.context["I3FileStager"] = dataio.get_stagers()
            tray.AddModule(
                "I3Reader",
                "reader",
                FilenameList=[gcd_file, INFILE],
            )

            for graphnet_i3_module in self._modules:
                tray.AddModule(graphnet_i3_module)

            tray.Add(
                "I3Writer",
                Streams=[
                    icetray.I3Frame.DAQ,
                    icetray.I3Frame.Physics,
                    icetray.I3Frame.TrayInfo,
                    icetray.I3Frame.Simulation,
                ],
                filename=output_folder + "/" + INFILE.split("/")[-1],
            )
            tray.Execute()
            tray.Finish()
        return

    def run(
        self,
        input_files: str,
        output_folder: str,
        gcd_file: Optional[str] = None,
    ) -> None:
        """Apply the graphnet modules to all input files using n workers."""
        if __name__ == "__main__":
            start_time = time.time()
            if gcd_file is not None:
                self._gcd_file = gcd_file
            settings = self._setup_settings(
                input_files=input_files, output_folder=output_folder
            )
            print(
                f"processing {len(input_files)} i3 files using {self._n_workers} workers"
            )
            self._launch_jobs(settings)
            print(
                f"Processing {len(input_files)} was completed in {time.time() - start_time} seconds using {self._n_workers}."
            )
