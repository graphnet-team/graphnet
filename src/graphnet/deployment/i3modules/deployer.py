"""Contains the graphnet i3 deployment module."""
import os.path
import os
import random
import multiprocessing
from typing import TYPE_CHECKING, List, Union, Sequence
import time
import numpy as np
from dataclasses import dataclass

from graphnet.utilities.imports import has_icecube_package, has_torch_package
from graphnet.deployment.i3modules import (
    GraphNeTI3Module,
)

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
    from I3Tray import I3Tray

if has_torch_package or TYPE_CHECKING:
    import torch


@dataclass
class Settings:
    """Dataclass for workers in GraphNeTI3Deployer."""

    i3_files: List[str]
    gcd_file: str
    output_folder: str
    modules: List[GraphNeTI3Module]


class GraphNeTI3Deployer:
    """Deploys graphnet i3 modules to i3 files.

    Modules are applied in the order in which they appear in graphnet_modules.
    """

    def __init__(
        self,
        graphnet_modules: Union[GraphNeTI3Module, Sequence[GraphNeTI3Module]],
        gcd_file: str,
        n_workers: int = 1,
    ) -> None:
        """Initialize the deployer.

        Will apply graphnet i3 modules to i3 files in the order in which they
        appear in graphnet_modules.Each module is run independently.

        Args:
            graphnet_modules: List of graphnet i3 modules.
                              Order of appearence in the list determines order
                              of deployment.
            gcd_file: path to gcd file.
            n_workers: Number of workers. The deployer will divide the number
                       of input files across workers. Defaults to 1.
        """
        # This makes sure that one worker cannot access more
        # than 1 core's worth of compute.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Check
        if isinstance(graphnet_modules, list):
            self._modules = graphnet_modules
        else:
            self._modules = [graphnet_modules]
        self._gcd_file = gcd_file
        self._n_workers = n_workers

    def _prepare_settings(
        self, input_files: List[str], output_folder: str
    ) -> List[Settings]:
        """Will prepare the settings for each worker."""
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            assert False, f"""{output_folder} already exists. To avoid overwriting \n
                    existing files, the process has been stopped."""
        if self._n_workers > len(input_files):
            self._n_workers = len(input_files)
        if self._n_workers > 1:
            file_batches = np.array_split(input_files, self._n_workers)
            settings: List[Settings] = []
            for i in range(self._n_workers):
                settings.append(
                    Settings(
                        file_batches[i],
                        self._gcd_file,
                        output_folder,
                        self._modules,
                    )
                )
        else:
            settings = [
                Settings(
                    input_files,
                    self._gcd_file,
                    output_folder,
                    self._modules,
                )
            ]
        return settings

    def _launch_jobs(self, settings: List[Settings]) -> None:
        """Will launch jobs in parallel if n_workers > 1, else run on main."""
        if self._n_workers > 1:
            processes = []
            for i in range(self._n_workers):
                processes.append(
                    multiprocessing.Process(
                        target=self._process_files,
                        args=[settings[i]],  # type: ignore
                    )
                )

            for process in processes:
                process.start()

            for process in processes:
                process.join()
        else:
            self._process_files(settings[0])

    def _process_files(
        self,
        settings: Settings,
    ) -> None:
        """Will start an IceTray read/write chain with graphnet modules.

        If n_workers > 1, this function is run in parallel n_worker times. Each
        worker will loop over an allocated set of i3 files. The new i3 files
        will appear as copies of the original i3 files but with reconstructions
        added. Original i3 files are left untouched.
        """
        for i3_file in settings.i3_files:
            tray = I3Tray()
            tray.context["I3FileStager"] = dataio.get_stagers()
            tray.AddModule(
                "I3Reader",
                "reader",
                FilenameList=[settings.gcd_file, i3_file],
            )
            for i3_module in settings.modules:
                tray.AddModule(i3_module)
            tray.Add(
                "I3Writer",
                Streams=[
                    icetray.I3Frame.DAQ,
                    icetray.I3Frame.Physics,
                    icetray.I3Frame.TrayInfo,
                    icetray.I3Frame.Simulation,
                ],
                filename=settings.output_folder + "/" + i3_file.split("/")[-1],
            )
            tray.Execute()
            tray.Finish()
        return

    def run(
        self,
        input_files: Union[List[str], str],
        output_folder: str,
    ) -> None:
        """Apply given graphnet modules to input files using n workers.

        The i3 files with reconstructions will appear as copies of
        the original i3 files but with reconstructions added.
        Original i3 files are left untouched.

        Args:
            input_files: Path(s) to i3 file(s) that you wish to
                         apply the graphnet modules to.
            output_folder: The output folder to which the i3 files are written.
        """
        start_time = time.time()
        if isinstance(input_files, list):
            random.shuffle(input_files)
        else:
            input_files = [input_files]
        settings = self._prepare_settings(
            input_files=input_files, output_folder=output_folder
        )
        print(
            f"""processing {len(input_files)} i3 files \n
                using {self._n_workers} workers"""
        )
        self._launch_jobs(settings)
        print(
            f"""Processing {len(input_files)} files was completed in \n
         {time.time() - start_time} seconds using {self._n_workers} cores."""
        )
