"""Contains an IceCube-specific implementation of Deployer."""

from typing import TYPE_CHECKING, List, Union, Sequence
import os
import numpy as np

from graphnet.utilities.imports import has_icecube_package
from graphnet.deployment.icecube import I3InferenceModule
from graphnet.data.dataclasses import Settings
from graphnet.deployment import Deployer

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
    from icecube.icetray import I3Tray


class I3Deployer(Deployer):
    """A generic baseclass for applying `DeploymentModules` to analysis files.

    Modules are applied in the order that they appear in `modules`.
    """

    def __init__(
        self,
        modules: Union[I3InferenceModule, Sequence[I3InferenceModule]],
        gcd_file: str,
        n_workers: int = 1,
    ) -> None:
        """Initialize `Deployer`.

        Will apply `DeploymentModules` to files in the order in which they
        appear in `modules`. Each module is run independently.

        Args:
            modules: List of `DeploymentModules`.
                              Order of appearence in the list determines order
                              of deployment.
            gcd_file: path to gcd file.
            n_workers: Number of workers. The deployer will divide the number
                       of input files across workers. Defaults to 1.
        """
        super().__init__(modules=modules, n_workers=n_workers)

        # Member variables
        self._gcd_file = gcd_file

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

    def _prepare_settings(
        self, input_files: List[str], output_folder: str
    ) -> List[Settings]:
        """Will prepare the settings for each worker."""
        try:
            os.makedirs(output_folder)
        except FileExistsError as e:
            self.error(
                f"{output_folder} already exists. To avoid overwriting "
                "existing files, the process has been stopped."
            )
            raise e
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
