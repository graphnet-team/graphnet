"""Module containing `GraphNeTFileSaveMethod`(s).

These modules are used to save the interim data format from `DataConverter` to
a deep-learning friendly file format.
"""

import os
from typing import Dict, List, Union
from abc import abstractmethod, ABC

from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger

import pandas as pd


class GraphNeTWriter(Logger, ABC):
    """Generic base class for saving interim data format in `DataConverter`.

    Classes inheriting from `GraphNeTFileSaveMethod` must implement the
    `save_file` method, which recieves the interim data format from
    from a single file.

    In addition, classes inheriting from `GraphNeTFileSaveMethod` must
    set the `file_extension` property. What
    """

    @abstractmethod
    def _save_file(
        self,
        data: Union[Dict[str, pd.DataFrame], Dict[str, List[pd.DataFrame]]],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save the interim data format from a single input file.

        Args:
            data: the interim data from a single input file.
            output_file_path: output file path.
            n_events: Number of events container in `data`.
        """
        raise NotImplementedError

    @abstractmethod
    def merge_files(
        self,
        files: List[str],
        output_dir: str,
    ) -> None:
        """Merge smaller files.

        Args:
            files: Files to be merged.
            output_dir: The directory to store the merged files in.
        """
        raise NotImplementedError

    @final
    def __call__(
        self,
        data: Union[Dict[str, pd.DataFrame], Dict[str, List[pd.DataFrame]]],
        file_name: str,
        output_dir: str,
        n_events: int,
    ) -> None:
        """Save data.

        Args:
            data: data to be saved.
            file_name: name of input file. Will be used to generate output
                        file name.
            output_dir: directory to save data to.
            n_events: Number of events in `data`.
        """
        # make dir
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = (
            os.path.join(output_dir, file_name) + self.file_extension
        )

        self._save_file(
            data=data, output_file_path=output_file_path, n_events=n_events
        )
        return

    @property
    def file_extension(self) -> str:
        """Return file extension used to store the data."""
        return self._file_extension  # type: ignore

    @property
    def expects_merged_dataframes(self) -> bool:
        """Return if writer expects input to be merged dataframes or not."""
        return self._merge_dataframes  # type: ignore
