"""Module containing `GraphNeTFileSaveMethod`(s).

These modules are used to save the interim data format from `DataConverter` to
a deep-learning friendly file format.
"""

import os
from typing import List, Union, OrderedDict, Any
from abc import abstractmethod, ABC

from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger


class GraphNeTFileSaveMethod(Logger, ABC):
    """Generic base class for saving interim data format in `DataConverter`.

    Classes inheriting from `GraphNeTFileSaveMethod` must implement the
    `save_file` method, which recieves the interim data format from
    from a single file.

    In addition, classes inheriting from `GraphNeTFileSaveMethod` must
    set the `file_extension` property.
    """

    @abstractmethod
    def _save_file(
        self, data: OrderedDict[str, Any], output_file_path: str
    ) -> None:
        """Save the interim data format from a single input file.

        Args:
            data: the interim data from a single input file.
            output_file_path: output file path.
        """
        return

    @final
    def __call__(
        self, data: OrderedDict[str, Any], file_name: str, out_dir: str
    ) -> None:
        """Save data.

        Args:
            data: data to be saved.
            file_name: name of input file. Will be used to generate output
                        file name.
            out_dir: directory to save data to.
        """
        output_file_path = os.path.join(
            out_dir, file_name, self.file_extension
        )
        self._save_file(data=data, output_file_path=output_file_path)
        return

    @property
    def file_extension(self) -> str:
        """Return file extension used to store the data."""
        return self._file_extension  # type: ignore
