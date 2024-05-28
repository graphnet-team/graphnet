"""Module for reading KM3NeTROOT files."""

from typing import Union, List, OrderedDict, Any, Dict


from graphnet.data.readers import GraphNeTFileReader
from graphnet.data.extractors.km3net import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net import (
    KM3NeTROOTTruthExtractor,
    KM3NeTROOTPulseExtractor,
    KM3NeTROOTTriggPulseExtractor,
)


# km3net specific imports
import km3io as ki


class KM3NeTROOTReader(GraphNeTFileReader):
    """Class for reading KM3NeTROOT files."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [
        KM3NeTROOTTruthExtractor,
        KM3NeTROOTPulseExtractor,
        KM3NeTROOTTriggPulseExtractor,
    ]

    def __call__(
        self, file_path: Union[str]
    ) -> Dict[str, Union[Dict[Any, Any], Any]]:
        """Open and apply extractors to a single root file.

        Args:
           file_path: The path to the file to be read.

        Returns:
              data in a list of ordered dataframes with a unique ID.
        """
        file = ki.OfflineReader(file_path)
        if len(file.mc_trks[:, 0]) > 0:
            data = {}
            for extractor in self._extractors:
                data[extractor._extractor_name] = extractor(
                    file
                )  # extractor returns dataframe

            return data
        else:
            print(f"File {file_path} has no events.")
            return {}

    def find_files(
        self, path: Union[str, List[str]]
    ) -> Union[List[str], List[Any]]:
        """Find all files in a directory with the correct extension.

        Args:
            path: The path to the directory to be searched.

        Returns:
            A list of file paths.
        """
        return super().find_files(path)

    