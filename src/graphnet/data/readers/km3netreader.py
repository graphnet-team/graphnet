"""Module for reading KM3NeT files."""

from typing import TYPE_CHECKING, Union, List, Any, Dict


from graphnet.data.readers import GraphNeTFileReader
from graphnet.utilities.imports import has_km3net_package
from graphnet.data.extractors.km3net import (
    KM3NeTTruthExtractor,
    KM3NeTFullPulseExtractor,
    KM3NeTTriggPulseExtractor,
    KM3NeTHNLTruthExtractor,
    KM3NeTRegularRecoExtractor,
    KM3NeTHNLRecoExtractor,
)


# km3net specific imports
if has_km3net_package() or TYPE_CHECKING:
    import km3io as ki # pyright: reportMissingImports=false


class KM3NeTReader(GraphNeTFileReader):
    """Class for reading KM3NeT files."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [
        KM3NeTTruthExtractor,
        KM3NeTFullPulseExtractor,
        KM3NeTTriggPulseExtractor,
        KM3NeTHNLTruthExtractor,
        KM3NeTRegularRecoExtractor,
        KM3NeTHNLRecoExtractor,
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
        if len(file.trks) > 0:
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
