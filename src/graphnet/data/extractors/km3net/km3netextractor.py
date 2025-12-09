"""Base class for all KM3NeTROOT extractors."""

from typing import Any
from abc import abstractmethod

from graphnet.data.extractors import Extractor

# needs to be implemented at the end. It is a class that will kind of
# gather all the specific extractors for the different data types and
# help to call them all from the reader. Equivalent to the I3extractor (that
# I don't yet understand) in the IceCube example


class KM3NeTExtractor(Extractor):
    """Base class for all KM3NeT extractors."""

    def __init__(self, extractor_name: str):
        """Initailize KM3NeTTExtractor.

        Args:
            extractor_name: Name of the `KM3NeTExtractor` instance.
            Used to keep track of the provenance of different data,
            and to name tables to which this data is saved.
        """
        super().__init__(extractor_name=extractor_name)

    @abstractmethod
    def __call__(self, file: Any) -> dict:
        """Extract information from file."""
        pass
