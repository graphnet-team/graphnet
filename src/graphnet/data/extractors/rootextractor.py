"""Base I3Extractor class(es)."""

from abc import ABC, abstractmethod
from typing import List

from graphnet.utilities.logging import Logger

class rootExtractor(ABC, Logger):
    """Base class for extracting information from physics I3-frames.

    All classes inheriting from `rootExtractor` should implement the `__call__`
    method, and can be applied directly on `icetray.I3Frame` objects to return
    extracted, pure-python data.
    """

    def __init__(self, name: str):
        """Construct rootExtractor.

        Args:
            name: Name of the `rootExtractor` instance. Used to keep track of the
                provenance of different data, and to name tables to which this
                data is saved.
        """
        # Member variable(s)
        self._root_file: str = ""
        self._name: str = name

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def set_files(self, root_file: str) -> None:
        """Store references to the root-files being processed."""
        # @TODO: Is it necessary to set the `root_file`? It is only used in one
        #        place in `I3TruthExtractor`, and there only in a way that might
        #        be solved another way.
        self._root_file = root_file

    @abstractmethod
    def __call__(self, events: "root.events", index: int) -> dict:
        """Extract information from frame."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the `rootExtractor` instance."""
        return self._name


class rootExtractorCollection(list):
    """Class to manage multiple rootExtractors."""

    def __init__(self, *extractors: rootExtractor):
        """Construct rootExtractorCollection.

        Args:
            *extractors: List of `rootExtractor`s to be treated as a single
            collection.
        """
        # Check(s)
        for extractor in extractors:
            assert isinstance(extractor, rootExtractor)

        # Base class constructor
        super().__init__(extractors)

    def set_files(self, root_file: str) -> None:
        """Store references to the root-files being processed."""
        for extractor in self:
            extractor.set_files(root_file)

    def __call__(self, events: "root.events", index: int) -> List[dict]:
        """Extract information from events for each member `rootExtractor`."""
        return [extractor(events, index) for extractor in self]
