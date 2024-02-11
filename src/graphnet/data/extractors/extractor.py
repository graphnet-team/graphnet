"""Base I3Extractor class(es)."""
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class Extractor(ABC, Logger):
    """Base class for extracting information from data files.

    All classes inheriting from `Extractor` should implement the `__call__`
    method, and should return a pure python dictionary on the form

    output = [{'var1: ..,
                 ... ,
              'var_n': ..}]

    Variables can be scalar or array-like of shape [n, 1], where n denotes the
    number of elements in the array, and 1 the number of columns.

    An extractor is used in conjunction with a specific `FileReader`.
    """

    def __init__(self, extractor_name: str):
        """Construct Extractor.

        Args:
            extractor_name: Name of the `Extractor` instance. Used to keep track of the
                provenance of different data, and to name tables to which this
                data is saved. E.g. "mc_truth".
        """
        # Member variable(s)
        self._extractor_name: str = extractor_name

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract information from frame."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the `I3Extractor` instance."""
        return self._extractor_name
