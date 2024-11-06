"""Base I3Extractor class(es)."""

from typing import Any, Union
from abc import ABC, abstractmethod
import pandas as pd

from graphnet.utilities.logging import Logger


class Extractor(ABC, Logger):
    """Base class for extracting information from data files.

    All classes inheriting from `Extractor` should implement the `__call__`
    method, and should return a pure python dictionary on the form

    output = {'var1: ..,
                 ... ,
              'var_n': ..}

    Variables can be scalar or array-like of shape [n, 1], where n denotes the
    number of elements in the array, and 1 the number of columns.

    An extractor is used in conjunction with a specific `FileReader`.
    """

    def __init__(self, extractor_name: str):
        """Construct Extractor.

        Args:
            extractor_name: Name of the `Extractor` instance.
                            Used to keep track of the provenance of different
                            data, and to name tables to which this data is
                            saved. E.g. "mc_truth".
        """
        # Member variable(s)
        self._extractor_name: str = extractor_name

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def __call__(self, data: Any) -> Union[dict, pd.DataFrame]:
        """Extract information from data."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the `Extractor` instance."""
        return self._extractor_name
