"""Base I3Extractor class(es)."""

from typing import Any, Union, Callable
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

    def __init__(self, extractor_name: str, exclude: list = [None]):
        """Construct Extractor.

        Args:
            extractor_name: Name of the `Extractor` instance.
                            Used to keep track of the provenance of different
                            data, and to name tables to which this data is
                            saved. E.g. "mc_truth".
            exclude: List of keys to exclude from the extracted data.
        """
        # Member variable(s)
        self._extractor_name: str = extractor_name
        self._exclude = exclude

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def exclude(func: Callable) -> Callable:
        """Exclude specified keys from the extracted data."""

        def wrapper(
            self: "Extractor", *args: Any
        ) -> Union[dict, pd.DataFrame]:
            result = func(self, *args)
            if isinstance(result, dict):
                for key in self._exclude:
                    if key in result:
                        del result[key]
            elif isinstance(result, pd.DataFrame):
                for key in self._exclude:
                    if key in result.columns:
                        result = result.drop(columns=[key])
            return result

        return wrapper

    @abstractmethod
    def __call__(self, data: Any) -> Union[dict, pd.DataFrame]:
        """Extract information from data."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the `Extractor` instance."""
        return self._extractor_name

    def __init_subclass__(cls) -> None:
        """Initialize subclass and apply the exclude decorator to __call__."""
        super().__init_subclass__()
        if cls._get_root_logger().getEffectiveLevel() > 10:
            cls.__call__ = cls.exclude(cls.__call__)  # type: ignore
        else:
            import time

            cls.__call__ = cls.exclude(cls.__call__)  # type: ignore
            # wrap with time logging
            original_call = cls.__call__

            def timed_call(
                cls: "Extractor", *args: Any
            ) -> Union[dict, pd.DataFrame]:
                start_time = time.time()
                result = original_call(cls, *args)
                end_time = time.time()
                cls._logger.debug(
                    f"Extractor {cls.name} took {end_time - start_time:.4f} seconds."
                )
                return result

            cls.__call__ = timed_call  # type: ignore
