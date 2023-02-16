"""Base class for I3 generators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from collections import OrderedDict


from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import LoggerMixin
from copy import deepcopy

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class Generator(ABC, LoggerMixin):
    """Base class for generating additional data from Frames.

    All classes inheriting from `Generator` should implement the `__call__`
    method, and can be applied directly on an OrderedDict generated from an
    icetray.I3Frame objects to return generated table.
    """

    def __init__(self, name: str):
        """Construct Generator.

        Args:
            name: Name of the `Generator` instance. Used to keep track of the
                provenance of different data, and to name tables to which this
                data is saved.
        """
        # Member variable(s)
        self._name: str = name

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> dict:
        """Return ordered dict with generated features."""
        pass

    @property
    def name(self) -> str:
        """Get name of generator instance."""
        return self._name


class GeneratorCollection(list):
    """Collection of Generators, for generating additional data from Frames."""

    def __init__(self, *generators: Generator) -> None:
        """Construct GeneratorCollection."""
        for generator in generators:
            assert isinstance(
                generator, Generator
            ), "All generators must be of type Generator"

        super().__init__(generators)

    def __call__(self, data: OrderedDict) -> OrderedDict:
        """Update input dict with generated features."""
        tmp_data = deepcopy(data)
        for generator in self:
            data.update(generator(tmp_data))
        return data
