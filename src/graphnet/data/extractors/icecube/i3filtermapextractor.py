"""I3Extractor for extracting the boolean condition of the I3FilterMask."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.icecube import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3FilterMapExtractor(I3Extractor):
    """Class for extracting I3FilterMap properties.

    Can be used to extract predictions from other algorithms for comparisons
    with GraphNeT.
    """

    def __init__(
        self,
        key: str = "FilterMask",
        extractor_name: str = "I3FilterMap",
        exclude: list = [None],
    ) -> None:
        """Construct I3FilterMapExtractor."""
        # Base class constructor
        super().__init__(extractor_name=extractor_name, exclude=exclude)
        self._key = key

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, bool]:
        """Extract I3FilterMap properties from I3FilterMap in frame."""
        output = {}
        if self._key not in frame:
            raise KeyError(f"Key {self._key} not found in frame.")
        items = frame[self._key].items()
        for item in items:
            output[item[0]] = item[1].condition_passed
        return output
