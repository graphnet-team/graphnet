"""I3Extractor class for extracting features saved in I3 in a dict format."""

from typing import TYPE_CHECKING, Any, Dict, List
from .i3extractor import I3Extractor
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3DictValueExtractor(I3Extractor):
    """Extracts a dictionary of values from an I3Frame."""

    def __init__(
        self, keys: List[str], extractor_name: str = "I3DictValueExtractor"
    ) -> None:
        """Construct I3DictValueExtractor.

        Args:
            keys: List of keys to extract from the I3Frame.
            extractor_name: Name of the extractor.
        """
        super().__init__(extractor_name=extractor_name)
        self._keys = keys

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract values from the I3Frame.

        Args:
            frame: The I3Frame to extract values from.

        Returns:
            A dictionary of all values extracted from the frame[key].
        """
        output = {}
        for key in self._keys:
            if key not in frame:
                raise KeyError(f"Key {key} not found in frame.")
            items = frame[key].keys()
            for item in items:
                output[item] = frame[key][item]
        return output
