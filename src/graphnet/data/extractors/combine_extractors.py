"""Module for combining multiple extractors into a single extractor."""

from typing import TYPE_CHECKING

from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube.i3extractor import I3Extractor
from typing import List, Dict

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class CombinedExtractor(I3Extractor):
    """Class for combining multiple extractors.

    This class is used to combine multiple extractors into a single extractor
    with a new name.
    """

    def __init__(self, extractors: List[I3Extractor], extractor_name: str):
        """Construct CombinedExtractor.

        Args:
        extractors: List of extractors to combine.
                    The extractors must all return data on the same level;
                    e.g. all event-level data or pulse-level data.
                    Mixing tables that contain event-level and
                    pulse-level information will fail.
        extractor_name: Name of the new extractor.
        """
        super().__init__(extractor_name=extractor_name)
        self._extractors = extractors

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract data from frame using all extractors.

        Args:
        frame: I3Frame to extract data from.
        """
        output = {}
        for extractor in self._extractors:
            output.update(extractor(frame))
        return output
