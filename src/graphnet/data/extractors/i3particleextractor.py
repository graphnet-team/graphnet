"""I3Extractor class(es) for extracting SplineMPE reconstruction."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3ParticleExtractor(I3Extractor):
    """Class for extracting I3Particle properties.

    Can be used to extract predictions from other algorithms for comparisons
    with GraphNeT.
    """

    def __init__(self, name: str):
        """Construct I3ComparisonExtractor."""
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract pointing predictions."""
        output = {}
        if self._name in frame:
            output.update(
                {
                    "zenith_" + self._name: frame[self._name].dir.zenith,
                    "azimuth_" + self._name: frame[self._name].dir.azimuth,
                }
            )

        return output
