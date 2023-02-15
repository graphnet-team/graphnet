"""I3Extractor class(es) for extracting SplineMPE reconstruction."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3SplineMPEICExtractor(I3Extractor):
    """Class for extracting SplineMPE pointing predictions."""

    def __init__(self, name: str = "spline_mpe_ic"):
        """Construct I3SplineMPEICExtractor."""
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract SplineMPE pointing predictions."""
        output = {}
        if "SplineMPEIC" in frame:
            output.update(
                {
                    "zenith_spline_mpe_ic": frame["SplineMPEIC"].dir.zenith,
                    "azimuth_spline_mpe_ic": frame["SplineMPEIC"].dir.azimuth,
                }
            )

        return output


class I3ComparisonExtractor(I3Extractor):
    """Class for extracting pointing predictions from various algorithms."""

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
