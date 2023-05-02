"""I3Extractor class(es) for extracting I3Particle properties."""

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
        """Construct I3ParticleExtractor."""
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract I3Particle properties from I3Particle in frame."""
        output = {}
        if self._name in frame:
            output.update(
                {
                    "zenith_" + self._name: frame[self._name].dir.zenith,
                    "azimuth_" + self._name: frame[self._name].dir.azimuth,
                    "dir_x_" + self._name: frame[self._name].dir.x,
                    "dir_y_" + self._name: frame[self._name].dir.y,
                    "dir_z_" + self._name: frame[self._name].dir.z,
                    "pos_x_" + self._name: frame[self._name].pos.x,
                    "pos_y_" + self._name: frame[self._name].pos.y,
                    "pos_z_" + self._name: frame[self._name].pos.z,
                    "time_" + self._name: frame[self._name].time,
                    "speed_" + self._name: frame[self._name].speed,
                    "energy_" + self._name: frame[self._name].energy,
                }
            )

        return output
