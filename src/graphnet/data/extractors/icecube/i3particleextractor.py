"""I3Extractor class(es) for extracting I3Particle properties."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.icecube import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3ParticleExtractor(I3Extractor):
    """Class for extracting I3Particle properties.

    Can be used to extract predictions from other algorithms for comparisons
    with GraphNeT.
    """

    def __init__(self, extractor_name: str, exclude: list = [None]):
        """Construct I3ParticleExtractor."""
        # Base class constructor
        super().__init__(extractor_name=extractor_name, exclude=exclude)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract I3Particle properties from I3Particle in frame."""
        output = {}
        name = self._extractor_name
        if name in frame:
            output.update(
                {
                    "zenith_" + name: frame[name].dir.zenith,
                    "azimuth_" + name: frame[name].dir.azimuth,
                    "dir_x_" + name: frame[name].dir.x,
                    "dir_y_" + name: frame[name].dir.y,
                    "dir_z_" + name: frame[name].dir.z,
                    "pos_x_" + name: frame[name].pos.x,
                    "pos_y_" + name: frame[name].pos.y,
                    "pos_z_" + name: frame[name].pos.z,
                    "time_" + name: frame[name].time,
                    "speed_" + name: frame[name].speed,
                    "energy_" + name: frame[name].energy,
                }
            )

        return output
