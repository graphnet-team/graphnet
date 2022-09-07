from graphnet.data.extractors.i3extractor import I3Extractor
import numpy as np


class I3SplineMPEICExtractor(I3Extractor):
    def __init__(self, name="spline_mpe_ic"):
        super().__init__(name)

    def __call__(self, frame) -> dict:
        """Extracts TUM DNN Recos and associated variables"""
        output = {}
        if self._frame_contains_retro(frame):
            output.update(
                {
                    "zenith_spline_mpe_ic": frame[
                        "SplineMPEIC"
                    ].dir.zenith.value,
                    "azimuth_spline_mpe_ic": frame[
                        "SplineMPEIC"
                    ].dir.zenith.value,
                    "pbfErr1_spline_mpe_ic": frame[
                        "SplineMPEICParaboloidFitParams"
                    ].pbfErr1.value,
                }
            )

        return output
