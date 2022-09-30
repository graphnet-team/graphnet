from graphnet.data.extractors.i3extractor import I3Extractor


class I3SplineMPEICExtractor(I3Extractor):
    def __init__(self, name="spline_mpe_ic"):
        super().__init__(name)

    def __call__(self, frame) -> dict:
        """Extracts SplineMPE pointing predictions."""
        output = {}
        if "SplineMPEIC" in frame:
            output.update(
                {
                    "zenith_spline_mpe_ic": frame["SplineMPEIC"].dir.zenith,
                    "azimuth_spline_mpe_ic": frame["SplineMPEIC"].dir.azimuth,
                }
            )

        return output
