"""I3Extractor class(es) for extracting hybrid reconstructions."""
from graphnet.data.extractors.i3extractor import I3Extractor


class I3GalacticPlaneHybridRecoExtractor(I3Extractor):
    """Class for extracting galatictic plane hybrid reconstruction."""

    def __init__(self, name: str = "dnn_hybrid"):
        """Construct instance.

        Args:
            name (str, optional): Name of the `I3Extractor` instance. Defaults
                to "dnn_hybrid".
        """
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame) -> dict:
        """Extract TUMs DNN reconcstructions and associated variables."""
        output = {}
        if "DNNCascadeAnalysis_version_001_p00" in frame:
            reco_object = frame["DNNCascadeAnalysis_version_001_p00"]
            keys = [
                "angErr",
                "angErr_uncorrected",
                "dec",
                "dpsi",
                "energy",
                "event",
                "ra",
                "run",
                "subevent",
                "time",
                "trueDec",
                "trueE",
                "trueRa",
                "true_azi",
                "true_zen",
            ]
            for key in keys:
                output.update({key: reco_object[key]})
            output.update(
                {
                    "zenith_hybrid": reco_object["zen"],
                    "azimuth_hybrid": reco_object["azi"],
                    "energy_hybrid_log": reco_object["logE"],
                }
            )

        return output
