"""I3Extractor class(es) for extracting hybrid reconstructions."""

from typing import TYPE_CHECKING, Any, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3GalacticPlaneHybridRecoExtractor(I3Extractor):
    """Class for extracting galatictic plane hybrid reconstruction."""

    def __init__(self, name: str = "dnn_hybrid"):
        """Construct I3GalacticPlaneHybridRecoExtractor."""
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
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
