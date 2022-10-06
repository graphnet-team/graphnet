from graphnet.data.extractors.i3extractor import I3Extractor


class I3GalacticPlaneHybridRecoExtractor(I3Extractor):
    def __init__(self, name="dnn_hybrid"):
        super().__init__(name)

    def __call__(self, frame) -> dict:
        """Extracts TUMs DNN Recos and associated variables"""
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
