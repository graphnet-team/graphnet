from graphnet.data.extractors.i3extractor import I3Extractor


class I3TUMExtractor(I3Extractor):
    def __init__(self, name="tum_dnn"):
        super().__init__(name)

    def __call__(self, frame) -> dict:
        """Extracts TUM DNN Recos and associated variables"""
        output = {}
        if "TUM_dnn_energy_hive" in frame:
            output.update(
                {
                    "tum_dnn_energy_hive": 10
                    ** frame["TUM_dnn_energy_hive"]["mu_E_on_entry"],
                    "tum_dnn_energy_dst": 10
                    ** frame["TUM_dnn_energy_dst"]["mu_E_on_entry"],
                    "tum_bdt_sigma": frame["TUM_bdt_sigma"].value,
                }
            )

        return output
