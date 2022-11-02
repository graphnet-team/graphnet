from graphnet.data.extractors.i3extractor import I3Extractor


class I3PISAExtractor(I3Extractor):
    def __init__(self, name="pisa_dependencies"):
        super().__init__(name)

    def __call__(self, frame, padding_value=-1) -> dict:
        """Extracts quantities required by PISA"""
        output = {}
        required_keys = ["OneWeight", "gen_ratio", "NEvents", "GENIEWeight"]
        if "I3MCWeightDict" in frame:
            for key in required_keys:
                try:
                    output.update({key: frame["I3MCWeightDict"][key]})
                except KeyError:
                    output.update({key: padding_value})

        return output
