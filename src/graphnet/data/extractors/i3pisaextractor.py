"""I3Extractor class(es) for extracting quantities required by PISA."""

from typing import TYPE_CHECKING, Any, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3PISAExtractor(I3Extractor):
    """Class for extracting quantities required by PISA."""

    def __init__(self, name: str = "pisa_dependencies"):
        """Construct `I3PISAExtractor`."""
        # Base class constructor
        super().__init__(name)

    def __call__(
        self, frame: "icetray.I3Frame", padding_value: float = -1.0
    ) -> Dict[str, Any]:
        """Extract quantities required by PISA."""
        output = {}
        required_keys = ["OneWeight", "gen_ratio", "NEvents", "GENIEWeight"]
        for key in required_keys:
            output.update({key: padding_value})  # pads the entry
        if "I3MCWeightDict" in frame:
            for key in required_keys:
                try:
                    output.update(
                        {key: frame["I3MCWeightDict"][key]}
                    )  # removes the padding if value is in frame
                except KeyError:
                    pass

        return output
