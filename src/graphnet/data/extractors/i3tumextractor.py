"""I3Extractor class(es) for extracting TUM DNN reconstruction."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3TUMExtractor(I3Extractor):
    """Class for extracting TUM DNN predictions."""

    def __init__(self, name: str = "tum_dnn"):
        """Construct I3TUMExtractor."""
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract TUM DNN recoconstruction and associated variables."""
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
