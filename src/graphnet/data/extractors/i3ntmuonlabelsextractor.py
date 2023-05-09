"""I3Extractor class(es) for extracting TUM DNN reconstruction."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3NTMuonLabelExtractor(I3Extractor):
    """Class for extracting muon labels from the Northeren Track Dataset."""

    def __init__(
        self,
        name: str = "northeren_tracks_muon_labels",
        padding_value: int = -1,
    ):
        """Construct I3NTMuonLabelExtractor."""
        # Base class constructor
        super().__init__(name)
        self._padding_value = padding_value

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract muon labels from the Northeren Track Dataset."""
        keys = [
            "classification",
            "classification_ic79",
            "classification_emuon_deposited",
            "classification_emuon_entry",
            "classification_emuon_cascade_energy",
            "classification_emuon_track_energy",
            "classification_emuon_track_length",
            "energy_on_muon_appearance",
            "ic79_energy_on_muon_appearance",
            "ic79_classification_emuon_deposited",
            "ic79_classification_emuon_entry",
            "ic79_classification_emuon_cascade_energy",
            "ic79_classification_emuon_track_energy",
            "ic79_classification_emuon_track_length",
            "classification_label",
            "classification_label_ic79",
            "coincident_muons",
            "coincident_muons_ic79",
        ]
        output = {}
        for key in keys:
            try:
                value = frame[key].value
            except KeyError:
                value = self._padding_value
            output.update(
                {
                    key: value,
                }
            )

        return output
