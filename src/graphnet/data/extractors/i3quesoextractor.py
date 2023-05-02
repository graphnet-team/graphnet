"""Extract Event Selection Labels from the QUESO event selection."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.i3extractor import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3QUESOExtractor(I3Extractor):
    """Class for extracting labels from the QUESO event selection."""

    def __init__(
        self,
        name: str = "queso",
        padding_value: int = -1,
    ):
        """Construct I3QUESOExtractor."""
        # Base class constructor
        super().__init__(name)
        self._padding_value = padding_value

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract Event Selection Labels from the QUESO event selection."""
        keys = [
            "QuesoL3_Bool",
            "QuesoL3_Vars_cleaned_length",
            "QuesoL3_Vars_cleaned_num_hit_modules",
            "QuesoL3_Vars_cleaned_num_hits_fid_vol",
            "QuesoL3_Vars_cleaned_vertexZ",
            "QuesoL3_Vars_uncleaned_length",
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
