"""I3Extractor class(es) for extracting RETRO reconstruction."""

from typing import TYPE_CHECKING, Any, Dict

from graphnet.data.extractors.i3extractor import I3Extractor
from graphnet.data.extractors.utilities.frames import (
    frame_is_montecarlo,
    frame_is_noise,
)

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3RetroExtractor(I3Extractor):
    """Class for extracting RETRO reconstruction."""

    def __init__(self, name: str = "retro"):
        """Construct `I3RetroExtractor`."""
        # Base class constructor
        super().__init__(name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract RETRO reconstruction and associated quantities."""
        output = {}

        if self._frame_contains_retro(frame):
            output.update(
                {
                    "azimuth_retro": frame["L7_reconstructed_azimuth"].value,
                    "time_retro": frame["L7_reconstructed_time"].value,
                    "energy_retro": frame[
                        "L7_reconstructed_total_energy"
                    ].value,
                    "position_x_retro": frame[
                        "L7_reconstructed_vertex_x"
                    ].value,
                    "position_y_retro": frame[
                        "L7_reconstructed_vertex_y"
                    ].value,
                    "position_z_retro": frame[
                        "L7_reconstructed_vertex_z"
                    ].value,
                    "zenith_retro": frame["L7_reconstructed_zenith"].value,
                    "azimuth_sigma": frame[
                        "L7_retro_crs_prefit__azimuth_sigma_tot"
                    ].value,
                    "position_x_sigma": frame[
                        "L7_retro_crs_prefit__x_sigma_tot"
                    ].value,
                    "position_y_sigma": frame[
                        "L7_retro_crs_prefit__y_sigma_tot"
                    ].value,
                    "position_z_sigma": frame[
                        "L7_retro_crs_prefit__z_sigma_tot"
                    ].value,
                    "time_sigma": frame[
                        "L7_retro_crs_prefit__time_sigma_tot"
                    ].value,
                    "zenith_sigma": frame[
                        "L7_retro_crs_prefit__zenith_sigma_tot"
                    ].value,
                    "energy_sigma": frame[
                        "L7_retro_crs_prefit__energy_sigma_tot"
                    ].value,
                    "cascade_energy_retro": frame[
                        "L7_reconstructed_cascade_energy"
                    ].value,
                    "track_energy_retro": frame[
                        "L7_reconstructed_track_energy"
                    ].value,
                    "track_length_retro": frame[
                        "L7_reconstructed_track_length"
                    ].value,
                }
            )

        if self._frame_contains_classifiers(frame):
            classifiers = [
                "L7_MuonClassifier_FullSky_ProbNu",
                "L4_MuonClassifier_Data_ProbNu",
                "L4_NoiseClassifier_ProbNu",
                "L7_PIDClassifier_FullSky_ProbTrack",
            ]
            for classifier in classifiers:
                if classifier in frame:
                    output.update({classifier: frame[classifier].value})

        if frame_is_montecarlo(frame):
            if frame_is_noise(frame):
                output.update(
                    {
                        "osc_weight": frame["noise_weight"]["weight"],
                    }
                )
            else:
                output["osc_weight"] = self._try_get_key(
                    frame["I3MCWeightDict"], "weight", default_value=-1
                )

        return output

    def _frame_contains_retro(self, frame: "icetray.I3Frame") -> bool:
        return "L7_reconstructed_zenith" in frame

    def _frame_contains_classifiers(self, frame: "icetray.I3Frame") -> bool:
        return "L4_MuonClassifier_Data_ProbNu" in frame

    def _try_get_key(
        self, frame: "icetray.I3Frame", key: str, default_value: int = -1
    ) -> Any:
        """Return `key` in `frame` if it exists; otherwise `default_value`."""
        try:
            return frame[key]
        except KeyError:
            return default_value
