"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List
from graphnet.data.extractors.i3extractor import I3Extractor
from graphnet.data.extractors.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
    )  # pyright: reportMissingImports=false


class I3FeatureExtractor(I3Extractor):
    """Base class for extracting specific, reconstructed features."""

    def __init__(self, pulsemap: str):
        """Construct I3FeatureExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
        """
        # Member variable(s)
        self._pulsemap = pulsemap

        # Base class constructor
        super().__init__(pulsemap)


class I3FeatureExtractorIceCube86(I3FeatureExtractor):
    """Class for extracting reconstructed features for IceCube-86."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "charge": [],
            "dom_time": [],
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_area": [],
            "rde": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            "event_time": [],
            "hlc": [],
            "awtd": [],
            "fadc": [],
        }

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # Added these :
        bright_doms = None
        bad_doms = None
        saturation_windows = None
        calibration_errata = None
        if "BrightDOMs" in frame:
            bright_doms = frame.Get("BrightDOMs")

        if "BadDomsList" in frame:
            bad_doms = frame.Get("BadDomsList")

        if "SaturationWindows" in frame:
            saturation_windows = frame.Get("SaturationWindows")

        if "CalibrationErrata" in frame:
            calibration_errata = frame.Get("CalibrationErrata")

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            # DOM flags
            if bright_doms:
                is_bright_dom = 1 if om_key in bright_doms else 0
            else:
                is_bright_dom = int(padding_value)

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)

            if saturation_windows:
                is_saturated_dom = 1 if om_key in saturation_windows else 0
            else:
                is_saturated_dom = int(padding_value)

            if calibration_errata:
                is_errata_dom = 1 if om_key in calibration_errata else 0
            else:
                is_errata_dom = int(padding_value)

            # Loop over pulses for each OM
            pulses = data[om_key]
            for pulse in pulses:
                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )
                output["dom_time"].append(
                    getattr(pulse, "time", padding_value)
                )
                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                # DOM flags
                output["is_bright_dom"].append(is_bright_dom)
                output["is_bad_dom"].append(is_bad_dom)
                output["is_saturated_dom"].append(is_saturated_dom)
                output["is_errata_dom"].append(is_errata_dom)
                output["event_time"].append(event_time)
                # Pulse flags
                output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                output["awtd"].append((pulse.flags >> 1) & 0x1)  # bit 1
                output["fadc"].append((pulse.flags >> 2) & 0x1)  # bit 2

        return output

    def _get_relative_dom_efficiency(
        self, frame: "icetray.I3Frame", om_key: int, padding_value: float
    ) -> float:
        if (
            "I3Calibration" in frame
        ):  # Not available for e.g. mDOMs in IceCube Upgrade
            rde = frame["I3Calibration"].dom_cal[om_key].relative_dom_eff
        else:
            try:
                assert self._calibration is not None
                rde = self._calibration.dom_cal[om_key].relative_dom_eff
            except:  # noqa: E722
                rde = padding_value
        return rde


class I3FeatureExtractorIceCubeDeepCore(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for IceCube-DeepCore."""


class I3FeatureExtractorIceCubeUpgrade(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for IceCube-Upgrade."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "pmt_dir_x": [],
            "pmt_dir_y": [],
            "pmt_dir_z": [],
            "dom_type": [],
        }

        # Add features from IceCube86
        output_icecube86 = super().__call__(frame)
        output.update(output_icecube86)

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Common values for each OM
            pmt_dir_x = self._gcd_dict[om_key].orientation.x
            pmt_dir_y = self._gcd_dict[om_key].orientation.y
            pmt_dir_z = self._gcd_dict[om_key].orientation.z
            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # Loop over pulses for each OM
            pulses = data[om_key]
            for _ in pulses:
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["pmt_dir_x"].append(pmt_dir_x)
                output["pmt_dir_y"].append(pmt_dir_y)
                output["pmt_dir_z"].append(pmt_dir_z)
                output["dom_type"].append(dom_type)

        return output


class I3PulseNoiseTruthFlagIceCubeUpgrade(I3FeatureExtractorIceCube86):
    """Feature extractor class with pulse noise truth flag added."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "truth_flag": [],
        }

        # Add features from IceCubeUpgrade
        output_icecube_upgrade = super().__call__(frame)
        output.update(output_icecube_upgrade)

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Loop over pulses for each OM
            pulses = data[om_key]
            for truth_flag in pulses:
                output["truth_flag"].append(truth_flag)

        return output
