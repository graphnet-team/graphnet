"""Utility methods for working with I3Frames."""

from typing import Any, Optional, Tuple

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import (
        dataclasses,
        icetray,
    )  # pyright: reportMissingImports=false


def frame_is_montecarlo(
    frame: "icetray.I3Frame", mctree: Optional[str] = "I3MCTree"
) -> bool:
    """Check whether `frame` is from Monte Carlo simulation."""
    return ("MCInIcePrimary" in frame) or (mctree in frame)


def frame_is_noise(
    frame: "icetray.I3Frame", mctree: Optional[str] = "I3MCTree"
) -> bool:
    """Check whether `frame` is from noise."""
    try:
        frame[mctree][0].energy
        return False
    except:  # noqa: E722
        try:
            frame["MCInIcePrimary"].energy
            return False
        except:  # noqa: E722
            return True


def get_om_keys_and_pulseseries(
    frame: "icetray.I3Frame",
    pulseseries: str,
    calibration: Optional["dataclasses.I3Calibration"] = None,
) -> Tuple[Any, Any]:
    """Get the indicies for the gcd_dict and the pulse series.

    Args:
        frame: Physics (P) I3-frame from which to extract OM keys and pulse
            series

    Returns:
        Tuple containing the OM keys/indicesfor the GCD dictionary, and the
        pulse series data.
    """
    try:
        data = frame[pulseseries]
    except KeyError:
        raise KeyError(f"Pulse series {pulseseries} does not exist in `frame`")

    try:
        om_keys = data.keys()
    except AttributeError:

        try:
            if "I3Calibration" in frame.keys():
                data = frame[pulseseries].apply(frame)
                om_keys = data.keys()
            else:
                if calibration is None:
                    raise ValueError(
                        "Need I3Calibration object for pulse series "
                        f"{pulseseries}."
                    )

                frame["I3Calibration"] = calibration
                data = frame[pulseseries].apply(frame)
                om_keys = data.keys()

                # Avoid adding unneccesary data to frame
                del frame["I3Calibration"]

        except:  # noqa: E722
            data = dataclasses.I3RecoPulseSeriesMap.from_frame(
                frame, pulseseries
            )
            om_keys = data.keys()

    return om_keys, data
