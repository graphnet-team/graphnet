from typing import Optional, Tuple

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import get_logger

logger = get_logger()

if has_icecube_package():
    from icecube import (
        dataclasses,
        icetray,
    )  # pyright: reportMissingImports=false


def frame_is_montecarlo(frame: "icetray.I3Frame") -> bool:
    return ("MCInIcePrimary" in frame) or ("I3MCTree" in frame)


def frame_is_noise(frame: "icetray.I3Frame") -> bool:
    try:
        frame["I3MCTree"][0].energy
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
) -> Tuple:
    """Gets the indicies for the gcd_dict and the pulse series

    Args:
        frame (i3 physics frame): i3 physics frame

    Returns:
        om_keys (index): the indicies for the gcd_dict
        data    (??)   : the pulse series
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
