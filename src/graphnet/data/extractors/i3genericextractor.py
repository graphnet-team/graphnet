import inspect
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, OrderedDict
from graphnet.data.extractors.i3extractor import I3Extractor
from graphnet.utilities.logging import get_logger

logger = get_logger()
try:
    from icecube import (
        dataclasses,
        icetray,
    )  # pyright: reportMissingImports=false
except ImportError:
    logger.warning("icecube package not available.")


# Utility class(es)
class NonFlattenableDict(dict):
    pass


# Utility function(s)
def is_boost_enum(obj: Any) -> bool:
    """Check whether `obj` inherits from Boost.Python.enum."""
    for cls in type(obj).__bases__:
        if "Boost.Python.enum" in str(cls):
            return True
    return False


def is_icecube_class(obj: Any) -> bool:
    classname = str(type(obj))
    return "icecube." in classname


def is_type(obj: Any) -> bool:
    """Checks whether `obj` is a type, and not an instance."""
    return type(obj).__name__ == "type"


def is_method(obj: Any) -> bool:
    return inspect.ismethod(obj) or "Boost.Python.function" in str(type(obj))


def flatten_nested_dictionary(
    results: Dict, parent_key: str = "", separator: str = "."
) -> Dict:
    """Turn nested dictionary into flat one, with nested key seperated by `separator`."""
    if not isinstance(results, dict):
        return results

    items = []
    for key, value in results.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping) and not isinstance(
            value, NonFlattenableDict
        ):
            items.extend(
                flatten_nested_dictionary(
                    value, new_key, separator=separator
                ).items()
            )
        else:
            items.append((new_key, value))
    return dict(items)


class I3GenericExtractor(I3Extractor):
    """Dynamically and generically extract all information from frames."""

    def __init__(self):
        # Reference to frame currently being processed
        self._frame: Optional[icetray.I3Frame] = None

        super().__init__("Generic")

    @property
    def frame(self):
        if self._frame is None:
            raise ValueError("No I3Frame currently being processed.")
        return self._frame

    def _get_member_variables(
        self, obj: Any, return_discarded: bool = False
    ) -> List[str]:
        """Returns list of valid member variables.

        Ignoring mangled (__*) variables, types, methods, and Boost enums.
        """
        valid_member_variables = []
        discarded_member_variables = {
            "mangled": [],
            "is_type": [],
            "is_method": [],
            "is_boost_enum": [],
        }
        for attr in dir(obj):
            if attr.startswith("__"):
                discarded_member_variables["mangled"].append(attr)
                continue

            value = getattr(obj, attr)

            if is_type(value):
                discarded_member_variables["is_type"].append(attr)
                continue
            if is_method(value):
                discarded_member_variables["is_method"].append(attr)
                continue
            if is_boost_enum(value):
                discarded_member_variables["is_boost_enum"].append(attr)
                continue
            valid_member_variables.append(attr)

        if return_discarded:
            return valid_member_variables, discarded_member_variables

        return valid_member_variables

    def _extract(self, obj: Any) -> Any:
        """Extracts all possible pure-python data from `obj`."""

        self.logger.debug(f"Value: {obj}")
        self.logger.debug(f"Type: {str(type(obj))}")

        if not is_icecube_class(obj):
            self.logger.debug("Found non-I3 class. Exiting.")
            if isinstance(obj, (list, tuple, set)):
                return [self._extract(element) for element in obj]
            elif isinstance(obj, dict):
                return {
                    str(key): self._extract(value)
                    for key, value in obj.items()
                }
            else:
                return obj

        (
            member_variables,
            discarded_member_variables,
        ) = self._get_member_variables(obj, return_discarded=True)
        self.logger.debug(
            f"Found the following member variables: {member_variables}"
        )
        self.logger.debug(
            f"Discarded the following member variables: {discarded_member_variables}"
        )

        results = None  # Default

        # Has valid member variables -- stick to these, then.
        if len(member_variables) > 0:
            results = {}
            for attr in member_variables:
                value = getattr(obj, attr)
                self.logger.debug(
                    f"Calling `extract` on valid member attribute: {attr}"
                )
                result = self._extract(value)
                results[attr] = result

        else:
            self.logger.debug(
                "Found no member variables! Trying to check for python object-like signatures."
            )

            # Dict-like
            if hasattr(obj, "items"):
                # Call function again
                results = NonFlattenableDict(self._extract(dict(obj)))

            # List-like
            elif hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
                # Call function again
                results = self._extract(list(obj))

            else:
                self.logger.warning(
                    f"Cannot extract any information to pure python from {obj}"
                )
                results = "!!! NOT PARSED !!!"

        return results

    def __call__(self, frame: icetray.I3Frame) -> dict:
        """Extract all possible data from `frame`."""

        results = {}
        self._frame = frame
        for key in ["IceCubePulses"]:  # frame.keys():  # TEMP!!!
            try:
                obj = frame[key]
            except (KeyError, RuntimeError):
                self.logger.debug(f"Frame {key} not supported. Skipping.")
                continue

            self.logger.debug(f"Calling `extract` on key {key} in frame.")
            results[key] = flatten_nested_dictionary(self._extract(obj))

        self._frame = None

        return results

        """
        # Get OM data
        om_keys, data = self._get_om_keys_and_pulseseries(frame)

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(frame, om_key)

            # Loop over pulses for each OM
            pulses = data[om_key]
            for pulse in pulses:
                output["charge"].append(pulse.charge)
                output["dom_time"].append(pulse.time)
                output["width"].append(pulse.width)
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)

        return output"""

    def _get_om_keys_and_pulseseries(self, frame):
        """Gets the indicies for the gcd_dict and the pulse series

        Args:
            frame (i3 physics frame): i3 physics frame

        Returns:
            om_keys (index): the indicies for the gcd_dict
            data    (??)   : the pulse series
        """
        try:
            data = frame[self._pulsemap]
        except KeyError:
            self.logger.error(
                f"Pulsemap {self._pulsemap} does not exist in `frame`"
            )
            raise

        try:
            om_keys = data.keys()
        except:  # noqa: E722
            try:
                if "I3Calibration" in frame.keys():
                    data = frame[self._pulsemap].apply(frame)
                    om_keys = data.keys()
                else:
                    frame["I3Calibration"] = self._calibration
                    data = frame[self._pulsemap].apply(frame)
                    om_keys = data.keys()
                    del frame[
                        "I3Calibration"
                    ]  # Avoid adding unneccesary data to frame
            except:  # noqa: E722
                data = dataclasses.I3RecoPulseSeriesMap.from_frame(
                    frame, self._pulsemap
                )
                om_keys = data.keys()
        return om_keys, data
