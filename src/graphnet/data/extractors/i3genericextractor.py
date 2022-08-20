from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional

from graphnet.data.extractors.i3extractor import I3Extractor
from graphnet.data.extractors.utilities.types import (
    is_boost_class,
    is_boost_enum,
    is_icecube_class,
    is_method,
    is_type,
)
from graphnet.data.extractors.utilities.frames import (
    get_om_keys_and_pulseseries,
)
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

    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract all possible data from `frame`."""

        results = {}
        self._frame = frame
        for key in [
            # "I3TriggerHierarchy"
            "I3MCTree",
        ]:  # frame.keys():
            try:
                obj = frame[key]
            except (KeyError, RuntimeError):
                self.logger.debug(f"Frame {key} not supported. Skipping.")
                continue

            # Special case(s)
            if isinstance(
                obj,
                (
                    dataclasses.I3DOMLaunchSeriesMap,
                    dataclasses.I3RecoPulseSeriesMap,
                    dataclasses.I3RecoPulseSeriesMapMask,
                    dataclasses.I3RecoPulseSeriesMapUnion,
                ),
            ):
                self.logger.info(f"Got pulse series - {key}!")
                result = self._cast_pulse_series_to_pure_python(frame, key)

            elif isinstance(
                obj,
                (
                    dataclasses.I3MapKeyUInt,
                    dataclasses.I3MapKeyDouble,
                    dataclasses.I3MapKeyVectorInt,
                    dataclasses.I3MapKeyVectorDouble,
                ),
            ):
                self.logger.info(f"Got per-pulse map - {key}!")
                result = self._cast_pulse_series_to_pure_python(frame, key)

                if result is None:
                    self.logger.debug(
                        f"Per-pulse map {key} didn't return anything."
                    )
                    # result = {}

                else:
                    # If we get a per-pulse map, which isn't a
                    # "I3RecoPulseSeriesMap*", we don't care about area,
                    # direction, orientation, and position -- we only care
                    # about the OM index for future reference. We therefore
                    # only keep these indices and the associated mapping value.
                    keep_keys = ["value"] + [
                        key_ for key_ in result if key_.startswith("index.")
                    ]
                    result = {key_: result[key_] for key_ in keep_keys}

            elif isinstance(obj, dataclasses.I3MCTree):
                self.logger.info(f"Got MC tree {key} in frame.")
                result = self._cast_object_to_pure_python(obj)

                # Assing parent and children links to all particles in tree
                result["particles"] = result.pop("_list")
                for ix, particle in enumerate(obj):
                    try:
                        parent = obj.parent(particle).minor_id
                    except IndexError:
                        parent = None

                    children = [p.minor_id for p in obj.children(particle)]

                    result["particles"][ix]["parent"] = parent
                    result["particles"][ix]["children"] = children

            # Generic case
            else:

                self.logger.info(f"Got generic object {key} in frame.")
                result = self._cast_object_to_pure_python(obj)

            results[key] = flatten_nested_dictionary(result)

        self._frame = None

        return results

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
            "is_boost_class": [],
        }
        for attr in dir(obj):
            if attr.startswith("__"):
                discarded_member_variables["mangled"].append(attr)
                continue

            value = getattr(obj, attr)

            if is_type(value):
                discarded_member_variables["is_type"].append(attr)
            elif is_method(value):
                discarded_member_variables["is_method"].append(attr)
            elif is_boost_enum(value):
                discarded_member_variables["is_boost_enum"].append(attr)
            elif is_boost_class(value):
                discarded_member_variables["is_boost_class"].append(attr)
            else:
                valid_member_variables.append(attr)

        if return_discarded:
            return valid_member_variables, discarded_member_variables

        return valid_member_variables

    def _cast_object_to_pure_python(self, obj: Any) -> Any:
        """Extracts all possible pure-python data from `obj`."""

        self.logger.debug(f"Value: {obj}")
        self.logger.debug(f"Type: {str(type(obj))}")

        if not is_icecube_class(obj):
            self.logger.debug("Found non-I3 class. Exiting.")
            if isinstance(obj, (list, tuple, set)):
                return [
                    self._cast_object_to_pure_python(element)
                    for element in obj
                ]
            elif isinstance(obj, dict):
                return {
                    str(key): self._cast_object_to_pure_python(value)
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
            "Discarded the following member variables: "
            f"{discarded_member_variables}"
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
                result = self._cast_object_to_pure_python(value)
                results[attr] = result

        # Dict-like
        if hasattr(obj, "items"):
            self.logger.critical("Is dict-like")
            # Call function again
            results_dict = NonFlattenableDict(
                self._cast_object_to_pure_python(dict(obj))
            )
            if results is None:
                results = results_dict
            else:
                assert "_dict" not in results
                results["_dict"] = results_dict

        # List-like
        if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
            self.logger.critical("Is list-like")
            # Call function again
            results_list = self._cast_object_to_pure_python(list(obj))
            if results is None:
                results = results_list
            else:
                assert "_list" not in results
                results["_list"] = results_list

        if results is None:
            self.logger.warning(
                f"Cannot extract any information to pure python from {obj}"
            )
            results = "!!! NOT PARSED !!!"

        return results

    def _cast_pulse_series_to_pure_python(
        self,
        frame: "icetray.I3Frame",
        key: str,
    ) -> Optional[Dict[str, List[Any]]]:
        """Cast pulse series `key` to a pure-python data representation.

        Args:
            frame (icetray.I3Frame): I3 physics frame.
            key (str): Name of the pulse series to be cast.

        Returns:
            Dict[str, List[Any]]: Dictionary of lists of properties for each
                pulse across optical modules (OMs), if any pulses are found.
            None, otherwise
        """
        om_keys, data = get_om_keys_and_pulseseries(
            frame,
            key,
            self._calibration,
        )

        result = []
        for om_key in om_keys:
            om_data = self._cast_object_to_pure_python(self._gcd_dict[om_key])

            # Remove all "orientation.*"-type keys. They provide no
            # information apart from the (hopefully!) standard
            # coordinate system and the OM direction, which is covered
            # by the "direction.*" keys anyway.
            om_data.pop("orientation", None)

            om_indices = self._cast_object_to_pure_python(om_key)
            om_data["index"] = om_indices

            om_data = flatten_nested_dictionary(om_data)

            pulses = data[om_key]

            if len(pulses) == 0:
                continue

            pulse_data = self._cast_object_to_pure_python(pulses)

            # Ensure that `pulse_data` has the form of a list of dictionary of
            # per-pulse properties
            if isinstance(pulse_data, (list, tuple)):
                if not isinstance(pulse_data[0], dict):
                    pulse_data = [{"value": value} for value in pulse_data]
            else:
                pulse_data = [{"value": pulse_data}]

            for ix in range(len(pulse_data)):
                pulse_data[ix].update(om_data)

            # "Transpose" list of dicts to dict of lists
            pulse_data = {
                key: [pulse[key] for pulse in pulse_data]
                for key in pulse_data[0]
            }
            result.append(pulse_data)

        # Concatenate list of pulses from different OMs
        if len(result):
            result = {
                key: [pulse for pulses in result for pulse in pulses[key]]
                for key in result[0]
            }
            return result
        else:
            return None
