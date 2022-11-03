"""Utility methods for checking the types of objects."""

from functools import wraps
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from graphnet.data.extractors.utilities.collections import (
    transpose_list_of_dicts,
    flatten_nested_dictionary,
)
from graphnet.data.extractors.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import get_logger, warn_once

logger = get_logger()

if has_icecube_package():
    from icecube import (
        icetray,
    )  # pyright: reportMissingImports=false


def is_boost_enum(obj: Any) -> bool:
    """Check whether `obj` inherits from Boost.Python.enum."""
    for cls in type(obj).__bases__:
        if "Boost.Python.enum" in str(cls):
            return True
    return False


def is_boost_class(obj: Any) -> bool:
    """Check whether `obj` is instance of Boost.Python.enum."""
    return "Boost.Python.class" in str(type(obj))


def is_icecube_class(obj: Any) -> bool:
    """Check whether `obj` is an IceCube-specific class."""
    classname = str(type(obj))
    return "icecube." in classname


def is_type(obj: Any) -> bool:
    """Check whether `obj` is a type, and not an instance."""
    return type(obj).__name__ == "type"


def is_method(obj: Any) -> bool:
    """Check whether `obj` is a method."""
    return inspect.ismethod(obj) or "Boost.Python.function" in str(type(obj))


BEING_EVALUATED = set()


def break_cyclic_recursion(fn: Callable) -> Callable:
    """Ensure that method isn't called recursively on the same object."""

    @wraps(fn)
    def wrapper(obj: Any) -> Any:
        global BEING_EVALUATED
        try:
            hash_ = (hash(fn), hash(obj))
            if hash_ in BEING_EVALUATED:
                warn_once(
                    logger,
                    "break_cyclic_recursion - Already evaluating object. Skipping recusion.",
                )
                return
            BEING_EVALUATED.add(hash_)
            ret = fn(obj)
            BEING_EVALUATED.remove(hash_)
            return ret
        except TypeError:
            return fn(obj)

    return wrapper


def get_member_variables(
    obj: Any, return_discarded: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, List[str]]]]:
    """Return list of valid member variables.

    Ignoring mangled (__*) variables, types, methods, and Boost enums.
    """
    valid_member_variables = []
    discarded_member_variables: Dict[str, List[str]] = {
        "mangled": [],
        "is_type": [],
        "invalid_attr": [],
        "is_method": [],
        "is_boost_enum": [],
        "is_boost_class": [],
    }
    for attr in dir(obj):
        if attr.startswith("__"):
            discarded_member_variables["mangled"].append(attr)
            continue

        try:
            value = getattr(obj, attr)
        except RuntimeError:
            discarded_member_variables["invalid_attr"].append(attr)
            continue

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


@break_cyclic_recursion
def cast_object_to_pure_python(obj: Any) -> Any:
    """Cast `obj`, and any members/elements, to pure-python classes.

    The function takes any object `obj` and tries to cast it to a pure python
    class. This is mainly relevant for IceCube-specific classes (I3*) that
    cannot be cast trivially.

    For IceCube-specific classes, we check whether the object has any member,
    variables and if does, we recursively try to cast these to pure python.
    Similarly, if an IceCube-specific class has a signature similar to a python
    list or dict (e.g, it has a length and supports indexation), we cast it to
    the corresponding pure python equivalent, and recursively try to cast its
    elements.

    For regular-python, non-Icecube-specific, classes, we cast to list-like
    objects to list and dict-like objects to list, and otherwise return the
    object itself if it deemed "pythonic" in this way.
    """
    logger.debug(f"Value: {obj}")
    logger.debug(f"Type: {str(type(obj))}")

    if not is_icecube_class(obj):
        logger.debug("Found non-I3 class. Exiting.")
        if isinstance(obj, (list, tuple, set)):
            return [cast_object_to_pure_python(element) for element in obj]
        elif isinstance(obj, dict):
            return {
                str(key): cast_object_to_pure_python(value)
                for key, value in obj.items()
            }
        else:
            return obj

    (
        member_variables,
        discarded_member_variables,
    ) = get_member_variables(obj, return_discarded=True)

    logger.debug(f"Found the following member variables: {member_variables}")
    logger.debug(
        "Discarded the following member variables: "
        f"{discarded_member_variables}"
    )

    # Has valid member variables -- stick to these, then.
    results = {}
    if len(member_variables) > 0:
        for attr in member_variables:
            value = getattr(obj, attr)
            logger.debug(
                f"Calling `extract` on valid member attribute: {attr}"
            )
            result = cast_object_to_pure_python(value)
            results[attr] = result

    # Dict-like
    if hasattr(obj, "items"):
        # Call function again
        results_dict = cast_object_to_pure_python(dict(obj))
        assert "_dict" not in results
        results["_dict"] = results_dict

    # List-like
    elif hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        # Call function again
        results_list = cast_object_to_pure_python(list(obj))
        assert "_list" not in results
        results["_list"] = results_list

    # If `obj` has no actual member variables, but is otherwise python
    # dict- or list-like, there is no need to wrap the data in a single-
    # key dict.
    if list(results.keys()) == ["_dict"]:
        results = results.pop("_dict")
    elif list(results.keys()) == ["_list"]:
        results = results.pop("_list")

    if len(results) == 0:
        logger.warning(
            f"Cannot extract any information to pure python from {obj}"
        )

    return results


def cast_pulse_series_to_pure_python(
    frame: "icetray.I3Frame",
    key: str,
    calibration: Any,
    gcd_dict: Dict,
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
        calibration,
    )

    result = []
    for om_key in om_keys:
        om_data = cast_object_to_pure_python(gcd_dict[om_key])

        # Add calibration information
        om_data.update(cast_object_to_pure_python(calibration.dom_cal[om_key]))

        # Remove all "orientation.*"-type keys. They provide no
        # information apart from the (hopefully!) standard
        # coordinate system and the OM direction, which is covered
        # by the "direction.*" keys anyway.
        om_data.pop("orientation", None)

        om_indices = cast_object_to_pure_python(om_key)
        om_data["index"] = om_indices

        try:
            om_data = flatten_nested_dictionary(om_data)
        except TypeError:
            logger.warning("Couldn't call `flatten_nested_dictionary` on:")
            print(om_data)
            raise

        pulses = data[om_key]

        if len(pulses) == 0:
            continue

        pulse_data: List[Dict[str, Any]] = cast_object_to_pure_python(pulses)

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
        pulse_data_dict = transpose_list_of_dicts(pulse_data)
        result.append(pulse_data_dict)

    # Concatenate list of pulses from different OMs
    if len(result):
        result_combined = {
            key: [pulse for pulses in result for pulse in pulses[key]]
            for key in result[0]
        }
        return result_combined
    else:
        return None
