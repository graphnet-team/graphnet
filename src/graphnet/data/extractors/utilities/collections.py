"""Utility methods for manipulating python-type collections of data."""

# Utility function(s)
import json
from typing import Any, Dict, Iterable, List, Tuple, Union

import pandas as pd


def flatten_nested_dictionary(
    obj: Union[Dict, Any],
    parent_key: str = "",
    separator: str = "__",
) -> Union[Dict, Any]:
    """Flatten a nested dictionary to a dictionary with non-dict values.

    Example:
        d = {"a": {"b": 1}, "c": 2}
        flatten_nested_dictionary(d)
        >>> {"a__b": 1, "c": 2}

    Args:
        obj: The object that should be flattened, if applicable.
        parent_key: The combined name of the parent key(s) containing `obj`.
        separator: The string used to concatenate nester parent keys.
    """
    # Dict-like
    if isinstance(obj, dict):
        items: List[Tuple[str, Any]] = []
        for key, value in obj.items():
            new_key = parent_key + separator + key if parent_key else key
            result = flatten_nested_dictionary(value, new_key, separator)
            items.extend(result.items())

        return dict(items)

    # Other non-dict like (list, str, float, int, etc.)
    else:
        return {parent_key: obj}


def serialise(obj: Union[Dict, Any]) -> Union[Dict, Any]:
    """Serialise the necessary keys in `obj` to JSON for saving to file.

    It is typically not possible to save nested collections (lists, dicts) to
    file. Therefore, if `obj` is, e.g., a list of lists, we need to serialise
    each element in the outer list to JSON (i.e., convert it to a JSON-
    formatted string) in order to be able to save `obj` to file. It will then
    be possible to de-serialise corresponding elements when reading them from
    file.
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            value = obj[key]
            if (
                isinstance(value, (list, tuple))
                and len(value)
                and isinstance(value[0], Iterable)
            ):
                obj[key] = [json.dumps(element) for element in value]

    elif isinstance(obj, (list, tuple)):
        obj = [json.dumps(element) for element in obj]

    return obj


def transpose_list_of_dicts(
    array: List[Dict[str, Any]]
) -> Dict[str, List[Any]]:
    """Transpose a list of dicts to a dict of lists."""
    if len(array) == 0:
        return {}

    all_keys = [key for d in array for key in d.keys()]
    keys = pd.unique(all_keys)  # Returns keys in order of appearance

    transposed_dict = {
        key: [element.get(key, None) for element in array] for key in keys
    }
    return transposed_dict
