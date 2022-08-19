"""Utility methods for checking the types of objects."""

import inspect
from typing import Any


def is_boost_enum(obj: Any) -> bool:
    """Check whether `obj` inherits from Boost.Python.enum."""
    for cls in type(obj).__bases__:
        if "Boost.Python.enum" in str(cls):
            return True
    return False


def is_icecube_class(obj: Any) -> bool:
    """Check whether `obj` is an IceCube-specific class."""
    classname = str(type(obj))
    return "icecube." in classname


def is_type(obj: Any) -> bool:
    """Checks whether `obj` is a type, and not an instance."""
    return type(obj).__name__ == "type"


def is_method(obj: Any) -> bool:
    """Check whether `obj` is a method."""
    return inspect.ismethod(obj) or "Boost.Python.function" in str(type(obj))
