"""Utility functions for parsing for using with Config-classes."""

import itertools
import pkgutil
import types
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from graphnet.utilities.logging import get_logger

logger = get_logger()


def traverse_and_apply(
    obj: Any, fn: Callable, fn_kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """Apply `fn` to all elements in `obj`, resulting in same structure."""
    if isinstance(obj, (list, tuple)):
        return [traverse_and_apply(elem, fn, fn_kwargs) for elem in obj]
    elif isinstance(obj, dict):
        return {
            key: traverse_and_apply(val, fn, fn_kwargs)
            for key, val in obj.items()
        }
    else:
        if fn_kwargs is None:
            fn_kwargs = {}
        return fn(obj, **fn_kwargs)


def list_all_submodules(*packages: types.ModuleType) -> List[types.ModuleType]:
    """List all submodules in `packages` recursively."""
    # Resolve one or more packages
    if len(packages) > 1:
        return list(
            itertools.chain.from_iterable(map(list_all_submodules, packages))
        )
    else:
        assert len(packages) == 1, "No packages specified"
        package = packages[0]

    submodules: List[types.ModuleType] = []
    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = __import__(module_name, fromlist="dummylist")
        submodules.append(module)
        if is_pkg:
            submodules.extend(list_all_submodules(module))

    return submodules


def get_all_grapnet_classes(*packages: types.ModuleType) -> Dict[str, type]:
    """List all grapnet classes in `packages`."""
    submodules = list_all_submodules(*packages)
    classes: Dict[str, type] = {}
    for submodule in submodules:
        new_classes = get_graphnet_classes(submodule)
        for key in new_classes:
            if key in classes and classes[key] != new_classes[key]:
                logger.warning(
                    f"Class {key} found in both {classes[key]} and "
                    f"{new_classes[key]}. Keeping first instance. "
                    "Consider renaming."
                )
        classes.update(new_classes)

    return classes


def is_graphnet_module(obj: types.ModuleType) -> bool:
    """Return whether `obj` is a module in graphnet."""
    return isinstance(obj, types.ModuleType) and obj.__name__.startswith(
        "graphnet."
    )


def is_graphnet_class(obj: type) -> bool:
    """Return whether `obj` is a class in graphnet."""
    return isinstance(obj, type) and obj.__module__.startswith("graphnet.")


def get_graphnet_classes(module: types.ModuleType) -> Dict[str, type]:
    """Return a lookup of all graphnet class names in `module`."""
    if not is_graphnet_module(module):
        logger.info(f"{module} is not a graphnet module")
        return {}
    classes = {
        key: val
        for key, val in module.__dict__.items()
        if is_graphnet_class(val)
    }
    return classes
