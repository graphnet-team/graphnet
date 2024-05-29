"""Common functions for icetray/data-based unit tests."""

from functools import wraps
from typing import Any, Callable

from graphnet.utilities.logging import Logger


def has_icecube_package() -> bool:
    """Check whether the `icecube` package is available."""
    try:
        import icecube  # pyright: reportMissingImports=false
        from icecube import icetray, dataio

        return True
    except ImportError:
        Logger(log_folder=None).warning_once(
            "`icecube` not available. Some functionality may be missing.",
        )
        return False


def has_torch_package() -> bool:
    """Check whether the `torch` package is available."""
    try:
        import torch  # pyright: reportMissingImports=false

        return True
    except ImportError:
        Logger(log_folder=None).warning_once(
            "`torch` not available. Some functionality may be missing."
        )
        return False


def has_jammy_flows_package() -> bool:
    """Check if the `jammy_flows` package is available."""
    try:
        import jammy_flows  # pyright: reportMissingImports=false

        return True
    except ImportError:
        Logger(log_folder=None).warning_once(
            "`jammy_flows` not available. Normalizing Flow functionality is "
            "missing."
        )
        return False


def requires_icecube(test_function: Callable) -> Callable:
    """Decorate `test_function` for use only if `icecube` module is present."""

    @wraps(test_function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if has_icecube_package():
            return test_function(*args, **kwargs)
        else:
            Logger(log_folder=None).info(
                f"Function `{test_function.__name__}` not used since `icecube` isn't available."
            )
            return

    return wrapper
