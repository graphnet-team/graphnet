"""Common functions for icetray/data-based unit tests."""

from functools import wraps
from typing import Any, Callable

from graphnet.utilities.logging import get_logger, warn_once


logger = get_logger()


def has_icecube_package() -> bool:
    """Check whether the `icecube` package is available."""
    try:
        import icecube  # pyright: reportMissingImports=false

        return True
    except ImportError:
        warn_once(
            logger,
            "`icecube` not available. Some functionality may be missing.",
        )
        return False


def has_torch_package() -> bool:
    """Check whether the `torch` package is available."""
    try:
        import torch  # pyright: reportMissingImports=false

        return True
    except ImportError:
        warn_once(
            logger, "`torch` not available. Some functionality may be missing."
        )
        return False


def has_pisa_package() -> bool:
    """Check whether the `pisa` package is available."""
    try:
        import pisa  # pyright: reportMissingImports=false

        return True
    except ImportError:
        warn_once(
            logger,
            "`pisa` not available. Some functionality may be missing.",
        )
        return False


def requires_icecube(test_function: Callable) -> Callable:
    """Decorate `test_function` for use only if `icecube` module is present."""

    @wraps(test_function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if has_icecube_package():
            return test_function(*args, **kwargs)
        else:
            logger.info(
                f"Function `{test_function.__name__}` not used since `icecube` isn't available."
            )
            return

    return wrapper
