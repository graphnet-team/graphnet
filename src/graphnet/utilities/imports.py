"""Common functionns for icetray/data-based unit tests."""

from functools import wraps

from graphnet.utilities.logging import get_logger


logger = get_logger()


def has_icecube_package() -> bool:
    """Check whether the `icecube` package is available."""
    try:
        import icecube  # pyright: reportMissingImports=false

        return True
    except ImportError:
        return False


def requires_icecube(test_function):
    """Decorator for only exposing function if `icecube` module is present."""

    @wraps(test_function)
    def wrapper(*args, **kwargs):
        if has_icecube_package():
            return test_function(*args, **kwargs)
        else:
            logger.info(
                f"Function `{test_function.__name__}` not used since `icecube` isn't available."
            )
            return

    return wrapper
