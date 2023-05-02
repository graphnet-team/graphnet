"""Common decorators."""

try:
    from typing import final
except ImportError:  # Python version < 3.8

    # Identity decorator
    def final(f):  # type: ignore  # noqa: D103
        return f
