"""Module containing experiment-specific dataclasses."""


from dataclasses import dataclass


@dataclass
class I3FileSet:  # noqa: D101
    i3_file: str
    gcd_file: str
