"""Module containing experiment-specific dataclasses."""

from typing import List, Any
from dataclasses import dataclass


@dataclass
class I3FileSet:  # noqa: D101
    i3_file: str
    gcd_file: str


@dataclass
class SQLiteFileSet:  # noqa: D101
    db_path: str
    event_nos: List[int]


@dataclass
class Settings:
    """Dataclass for workers in I3Deployer."""

    i3_files: List[str]
    gcd_file: str
    output_folder: str
    modules: List[Any]
