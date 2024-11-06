"""Datasets using SQLite backend."""

from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    from .sqlite_dataset import SQLiteDataset

del has_torch_package
