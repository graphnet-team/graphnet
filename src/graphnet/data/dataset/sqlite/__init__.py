"""Datasets using SQLite backend."""
from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    from .sqlite_dataset import SQLiteDataset
    from .sqlite_dataset_perturbed import SQLiteDatasetPerturbed

del has_torch_package
