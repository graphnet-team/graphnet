"""SQLite-specific implementation of data classes."""

from graphnet.utilities.imports import has_torch_package

from .sqlite_dataconverter import SQLiteDataConverter
from .sqlite_utilities import create_table_and_save_to_sql

if has_torch_package():
    from .sqlite_dataset import SQLiteDataset
    from .sqlite_dataset_perturbed import SQLiteDatasetPerturbed

del has_torch_package
