"""Datasets using parquet backend."""

# Configuration
from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    import torch.multiprocessing
    from .parquet_dataset import ParquetDataset

    torch.multiprocessing.set_sharing_strategy("file_system")

del has_torch_package
