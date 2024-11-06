"""Dataset classes for training in GraphNeT."""

# Configuration
from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    import torch.multiprocessing
    from .dataset import EnsembleDataset, Dataset, ColumnMissingException
    from .samplers import (
        RandomChunkSampler,
        LenMatchBatchSampler,
    )
    from .parquet.parquet_dataset import ParquetDataset
    from .sqlite.sqlite_dataset import SQLiteDataset

    torch.multiprocessing.set_sharing_strategy("file_system")

del has_torch_package
