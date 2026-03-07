"""Datasets using LMDB backend."""

from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    from .lmdb_dataset import LMDBDataset

del has_torch_package
