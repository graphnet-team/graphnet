"""Modules for converting and ingesting data.

`graphnet.data` enables converting domain-specific data to industry-standard,
intermediate file formats  and reading this data.
"""

# Configuration
from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

del has_torch_package
