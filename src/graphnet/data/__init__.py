"""Modules for converting and ingesting data.

`graphnet.data` enables converting domain-specific data to industry-standard,
intermediate file formats  and reading this data.
"""

# Configuration
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
