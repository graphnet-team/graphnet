"""Modules for configuring and building models.

`graphnet.models` allows for configuring and building complex GNN models using
simple, physics-oriented components. This module provides modular components
subclassing `torch.nn.Module`, meaning that users only need to import a few,
existing, purpose-built components and chain them together to form a complete
GNN
"""
from graphnet.utilities.imports import has_jammy_flows_package
from .model import Model
from .standard_model import StandardModel
from .standard_averaged_model import StandardAveragedModel

if has_jammy_flows_package():
    from .normalizing_flow import NormalizingFlow
