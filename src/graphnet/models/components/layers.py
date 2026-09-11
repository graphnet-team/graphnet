"""Removed module: layer classes live in dedicated modules.

Importing this module succeeds so that package-wide module scans (e.g.
config-class namespace building) keep working, but accessing any name
on it raises an instructive error pointing at the new location.
"""

from typing import Any

_NEW_HOMES = {
    "DynEdgeConv": "edge_convolutions",
    "EdgeConvTito": "edge_convolutions",
    "DynTrans": "edge_convolutions",
    "DropPath": "attention_blocks",
    "Mlp": "attention_blocks",
    "Attention_rel": "attention_blocks",
    "Block_rel": "attention_blocks",
    "Block": "attention_blocks",
    "GritSparseMHA": "grit_layers",
    "GritTransformerLayer": "grit_layers",
    "SANGraphHead": "grit_layers",
}


def __getattr__(name: str) -> Any:
    """Raise an instructive error for any name looked up on this module."""
    if name in _NEW_HOMES:
        raise ImportError(
            f"graphnet.models.components.layers has been removed; import "
            f"{name} from graphnet.models.components.{_NEW_HOMES[name]} "
            "instead."
        )
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r} "
        "(graphnet.models.components.layers has been removed)"
    )
