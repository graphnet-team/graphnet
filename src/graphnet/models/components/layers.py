"""Backward-compatibility re-exports of model layer classes.

The implementations live in the dedicated modules `edge_convolutions`,
`attention_blocks`, and `grit_layers`. This module re-exports them so
that existing imports and saved model configs referencing
`graphnet.models.components.layers` keep working.
"""

from graphnet.models.components.edge_convolutions import (
    DynEdgeConv,
    EdgeConvTito,
    DynTrans,
)

from graphnet.models.components.attention_blocks import (
    DropPath,
    Mlp,
    Attention_rel,
    Block_rel,
    Block,
)

from graphnet.models.components.grit_layers import (
    GritSparseMHA,
    GritTransformerLayer,
    SANGraphHead,
)

__all__ = [
    "DynEdgeConv",
    "EdgeConvTito",
    "DynTrans",
    "DropPath",
    "Mlp",
    "Block_rel",
    "Attention_rel",
    "Block",
    "GritSparseMHA",
    "GritTransformerLayer",
    "SANGraphHead",
]
