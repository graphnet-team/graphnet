"""Class(es) implementing layers to be used in `graphnet` models."""

import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.nn.pool import (
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.data import Data

from pytorch_lightning import LightningModule

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


# TODO: This is a prediction head... we probably want only the graph stuff here
# and let the Tasks handle the last layer. -PW
class SANGraphHead(LightningModule):
    """SAN prediction head for graph prediction tasks.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/head/san_graph.py
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int = 1,
        L: int = 2,
        activation: nn.Module = nn.ReLU,
        pooling: str = "mean",
    ):
        """Construct `SANGraphHead`.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            L: Number of hidden layers.
            activation: Uninstantiated activation function.
                E.g. `torch.nn.ReLU`
            pooling: Node-wise pooling operation. Either "mean" or "add".
        """
        super().__init__()
        if pooling == "mean":
            self.pooling_fun = global_mean_pool
        elif pooling == "add":
            self.pooling_fun = global_add_pool
        else:
            raise RuntimeError("Currently supports only 'add' or 'mean'.")

        fc_layers = [
            nn.Linear(dim_in // 2**n, dim_in // 2 ** (n + 1), bias=True)
            for n in range(L)
        ]
        assert dim_in // 2**L >= dim_out, "Too much dim reduction!"
        fc_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.L = L
        self.activation = activation()
        self.dim_out = dim_out

    def forward(self, data: Data) -> Tensor:
        """Forward Pass."""
        graph_emb = self.pooling_fun(data.x, data.batch)
        for i in range(self.L):
            graph_emb = self.fc_layers[i](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.fc_layers[self.L](graph_emb)
        # Original code applied a final linear layer to project to dim_out,
        # but we will let the Task layer do that.
        return graph_emb
