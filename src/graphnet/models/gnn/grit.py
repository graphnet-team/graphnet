"""Implementation of GRIT, a graph transformer model.

Original author: Liheng Ma
Original code: https://github.com/LiamMa/GRIT
Paper: "Graph Inductive Biases in Transformers without Message Passing",
        https://arxiv.org/abs/2305.17589

Adapted by: Philip Weigel
"""

import torch.nn as nn

from torch import Tensor
from torch_geometric.data import Data

from graphnet.models.gnn.gnn import GNN

from graphnet.models.components.layers import (
    GritTransformerLayer,
    SANGraphHead,
)
from graphnet.models.components.embedding import (
    RRWPLinearEdgeEncoder,
    RRWPLinearNodeEncoder,
    LinearNodeEncoder,
    LinearEdgeEncoder,
)


class GRIT(GNN):
    """GRIT is a graph transformer model.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/network/grit_model.py
    """

    def __init__(
        self,
        nb_inputs: int,
        hidden_dim: int,
        dim_out: int = 1,
        ksteps: int = 21,
        n_layers: int = 10,
        n_heads: int = 8,
        pad_to_full_graph: bool = True,
        add_node_attr_as_self_loop: bool = False,
        dropout: float = 0.0,
        fill_value: float = 0.0,
        norm: nn.Module = nn.BatchNorm1d,
        attn_dropout: float = 0.2,
        edge_enhance: bool = True,
        update_edges: bool = True,
        attn_clamp: float = 5.0,
        activation: nn.Module = nn.ReLU,
        attn_activation: nn.Module = nn.ReLU,
        norm_edges: bool = True,
        enable_edge_transform: bool = True,
    ):
        """Construct `GRIT` model.

        Args:
            nb_inputs: Number of inputs.
            hidden_dim: Size of hidden dimension.
            dim_out: Size of output dimension.
            ksteps: Number of random walk steps.
            n_layers: Number of GRIT layers.
            n_heads: Number of heads in MHA.
            pad_to_full_graph: Pad to form fully-connected graph.
            add_node_attr_as_self_loop: Adds node attr as an self-edge.
            dropout: Dropout probability.
            fill_value: Padding value.
            norm: Normalization layer.
            attn_dropout: Attention dropout probability.
            edge_enhance: Applies learnable weight matrix with node-pair in
                output node calculation for MHA.
            update_edges: Update edge values after GRIT layer.
            attn_clamp: Clamp absolute value of attention scores to a value.
            activation: Activation function.
            attn_activation: Attention activation function.
            norm_edges: Apply normalization layer to edges.
            enable_edge_transform: Apply transformation to edges.
        """
        super().__init__(nb_inputs, dim_out)

        self.node_encoder = LinearNodeEncoder(nb_inputs, hidden_dim)
        self.edge_encoder = LinearEdgeEncoder(hidden_dim)

        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, hidden_dim)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(
            ksteps,
            hidden_dim,
            pad_to_full_graph=pad_to_full_graph,
            add_node_attr_as_self_loop=add_node_attr_as_self_loop,
            fill_value=fill_value,
        )

        layers = []
        for _ in range(n_layers):
            layers.append(
                GritTransformerLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    activation=activation,
                    attn_dropout=attn_dropout,
                    norm=norm,
                    residual=True,
                    norm_edges=norm_edges,
                    enable_edge_transform=enable_edge_transform,
                    update_edges=update_edges,
                    attn_activation=attn_activation,
                    attn_clamp=attn_clamp,
                    attn_edge_enhance=edge_enhance,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.head = SANGraphHead(dim_in=hidden_dim, dim_out=1)

    def forward(self, x: Data) -> Tensor:
        """Forward pass."""
        # Apply linear layers to node/edge features
        x = self.node_encoder(x)
        x = self.edge_encoder(x)

        # Encode with RRWP
        x = self.rrwp_abs_encoder(x)
        x = self.rrwp_rel_encoder(x)

        # Apply GRIT layers
        for layer in self.layers:
            x = layer(x)

        # Graph head
        x = self.head(x)

        return x
