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
    RWSELinearNodeEncoder,
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
        nb_outputs: int = 1,
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
        pred_head_layers: int = 2,
        pred_head_activation: nn.Module = nn.ReLU,
        pred_head_pooling: str = "mean",
        position_encoding: str = "NoPE",
    ):
        """Construct `GRIT` model.

        Args:
            nb_inputs: Number of inputs.
            hidden_dim: Size of hidden dimension.
            nb_outputs: Size of output dimension.
            ksteps: Number of random walk steps.
            n_layers: Number of GRIT layers.
            n_heads: Number of heads in MHA.
            pad_to_full_graph: Pad to form fully-connected graph.
            add_node_attr_as_self_loop: Adds node attr as an self-edge.
            dropout: Dropout probability.
            fill_value: Padding value.
            norm: Uninstantiated normalization layer.
                Either `torch.nn.BatchNorm1d` or `torch.nn.LayerNorm`.
            attn_dropout: Attention dropout probability.
            edge_enhance: Applies learnable weight matrix with node-pair in
                output node calculation for MHA.
            update_edges: Update edge values after GRIT layer.
            attn_clamp: Clamp absolute value of attention scores to a value.
            activation: Uninstantiated activation function.
                E.g. `torch.nn.ReLU`
            attn_activation: Uninstantiated attention activation function.
                E.g. `torch.nn.ReLU`
            norm_edges: Apply normalization layer to edges.
            enable_edge_transform: Apply transformation to edges.
            pred_head_layers: Number of layers in the prediction head.
            pred_head_activation: Uninstantiated prediction head activation
                    function. E.g. `torch.nn.ReLU`
            pred_head_pooling: Pooling function to use for the prediction head,
                either "mean" (default) or "add".
            position_encoding: Method of position encoding.
        """
        super().__init__(nb_inputs, nb_outputs)
        self.position_encoding = position_encoding.lower()
        if self.position_encoding == "nope":
            encoders = [
                LinearNodeEncoder(nb_inputs, hidden_dim),
                LinearEdgeEncoder(hidden_dim),
            ]
        elif self.position_encoding == "rrwp":
            encoders = [
                LinearNodeEncoder(nb_inputs, hidden_dim),
                LinearEdgeEncoder(hidden_dim),
                RRWPLinearNodeEncoder(ksteps, hidden_dim),
                RRWPLinearEdgeEncoder(
                    ksteps,
                    hidden_dim,
                    pad_to_full_graph=pad_to_full_graph,
                    add_node_attr_as_self_loop=add_node_attr_as_self_loop,
                    fill_value=fill_value,
                ),
            ]
        elif self.position_encoding == "rwse":
            encoders = [
                LinearNodeEncoder(nb_inputs, hidden_dim - (ksteps - 1)),
                RWSELinearNodeEncoder(ksteps - 1, hidden_dim),
            ]
        self.encoders = nn.ModuleList(encoders)

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
        self.head = SANGraphHead(
            dim_in=hidden_dim,
            dim_out=nb_outputs,
            L=pred_head_layers,
            activation=pred_head_activation,
            pooling=pred_head_pooling,
        )

    def forward(self, x: Data) -> Tensor:
        """Forward pass."""
        for encoder in self.encoders:
            x = encoder(x)

        # Apply GRIT layers
        for layer in self.layers:
            x = layer(x)

        # Graph head
        x = self.head(x)

        return x
