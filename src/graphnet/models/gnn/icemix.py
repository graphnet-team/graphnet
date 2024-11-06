"""Implementation of IceMix.

This method was a solution submitted to the IceCube - Neutrinos in Deep Ice
Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

import torch
import torch.nn as nn
from typing import Set, Dict, Any, Optional

from graphnet.models.components.layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoder,
    SpacetimeEncoder,
)
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class DeepIce(GNN):
    """DeepIce model."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Optional[Dict[str, Any]] = None,
        n_features: int = 6,
    ):
        """Construct `DeepIce`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with the original Kaggle
                Competition settings. If `include_dynedge` is False, this
                argument have no impact.
            n_features: The number of features in the input data.
        """
        super().__init__(seq_length, hidden_dim)
        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        self.fourier_ext = FourierEncoder(
            seq_length=seq_length,
            mlp_dim=None,
            output_dim=fourier_out_dim,
            scaled=scaled_emb,
            n_features=n_features,
        )
        self.rel_pos = SpacetimeEncoder(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(
                    input_dim=hidden_dim, num_heads=hidden_dim // head_size
                )
                for _ in range(depth_rel)
            ]
        )
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.n_rel = n_rel

        if include_dynedge and dynedge_args is None:
            self.warning_once("Running with default DynEdge settings")
            self.dyn_edge = DynEdge(
                nb_inputs=9,
                nb_neighbours=9,
                post_processing_layer_sizes=[336, hidden_dim // 2],
                dynedge_layer_sizes=[
                    (128, 256),
                    (336, 256),
                    (336, 256),
                    (336, 256),
                ],
                global_pooling_schemes=None,
                activation_layer="gelu",
                add_norm_layer=True,
                skip_readout=True,
            )
        elif include_dynedge and not (dynedge_args is None):
            self.dyn_edge = DynEdge(**dynedge_args)

        self.include_dynedge = include_dynedge

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x0, mask, seq_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        x = self.fourier_ext(x0, seq_length)
        rel_pos_bias = self.rel_pos(x0)
        batch_size = mask.shape[0]
        if self.include_dynedge:
            graph = self.dyn_edge(data)
            graph, _ = to_dense_batch(graph, data.batch)
            x = torch.cat([x, graph], 2)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [
                torch.ones(
                    batch_size, 1, dtype=mask.dtype, device=mask.device
                ),
                mask,
            ],
            1,
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            x = blk(x, None, attn_mask)

        return x[:, 0]
