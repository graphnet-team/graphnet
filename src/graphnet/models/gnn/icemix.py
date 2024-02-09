"""Implementation of IceMix architecture used in.

                    IceCube - Neutrinos in Deep Ice
Reconstruct the direction of neutrinos from the Universe to the South Pole

Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""
import torch
import torch.nn as nn
from typing import List

from graphnet.models.components.layers import FourierEncoder, SpacetimeEncoder, Block_rel, Block
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN

from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor

def convert_data(data: Data):
    """Convert the input data to a tensor of shape (B, L, D)"""
    _, seq_length = torch.unique(data.batch, return_counts=True)
    x_list = torch.split(data.x, seq_length.tolist())
    x = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True, padding_value=torch.inf)
    mask = torch.ne(x[:,:,1], torch.inf)
    x[~mask] = 0
    return x, mask, seq_length

class DeepIce(GNN):
    """DeepIce model."""
    def __init__(
        self,
        dim: int = 384,
        dim_base: int = 128,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
    ):
        """Construct `DeepIce`.

        Args:
            dim: The latent feature dimension.
            dim_base: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
        """
        super().__init__(dim_base, dim)
        self.fourier_ext = FourierEncoder(dim_base, dim)
        self.rel_pos = SpacetimeEncoder(head_size)
        self.sandwich = nn.ModuleList(
            [Block_rel(dim=dim, num_heads=dim // head_size) for i in range(depth_rel)]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.n_rel = n_rel
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x0, mask, seq_length = convert_data(data)
        x = self.fourier_ext(x0, seq_length)
        rel_pos_bias = self.rel_pos(x0)
        batch_size = mask.shape[0]
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            x = blk(x, None, attn_mask)

        return x[:, 0]
    
    
class DeepIceWithDynEdge(GNN):
    """DeepIce model with DynEdge."""	
    def __init__(
        self,
        dim: int = 384,
        dim_base: int = 128,
        depth: int = 8,
        head_size: int = 64,
        features_subset: List[int] = [0, 1, 2],
    ):
        """Construct `DeepIceWithDynEdge`.
        
        Args:
            dim: The latent feature dimension.
            dim_base: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            features_subset: The subset of features to 
                use for the edge construction.
        """
        super().__init__(dim_base, dim)
        self.features_subset = features_subset
        self.fourier_ext = FourierEncoder(dim_base, dim // 2, scaled=True)
        self.rel_pos = SpacetimeEncoder(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
            ]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.dyn_edge = DynEdge(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
            global_pooling_schemes=None,
            icemix_encoder=True,
        )
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x0, mask, seq_length = convert_data(data)
        #for i in range(3, 7):
        #    data.x[:, i] = torch.squeeze(data.x[:, i].view(-1, 1))
            
        x = self.fourier_ext(x0, seq_length)
        rel_pos_bias = self.rel_pos(x0)
        graph = self.dyn_edge(data)
        graph, _ = to_dense_batch(graph, data.batch)

        batch_size = mask.shape[0]
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        x = torch.cat([x, graph], 2)

        for blk in self.sandwich:
            x = blk(x, attn_mask, rel_pos_bias)
            if len(self.features_subset) == 3:
                rel_pos_bias = None
        mask = torch.cat(
            [torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            x = blk(x, None, attn_mask)

        return x[:, 0]