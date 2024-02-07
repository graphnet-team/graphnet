import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math

from graphnet.models.components.layers import FourierEncoder, SpacetimeEncoder, Block_rel, Block
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN

from timm.models.layers import trunc_normal_

from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor



class DeepIce(GNN):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=12,
        use_checkpoint=False,
        head_size=32,
        depth_rel=4,
        n_rel=1,
        **kwargs,
    ):
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
        #self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.n_rel = n_rel
        
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        mask = data.mask
        Lmax = data.n_pulses.sum(-1)
        x = self.fourier_ext(data, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(data, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        #x = self.proj_out(x[:, 0])  # cls token
        return x[:, 0]
    
    
class DeepIceWithDynEdge(GNN):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=8,
        use_checkpoint=False,
        head_size=64,
        knn_features=3,
        **kwargs,
    ):
        super().__init__(dim_base, dim)
        self.knn_features = knn_features
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
        #self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.dyn_edge = DynEdge(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
            global_pooling_schemes=None
        )
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        mask = data.mask
        graph_feature = torch.concat(
            [
                data.pos[mask],
                data.time[mask].view(-1, 1),
                data.auxiliary[mask].view(-1, 1),
                data.qe[mask].view(-1, 1),
                data.charge[mask].view(-1, 1),
                data.ice_properties[mask],
            ],
            dim=1,
        )
        Lmax = mask.sum(-1).max()
        x = self.fourier_ext(data, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(data, Lmax)
        mask = mask[:, :Lmax]
        batch_index = mask.nonzero()[:, 0]
        edge_index = knn_graph(x=graph_feature[:, :self.knn_features], k=8, batch=batch_index).to(
            mask.device
        )
        graph_feature = self.dyn_edge(
            graph_feature, edge_index, batch_index, data.n_pulses
        )
        graph_feature, _ = to_dense_batch(graph_feature, batch_index)

        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        x = torch.cat([x, graph_feature], 2)

        for blk in self.sandwich:
            x = blk(x, attn_mask, rel_pos_bias)
            if self.knn_features == 3:
                rel_pos_bias = None
        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        #x = self.proj_out(x[:, 0])  # cls token
        return x[:, 0]