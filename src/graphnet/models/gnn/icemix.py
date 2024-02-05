import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math

from graphnet.models.components.layers import Extractor, Spacetime_encoder, Block_rel, Block, ExtractorV11Scaled
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN

from timm.models.layers import trunc_normal_

from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import to_dense_batch


class DeepIceModel(GNN):
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
        self.extractor = Extractor(dim_base, dim)
        self.rel_pos = Spacetime_encoder(head_size)
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

    def forward(self, x0):
        mask = x0.mask
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
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
    
    
class EncoderWithDirectionReconstruction(GNN):
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
        self.extractor = ExtractorV11Scaled(dim_base, dim // 2)
        self.rel_pos = Spacetime_encoder(head_size)
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
        self.local_root = DynEdge(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        )
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0.mask
        graph_feature = torch.concat(
            [
                x0.pos[mask],
                x0.time[mask].view(-1, 1),
                x0.auxiliary[mask].view(-1, 1),
                x0.qe[mask].view(-1, 1),
                x0.charge[mask].view(-1, 1),
                x0.ice_properties[mask],
            ],
            dim=1,
        )
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        batch_index = mask.nonzero()[:, 0]
        edge_index = knn_graph(x=graph_feature[:, :self.knn_features], k=8, batch=batch_index).to(
            mask.device
        )
        graph_feature = self.local_root(
            graph_feature, edge_index, batch_index, x0.n_pulses
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