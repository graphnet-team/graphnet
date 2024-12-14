import torch
import torch.nn as nn

from graphnet.models.gnn.gnn import GNN

from graphnet.models.components.layers import GritTransformerLayer, SANGraphHead
from graphnet.models.components.embedding import RRWPLinearEdgeEncoder, RRWPLinearNodeEncoder

class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # if cfg.dataset.name in ['MNIST', 'CIFAR10']:
        #     self.in_dim = 1
        # elif cfg.dataset.name.startswith('attributed_triangle-'):
        #     self.in_dim = 2
        # else:
        #     raise ValueError("Input edge feature dim is required to be hardset "
        #                      "or refactored to use a cfg option.")
        self.in_dim = 1
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
    
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, dim_in, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(dim_in, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch


class GRIT(GNN):
    """
    """
    def __init__(
                 self,
                 nb_inputs,
                 hidden_dim,
                 dim_out=1,
                 ksteps=21,
                 n_layers=10,
                 n_heads=8,
                 pad_to_full_graph=True,
                 add_node_attr_as_self_loop=False,
                 dropout=0.0,
                 fill_value=0.0,
                 layer_norm=False,
                 batch_norm=True,
                 attn_dropout=0.2,
                 full_attn=True,
                 edge_enhance=True,
                 update_e=True,
                 clamp=5.0,
                 signed_sqrt=True,
                 act='relu',
                 attn_act='relu',
                 norm_e=True,
                 O_e=True,
                 ):
        super().__init__(nb_inputs, dim_out)
        
        self.node_encoder = LinearNodeEncoder(nb_inputs, hidden_dim)
        self.edge_encoder = LinearEdgeEncoder(hidden_dim)
        
        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, hidden_dim)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(ksteps, 
                                                      hidden_dim,
                                                      pad_to_full_graph=pad_to_full_graph,
                                                      add_node_attr_as_self_loop=add_node_attr_as_self_loop,
                                                      fill_value=fill_value)

        layers = []
        for _ in range(n_layers):
            layers.append(GritTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                act=act,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                residual=True,
                norm_e=norm_e,
                O_e=O_e,
                cfg={"update_e": update_e,
                     "attn": {
                         "clamp": clamp,
                         "act": attn_act,
                         "full_attn": full_attn,
                         "edge_enhance": edge_enhance,
                         "O_e": O_e,
                         "norm_e": norm_e,
                         "signed_sqrt": signed_sqrt,
                         },
                     },
            ))
        self.layers = nn.ModuleList(layers)
        self.head = SANGraphHead(dim_in=hidden_dim, dim_out=1)

    def forward(self, x):
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