"""Classes for performing embedding of input data."""

import torch
import torch.nn as nn
from torch.functional import Tensor

from typing import Optional

from pytorch_lightning import LightningModule
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
from torch_sparse import coalesce

class LinearEdgeEncoder(LightningModule):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.in_dim = 1  # TODO: generalize to more edge features -PW
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, data):
        data.edge_attr = self.encoder(data.edge_attr.view(-1, self.in_dim))
        return data
    
class LinearNodeEncoder(LightningModule):
    def __init__(self, dim_in, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(dim_in, emb_dim)

    def forward(self, data):
        data.x = self.encoder(data.x)
        return data


class SinusoidalPosEmb(LightningModule):
    """Sinusoidal positional embeddings module.

    This module is from the kaggle competition 2nd place solution (see
    arXiv:2310.15674): It performs what is called Fourier encoding or it's used
    in the Attention is all you need arXiv:1706.03762. It can be seen as a soft
    digitization of the input data
    """

    def __init__(
        self,
        dim: int = 16,
        n_freq: int = 10000,
        scaled: bool = False,
    ):
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            n_freq: Number of frequencies.
            scaled: Whether or not to scale the output.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim has to be even. Got: {dim}")
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        self.dim = dim
        self.n_freq = torch.Tensor([n_freq])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        device = x.device
        half_dim = self.dim / 2
        emb = torch.log(self.n_freq.to(device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb * self.scale


class FourierEncoder(LightningModule):
    """Fourier encoder module.

    This module incorporates sinusoidal positional embeddings and auxiliary
    embeddings to process input sequences and produce meaningful
    representations. The module assumes that the input data is in the format of
    (x, y, z, time, charge, auxiliary), being the first four features
    mandatory.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, `mlp_dim` is set automatically as multiples of
                `seq_length` (in consistent with the 2nd place solution),
                depending on `n_features`.
            output_dim: Dimension of the output (I.e. number of columns).
            scaled: Whether or not to scale the embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__()

        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if n_features < 4:
            raise ValueError(
                f"At least x, y, z and time of the DOM are required. Got only "
                f"{n_features} features."
            )
        elif n_features >= 6:
            self.aux_emb = nn.Embedding(2, seq_length // 2)
            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((n_features + 0.5) * seq_length)

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(seq_length.to(dtype=x.dtype))
        embeddings = [self.sin_emb(4096 * x[:, :, :3]).flatten(-2)]  # Position

        if self.n_features >= 5:
            embeddings.append(self.sin_emb(1024 * x[:, :, 4]))  # Charge

        embeddings.append(self.sin_emb(4096 * x[:, :, 3]))  # Time

        if self.n_features >= 6:
            embeddings.append(self.aux_emb(x[:, :, 5].long()))  # Auxiliary

        embeddings.append(
            self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1)
        )  # Length

        x = torch.cat(embeddings, -1)
        x = self.mlp(x)

        return x


class SpacetimeEncoder(LightningModule):
    """Spacetime encoder module."""

    def __init__(
        self,
        seq_length: int = 32,
    ):
        """Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events
        and generates sinusoidal positional embeddings to be added to input
        sequences.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)

    def forward(
        self,
        x: Tensor,
        # Lmax: Optional[int] = None,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:, :, :3]
        time = x[:, :, 3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(
            -1
        ) - ((time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        return rel_attn

# TODO: Put in some utils file -PW
def full_edge_index(edge_index, batch=None):
    """
    Retunr the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full

class RRWPLinearNodeEncoder(LightningModule):
    """
    FC_1(RRWP) + FC_2 (Node-attr)
    note: FC_2 is given by the Typedict encoder of node-attr in some cases
    Parameters:
    num_classes - the number of classes for the embedding mapping to learn
    """
    def __init__(self, emb_dim, out_dim, use_bias=False, batchnorm=False, layernorm=False, pe_name="rrwp"):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm: 
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, data):
        # Encode just the first dimension if more exist
        rrwp = data[f"{self.name}"]
        rrwp = self.fc(rrwp)
        
        if self.batchnorm:
            rrwp = self.bn(rrwp)

        if self.layernorm:
            rrwp = self.ln(rrwp)

        if "x" in data:
            data.x = data.x + rrwp
        else:
            data.x = rrwp

        return data


class RRWPLinearEdgeEncoder(LightningModule):
    '''
    Merge RRWP with given edge-attr and Zero-padding to all pairs of node
    FC_1(RRWP) + FC_2(edge-attr)
    - FC_2 given by the TypedictEncoder in same cases
    - Zero-padding for non-existing edges in fully-connected graph
    - (optional) add node-attr as the E_{i,i}'s attr
        note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, 
                 emb_dim, 
                 out_dim, 
                 batchnorm=False, 
                 layernorm=False, 
                 use_bias=False,
                 pad_to_full_graph=True,
                 fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr

        # TODO: Simplify the batch/layer norm business
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        # if self.batchnorm or self.layernorm:
        #     warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            # edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            edge_index, edge_attr = add_self_loops(edge_index, 
                                                   edge_attr, 
                                                   num_nodes=batch.num_nodes, 
                                                   fill_value=0.)

            out_idx, out_val = coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes, batch.num_nodes,
                op="add"
            )


        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"

