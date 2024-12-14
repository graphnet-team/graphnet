"""Utility functions for `graphnet.models`."""

from typing import List, Tuple, Any, Union, Optional
from torch_geometric.nn import knn_graph
from torch_geometric.data import Batch, Data
import torch
from torch import Tensor, LongTensor

from torch_geometric.utils import homophily
from torch_sparse import SparseTensor


def calculate_xyzt_homophily(
    x: Tensor, edge_index: LongTensor, batch: Batch
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate xyzt-homophily from a batch of graphs.

    Homophily is a graph scalar quantity that measures the likeness of
    variables in nodes. Notice that this calculator assumes a special order of
    input features in x.

    Returns:
        Tuple, each element with shape [batch_size,1].
    """
    hx = homophily(edge_index, x[:, 0], batch).reshape(-1, 1)
    hy = homophily(edge_index, x[:, 1], batch).reshape(-1, 1)
    hz = homophily(edge_index, x[:, 2], batch).reshape(-1, 1)
    ht = homophily(edge_index, x[:, 3], batch).reshape(-1, 1)
    return hx, hy, hz, ht


def calculate_distance_matrix(xyz_coords: Tensor) -> Tensor:
    """Calculate the matrix of pairwise distances between pulses.

    Args:
        xyz_coords: (x,y,z)-coordinates of pulses, of shape [nb_doms, 3].

    Returns:
        Matrix of pairwise distances, of shape [nb_doms, nb_doms]
    """
    diff = xyz_coords.unsqueeze(dim=2) - xyz_coords.T.unsqueeze(dim=0)
    return torch.sqrt(torch.sum(diff**2, dim=1))


def knn_graph_batch(batch: Batch, k: List[int], columns: List[int]) -> Batch:
    """Calculate k-nearest-neighbours with individual k for each batch event.

    Args:
        batch: Batch of events.
        k: A list of k's.
        columns: The columns of Data.x used for computing the distances. E.g.,
            Data.x[:,[0,1,2]]

    Returns:
        Returns the same batch of events, but with updated edges.
    """
    data_list = batch.to_data_list()
    for i in range(len(data_list)):
        data_list[i].edge_index = knn_graph(
            x=data_list[i].x[:, columns], k=k[i]
        )
    return Batch.from_data_list(data_list)


def array_to_sequence(
    x: Tensor,
    batch_idx: LongTensor,
    padding_value: Any = 0,
    excluding_value: Any = torch.inf,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert `x` of shape [n,d] into a padded sequence of shape [B, L, D].

       Where B is the batch size, L is the sequence length and D is the
       features for each time step.

    Args:
        x: array-like tensor with shape `[n,d]` where `n` is the total number
        of pulses in the batch and `d` is the number of  node features.
        batch_idx: a LongTensor identifying which row in `x` belongs to
                which training example.
                E.g. `torch_geometric.data.Batch.batch`.
        padding_value: The value to use for padding.
        excluding_value: This parameter represents a unique value that should
                not be present in the input tensor 'x'
    Returns:
        x: Padded sequence with dimensions  [B, L, D].
        mask: A tensor that identifies masked entries in `x`.
               E.g. : `masked_entries = x[mask]`
        seq_length: A tensor containing the number of pulses in each event.
    """
    if torch.any(torch.eq(x, excluding_value)):
        raise ValueError(
            f"Transformation cannot be made because input tensor "
            f"`x` contains at least one element equal to "
            f"excluding value {excluding_value}."
        )

    _, seq_length = torch.unique(batch_idx, return_counts=True)
    x_list = torch.split(x, seq_length.tolist())

    x = torch.nn.utils.rnn.pad_sequence(
        x_list, batch_first=True, padding_value=excluding_value
    )
    mask = torch.ne(x[:, :, 1], excluding_value)
    x[~mask] = padding_value
    return x, mask, seq_length


def get_fields(data: Union[Data, List[Data]], fields: List[str]) -> Tensor:
    """Extract named fields in Data object."""
    labels = []
    if not isinstance(data, list):
        data = [data]
    for label in list(fields):
        labels.append(
            torch.cat([d[label].reshape(-1, 1) for d in data], dim=0)
        )
    return torch.cat(labels, dim=1)

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

@torch.no_grad()
def add_full_rrwp(data,
                  walk_length=8,
                  attr_name_abs="rrwp", # name: 'rrwp'
                  attr_name_rel="rrwp", # name: ('rrwp_idx', 'rrwp_val')
                  add_identity=True,
                  spd=False,
                  **kwargs
                  ):
    device=data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes),
                                       )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)
    # the framework of GRIT performing right-mul while adj is row-normalized, 
    #                 need to switch the order or row and col.
    #    note: both can work but the current version is more reasonable.


    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data