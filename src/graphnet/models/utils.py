"""Utility functions for `graphnet.models`."""

from typing import List, Tuple, Any, Union, Optional

import torch
from torch import Tensor, LongTensor

from torch_geometric.nn import knn_graph
from torch_geometric.data import Batch, Data
from torch_geometric.utils import homophily, degree, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_scatter import scatter, scatter_add
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


def full_edge_index(
    edge_index: Tensor, batch: Optional[Tensor] = None
) -> Tensor:
    """Return the full batched sparse adjacency matrices given by edge indices.

    Return batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops. Implementation
    inspired by `torch_geometric.utils.to_dense_adj`.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/encoder/rrwp_encoder.py

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
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full


@torch.no_grad()
def add_full_rrwp(
    data: Data,
    walk_length: int = 8,
    attr_name_abs: str = "rrwp",  # name: 'rrwp'
    attr_name_rel: str = "rrwp",  # name: ('rrwp_idx', 'rrwp_val')
    add_identity: bool = True,
    spd: bool = False,
) -> Data:
    """Add relative random walk probabilities.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/transform/rrwp.py

    Args:
        data: Input data.
        walk_length: Number of random walks for encoding.
        attr_name_abs: Absolute position encoding name.
        attr_name_rel: Relative position encoding name.
        add_identity: Add identity matrix to position encoding.
        spd: Use shortest path distances.
    """
    # device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=data.x.dtype))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

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
        rel_pe_val = torch.nn.functional.one_hot(val, walk_length).type(
            torch.float
        )
        abs_pe = torch.zeros_like(abs_pe)

    data[attr_name_abs] = abs_pe
    data[f"{attr_name_rel}_index"] = rel_pe_idx
    data[f"{attr_name_rel}_val"] = rel_pe_val

    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data


@torch.no_grad()
def get_log_deg(data: Data) -> Tensor:
    """Get log of the degree number of a graph.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/layer/grit_layer.py
    """
    if "log_deg" in data:
        log_deg = data.log_deg
    elif "deg" in data:
        deg = data.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        deg = degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=data.x.dtype
        )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(data.num_nodes, 1)
    return log_deg


def get_rw_landing_probs(
    ksteps: List,
    edge_index: Tensor,
    edge_weight: Tensor = None,
    num_nodes: Optional[int] = None,
    space_dim: int = 0,
) -> Tensor:
    """Compute Random Walk landing probabilities for given list of K steps.

    Original code:
    https://github.com/ETH-DISCO/Benchmarking-PEs
    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number
            of steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    print(edge_index.shape)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source = edge_index[0]
    # dest = edge_index[1]

    # Out degrees
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        # 1 x (Num nodes) x (Num nodes)
        P = torch.diag(deg_inv) @ to_dense_adj(
            edge_index, max_num_nodes=num_nodes
        )
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(
                torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2))
            )
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(
                torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1)
                * (k ** (space_dim / 2))
            )

    # (Num nodes) x (K steps)
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)
    return rw_landing
