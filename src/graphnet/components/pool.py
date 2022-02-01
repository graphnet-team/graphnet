from typing import Optional

import torch
from torch import LongTensor, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_edge, pool_batch, pool_pos
from torch_scatter import scatter


def sum_pool_and_distribute(tensor: Tensor, cluster_index: LongTensor, batch: Optional[LongTensor] = None) -> Tensor:
    """Sum-pool values across the cluster, and distribute the individual nodes."""
    if batch is None:
        batch = torch.zeros(tensor.size(dim=0)).long()
    tensor_pooled, _ = sum_pool_x(cluster_index, tensor, batch)
    inv, _ = consecutive_cluster(cluster_index)
    tensor_unpooled = tensor_pooled[inv]
    return tensor_unpooled

def group_identical(tensor: Tensor, batch: Optional[LongTensor] = None) -> Tensor:
    if batch is not None:
        tensor = tensor.cat((tensor, batch.unsqueeze(dim=1)), dim=1)
    return torch.unique(tensor, return_inverse=True, dim=0)[1]

def group_pulses_to_dom(data: Data) -> Data:
    tensor = torch.stack((data.dom_number, data.string)).T.int()
    batch = getattr(tensor, 'batch', None)
    data.dom_index = group_identical(tensor, batch)
    return data

def group_pulses_to_pmt(data: Data) -> Data:
    tensor = torch.stack((data.pmt_number, data.dom_number, data.string)).T.int()
    batch = getattr(tensor, 'batch', None)
    data.pmt_index = group_identical(tensor, batch)
    return data

# Below mirroring `torch_geometric.nn.pool.{avg,max}_pool.py` exactly
def _sum_pool_x(cluster, x, size: Optional[int] = None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='sum')


def sum_pool_x(cluster, x, batch, size: Optional[int] = None):
    r"""Sum-Pools node features according to the clustering defined in
    :attr:`cluster`.

    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): The maximum number of clusters in a single
            example. This property is useful to obtain a batch-wise dense
            representation, *e.g.* for applying FC layers, but should only be
            used if the size of the maximum number of clusters per example is
            known in advance. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`LongTensor`) if :attr:`size` is
        :obj:`None`, else :class:`Tensor`
    """
    if size is not None:
        batch_size = int(batch.max().item()) + 1
        return _sum_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _sum_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def sum_pool(cluster, data, transform=None):
    r"""Pools and coarsens a graph given by the
    :class:`torch_geometric.data.Data` object according to the clustering
    defined in :attr:`cluster`.
    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the *sum* of features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster.

    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        data (Data): Graph data object.
        transform (callable, optional): A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version. (default: :obj:`None`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    cluster, perm = consecutive_cluster(cluster)

    x = None if data.x is None else _sum_pool_x(cluster, data.x)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data

#############################################################

def cluster_identical(tensor: Tensor, batch: Optional[LongTensor] = None, eps: float = 1e-03) -> Tensor:
    """Cluster identical rows in `tensor`.

    Args:
        tensor (Tensor): 2D tensor.
        eps (float, optional): Radius-style threshold for clustering rows in
            `tensor`. Defaults to 1e-03.

    Returns:
        Tensor: 1D array of integers assigning each row in `tensor` to a cluter.
    """
    # Check(s)
    assert tensor.ndim == 2

    # By default assign each pulse to a separate DOM
    cluster_index = torch.arange(tensor.size(dim=0), device=tensor.device)

    if batch is not None:
        for ix_batch in torch.unique(batch):
            cluster_index[batch == ix_batch] = cluster_identical(tensor[batch == ix_batch])
        return cluster_index

    # Get pairs
    nb_samples = tensor.size(dim=0)
    mask_triu = torch.ones(nb_samples, nb_samples).triu(1).bool()
    dist = torch.triu((tensor.T.unsqueeze(dim=0) - tensor.unsqueeze(dim=2)).abs().sum(dim=1))
    pairs = torch.stack(torch.where((dist < eps) & mask_triu)).T

    # Connect all pulses within a radius of < eps, and assign them to the same DOM
    for pair in pairs:
        cluster_index[pair[1]] = cluster_index[pair[0]]

    # Ensure that DOMs are squentially numbered from 0 to nb_clusters - 1
    ix_sort = torch.argsort(cluster_index)
    cluster_index_sorted = torch.cumsum(torch.diff(cluster_index[ix_sort], prepend=torch.tensor([0])).clip(0, 1), dim=0)
    cluster_index[ix_sort] = cluster_index_sorted

    return cluster_index

def cluster_pulses_to_dom(data: Data, eps: float = 1e-03) -> Data:
    # Could also use `dom_number` and `string` -- but these only exist for
    # IceCubeUpgrade, not for <= DeepCore

    # Extract spatial coordinates for each pulse
    xyz = torch.stack((data['dom_x'], data['dom_y'], data['dom_z']), dim=1)

    # Add DOM clustering as an attribute to the `Data` object
    data.dom_index = cluster_identical(xyz, getattr(data, 'batch', None))

    return data

def cluster_pulses_to_pmt(data: Data, eps: float = 1e-03) -> Data:
    # Could also use `pmt_number`, `dom_number`, and `string` -- but these only
    # exist for IceCubeUpgrade, not for <= DeepCore

    # Extract spatial coordinates for each pulse
    xyzdir = torch.stack((
            data['dom_x'],
            data['dom_y'],
            data['dom_z'],
            data['pmt_dir_x'],
            data['pmt_dir_y'],
            data['pmt_dir_z'],
        ), dim=1)

    # Add pmt clustering as an attribute to the `Data` object
    data.pmt_index = cluster_identical(xyzdir, getattr(data, 'batch', None))

    return data