"""Functions for performing pooling/clustering/coarsening."""

from typing import Any, Callable, List, Optional, Union

import torch
from torch import LongTensor, Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_edge, pool_batch, pool_pos
from torch_scatter import scatter, scatter_std

from torch_geometric.nn.pool import (  # noqa:F401
    max_pool,
    max_pool_x,
    avg_pool_x,
    avg_pool,
)


def min_pool(
    cluster: LongTensor, data: Data, transform: Optional[Any] = None
) -> Data:
    """Perform min-pooling of `Data`.

    Like `max_pool, just negating `data.x`.
    """
    data.x = -data.x
    data_pooled = max_pool(
        cluster,
        data,
        transform,
    )
    data.x = -data.x
    data_pooled.x = -data_pooled.x
    return data_pooled


def min_pool_x(
    cluster: LongTensor,
    x: Tensor,
    batch: LongTensor,
    size: Optional[int] = None,
) -> Tensor:
    """Perform min-pooling of `Tensor`.

    Like `max_pool_x, just negating `x`.
    """
    ret = max_pool_x(cluster, -x, batch, size)
    if size is None:
        return (-ret[0], ret[1])
    else:
        return -ret


def sum_pool_and_distribute(
    tensor: Tensor,
    cluster_index: LongTensor,
    batch: Optional[LongTensor] = None,
) -> Tensor:
    """Sum-pool values and distribute result to the individual nodes."""
    if batch is None:
        batch = torch.zeros(tensor.size(dim=0)).long()
    tensor_pooled, _ = sum_pool_x(cluster_index, tensor, batch)
    inv, _ = consecutive_cluster(cluster_index)
    tensor_unpooled = tensor_pooled[inv]
    return tensor_unpooled


def _group_identical(
    tensor: Tensor, batch: Optional[LongTensor] = None
) -> LongTensor:
    """Group rows in `tensor` that are identical.

    Args:
        tensor: Tensor of shape [N, F].
        batch: Batch indices, to only group identical rows within batches.

    Returns:
        List of group indices, from 0 to num. groups - 1, assigning all
            identical rows to the same group.
    """
    if batch is not None:
        tensor = torch.cat((batch.unsqueeze(dim=1), tensor), dim=1)
    return torch.unique(tensor, return_inverse=True, sorted=False, dim=0)[1]


def group_by(data: Union[Data, Batch], keys: List[str]) -> LongTensor:
    """Group nodes in `data` that have identical values of `keys`.

    This grouping is done with in each event in case of batching. This allows
    for, e.g., assigning the same index to all pulses on the same PMT or DOM in
    the same event. This can be used for coarsening graphs, e.g., from pulse-
    level to DOM-level by aggregating feature across each group returned by
    this method.

    Example:
      Given:
        data.f1 = [1,1,2,2,2]
        data.f2 = [6,7,7,7,8]
      Calls:
        groupby(data, ['f1'])       -> [0, 0, 1, 1, 1]
        groupby(data, ['f2'])       -> [0, 1, 1, 1, 2]
        groupby(data, ['f1', 'f2']) -> [0, 1, 2, 2, 3]
    """
    features = [getattr(data, key) for key in keys]
    tensor = torch.stack(features).T  # .int()  @TODO: Required? Use rounding?
    batch = getattr(data, "batch", None)
    index = _group_identical(tensor, batch)
    return index


def group_pulses_to_dom(data: Data) -> Data:
    """Group pulses on the same DOM, using DOM and string number."""
    data.dom_index = group_by(data, ["dom_number", "string"])
    return data


def group_pulses_to_pmt(data: Data) -> Data:
    """Group pulses on the same PMT, using PMT, DOM, and string number."""
    data.pmt_index = group_by(data, ["pmt_number", "dom_number", "string"])
    return data


# Below mirroring `torch_geometric.nn.pool.{avg,max}_pool.py`.
def _sum_pool_x(
    cluster: LongTensor, x: Tensor, size: Optional[int] = None
) -> Tensor:
    return scatter(x, cluster, dim=0, dim_size=size, reduce="sum")


def _std_pool_x(
    cluster: LongTensor, x: Tensor, size: Optional[int] = None
) -> Tensor:
    return scatter_std(x, cluster, dim=0, dim_size=size, unbiased=False)


def sum_pool_x(
    cluster: LongTensor,
    x: Tensor,
    batch: LongTensor,
    size: Optional[int] = None,
) -> Tensor:
    r"""Sum-pool node features according to the cluster defined in `cluster`.

    Args:
        cluster: Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        x: Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch: Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size: The maximum number of clusters in a single
            example. This property is useful to obtain a batch-wise dense
            representation, *e.g.* for applying FC layers, but should only be
            used if the size of the maximum number of clusters per example is
            known in advance.
    """
    if size is not None:
        batch_size = int(batch.max().item()) + 1
        return _sum_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _sum_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def std_pool_x(
    cluster: LongTensor,
    x: Tensor,
    batch: LongTensor,
    size: Optional[int] = None,
) -> Tensor:
    r"""Std-pool node features according to the cluster defined in `cluster`.

    Args:
        cluster: Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        x: Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch: Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size: The maximum number of clusters in a single
            example. This property is useful to obtain a batch-wise dense
            representation, *e.g.* for applying FC layers, but should only be
            used if the size of the maximum number of clusters per example is
            known in advance.
    """
    if size is not None:
        batch_size = int(batch.max().item()) + 1
        return _std_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _std_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def sum_pool(
    cluster: LongTensor, data: Data, transform: Optional[Callable] = None
) -> Data:
    r"""Pool and coarsen graph according to the cluster defined in `cluster`.

    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the *sum* of features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster.

    Args:
        cluster: Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        data: Graph data object.
        transform: A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version.
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


def std_pool(
    cluster: LongTensor, data: Data, transform: Optional[Callable] = None
) -> Data:
    r"""Pool and coarsen graph according to the cluster defined in `cluster`.

    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the *std* of features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster.

    Args:
        cluster: Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        data: Graph data object.
        transform: A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version.
    """
    cluster, perm = consecutive_cluster(cluster)

    x = None if data.x is None else _std_pool_x(cluster, data.x)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)

    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    if transform is not None:
        data = transform(data)

    return data
