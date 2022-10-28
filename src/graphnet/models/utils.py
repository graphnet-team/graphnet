"""Utility functions for `graphnet.models`."""

from typing import List, Tuple
from torch_geometric.nn import knn_graph
from torch_geometric.data import Batch
import torch
from torch import Tensor, LongTensor

from torch_geometric.utils.homophily import homophily


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
