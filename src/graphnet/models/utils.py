"""Utility functions for `graphnet.models`."""

from typing import List, Tuple, Any, Union
from torch_geometric.nn import knn_graph
from torch_geometric.data import Batch
import torch
from torch import Tensor, LongTensor

from torch_geometric.utils import homophily
from torch_geometric.data import Data


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
