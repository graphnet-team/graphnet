"""Functions for handling event level DOM data."""
from typing import Tuple, Optional
from torch_geometric.nn.pool import knn_graph

import torch


@torch.jit.script
def append_dom_id(
    tensor: torch.Tensor,
    batch: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Assign a unique ID to each DOM.

    The ID is assigned based on the position of the DOM in the input tensor. requires x,y,z as the first three columns of the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        batch (torch.Tensor, optional): Batch tensor. Defaults to None.
        device (torch.device): Device to use.

    Returns:
        torch.Tensor: Tensor of DOM IDs.
    """
    inverse_matrix = torch.zeros(
        (tensor.shape[0], 3), device=device, dtype=torch.int64
    )
    for i in range(3):
        _, inverse = torch.unique(tensor[:, i], return_inverse=True)
        inverse_matrix[:, i] = inverse

    if batch is None:
        batch = torch.zeros(tensor.shape[0], device=device, dtype=torch.int64)

    inverse_matrix = torch.hstack([batch.unsqueeze(1), inverse_matrix])
    for i in range(3):
        inverse_matrix[:, 1] = inverse_matrix[:, 0] + (
            (torch.max(inverse_matrix[:, 0]) + 1) * (inverse_matrix[:, 1] + 1)
        )

        _, inverse_matrix[:, 1] = torch.unique(
            inverse_matrix[:, 1], return_inverse=True
        )

        inverse_matrix = inverse_matrix[:, -(3 - i) :]

    inverse_matrix = inverse_matrix.flatten()
    tensor = torch.hstack([tensor, inverse_matrix.unsqueeze(1)])
    return tensor


torch.jit.script


def DOM_to_time_series(
    tensor: torch.Tensor, batch: torch.tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create time series for each activated DOM from dom activation data.

    Returns a time series view of the DOM activations as well as an updated batch tensor. REQURES the DOM ID to be the last column of the input tensor.
    Args:
        tensor (torch.Tensor): Input tensor.
        batch (torch.Tensor): Batch tensor.

    Returns:
        tensor (torch.Tensor): Times series DOM data.
        sort_batch (torch.Tensor): Batch tensor.
    """
    dom_activation_sort = tensor[:, -1].sort()[-1]
    tensor, batch = tensor[dom_activation_sort], batch[dom_activation_sort]
    bin_count = torch.bincount(tensor[:, -1].type(torch.int64)).cumsum(0)
    batch = batch[bin_count - 1]
    tensor = tensor[:, :-1]
    tensor = torch.tensor_split(tensor, bin_count.cpu()[:-1])
    lengths_index = (
        torch.as_tensor([v.size(0) for v in tensor]).sort()[-1].flip(0)
    )
    batch = batch[lengths_index]

    return tensor, batch


torch.jit.script


def DOM_time_series_to_pack_sequence(
    tensor: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packs DOM time series data into a pack sequence.

    Also returns a tensor of x/y/z-coordinates and time of first/mean activation for each DOM.

    Args:
        tensor (torch.Tensor): Input tensor.
        device (torch.device): Device to use.

    Returns:
        tensor (torch.Tensor): Packed sequence of DOM time series data.
        xyztt (torch.Tensor): Tensor of x/y/z-coordinates as well as time of mean & first activations for each DOM.
    """
    tensor = sorted_jit_ignore(tensor)
    xyztt = torch.stack(
        [
            torch.cat(
                [
                    v[0, :3],
                    torch.as_tensor(
                        [v[:, 4].mean(), v[:, 4].min()], device=device
                    ),
                ]
            )
            for v in tensor
        ]
    )

    tensor = torch.nn.utils.rnn.pack_sequence(tensor, enforce_sorted=True)
    return tensor, xyztt


@torch.jit.ignore
def sorted_jit_ignore(tensor: torch.Tensor) -> torch.Tensor:
    """Sort a tensor based on the length of the elements.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Sorted tensor.
        torch.Tensor: Indices of sorted tensor.
    """
    tensor = sorted(tensor, key=len, reverse=True)
    return tensor


@torch.jit.ignore
def knn_graph_ignore(
    x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Create a kNN graph based on the input data.

    Args:
        x: Input data.
        k: Number of neighbours.
        batch: Batch index.
    """
    return knn_graph(x=x, k=k, batch=batch)
