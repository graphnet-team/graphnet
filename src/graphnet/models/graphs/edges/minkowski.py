"""Module containing EdgeDefinitions based on the Minkowski Metric."""

from typing import Optional, List

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from graphnet.models.graphs.edges.edges import EdgeDefinition


def compute_minkowski_distance_mat(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float,
    space_coords: Optional[List[int]] = None,
    time_coord: Optional[int] = 3,
) -> torch.Tensor:
    """Compute all pairwise Minkowski distances.

    Args:
        x: First tensor of shape (n, d).
        y: Second tensor of shape (m, d).
        c: Speed of light, in scaled units.
        space_coords: Indices of space coordinates.
        time_coord: Index of time coordinate.

    Returns: Matrix of shape (n, m) of all pairwise Minkowski distances.
    """
    space_coords = space_coords or [0, 1, 2]
    assert x.dim() == 2, "x must be 2-dimensional"
    assert y.dim() == 2, "x must be 2-dimensional"
    dist = x[:, None] - y[None, :]
    pos = dist[:, :, space_coords]
    time = dist[:, :, time_coord] * c
    return (pos**2).sum(dim=-1) - time**2


class MinkowskiKNNEdges(EdgeDefinition):
    """Builds edges between most light-like separated."""

    def __init__(
        self,
        nb_nearest_neighbours: int,
        c: float,
        time_like_weight: float = 1.0,
        space_coords: Optional[List[int]] = None,
        time_coord: Optional[int] = 3,
    ):
        """Initialize MinkowskiKNNEdges.

        Args:
            nb_nearest_neighbours: Number of neighbours to connect to.
            c: Speed of light, in scaled units.
            time_like_weight: Preference to time-like over space-like edges.
                Scales time_like distances by this value, before finding
                nearest neighbours.
            space_coords: Coordinates of x, y, z.
            time_coord: Coordinate of time.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self.nb_nearest_neighbours = nb_nearest_neighbours
        self.c = c
        self.time_like_weight = time_like_weight
        self.space_coords = space_coords or [0, 1, 2]
        self.time_coord = time_coord

    def _construct_edges(self, graph: Data) -> Data:
        x, mask = to_dense_batch(graph.x, graph.batch)
        count = 0
        row = []
        col = []
        for batch in range(x.shape[0]):
            x_masked = x[batch][mask[batch]]
            distance_mat = compute_minkowski_distance_mat(
                x=x_masked,
                y=x_masked,
                c=self.c,
                space_coords=self.space_coords,
                time_coord=self.time_coord,
            )
            num_points = x_masked.shape[0]
            num_edges = min(self.nb_nearest_neighbours, num_points)
            col += [
                c
                for c in range(num_points)
                for _ in range(count, count + num_edges)
            ]
            distance_mat[distance_mat < 0] *= -self.time_like_weight
            distance_mat += (
                torch.eye(distance_mat.shape[0]) * 1e9
            )  # self-loops
            distance_sorted = distance_mat.argsort(dim=1)
            distance_sorted += count  # offset by previous events
            row += distance_sorted[:num_edges].flatten().tolist()
            count += num_points

        graph.edge_index = torch.tensor(
            [row, col], dtype=torch.long, device=graph.x.device
        )
        return graph
