"""Unit tests for minkowski based edges."""

import torch
from torch_geometric.data.data import Data

from graphnet.models.graphs.edges import MinkowskiKNNEdges
from graphnet.models.graphs.edges.minkowski import (
    compute_minkowski_distance_mat,
)


def test_compute_minkowski_distance_mat() -> None:
    """Testing the computation of the Minkowski distance matrix."""
    vec1 = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                1.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
                1.0,
                2.0,
            ],
        ]
    )
    vec2 = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
            [
                1.0,
                1.0,
                1.0,
                0.0,
            ],
        ]
    )
    expected11 = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.0,
                -2.0,
            ],
            [
                0.0,
                0.0,
                2.0,
                0.0,
            ],
            [
                0.0,
                2.0,
                0.0,
                0.0,
            ],
            [
                -2.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    expected12 = torch.tensor(
        [[-1.0, 3.0], [-3.0, 1.0], [-3.0, 1.0], [-7.0, -3.0]]
    )
    expected22 = torch.tensor(
        [
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )
    mat11 = compute_minkowski_distance_mat(vec1, vec1, c=1.0)
    mat12 = compute_minkowski_distance_mat(vec1, vec2, c=1.0)
    mat22 = compute_minkowski_distance_mat(vec2, vec2, c=1.0)

    assert torch.allclose(mat11, expected11)
    assert torch.allclose(mat12, expected12)
    assert torch.allclose(mat22, expected22)


def test_minkowski_knn_edges() -> None:
    """Testing the minkowski knn edge definition."""
    data = Data(
        x=torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                [
                    1.0,
                    0.0,
                    1.0,
                    2.0,
                ],
            ]
        )
    )
    edge_index = MinkowskiKNNEdges(
        nb_nearest_neighbours=2,
        c=1.0,
    )(data).edge_index
    expected = torch.tensor(
        [
            [1, 2, 0, 3, 0, 3, 1, 2],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    assert torch.allclose(edge_index[1], expected[1])

    # Allow for "permutation of connections" in edge_index[1]
    assert torch.allclose(
        edge_index[0, [0, 1]], expected[0, [0, 1]]
    ) or torch.allclose(edge_index[1, [0, 1]], expected[1, [1, 0]])
    assert torch.allclose(
        edge_index[0, [2, 3]], expected[0, [2, 3]]
    ) or torch.allclose(edge_index[1, [2, 3]], expected[1, [3, 2]])
    assert torch.allclose(
        edge_index[0, [4, 5]], expected[0, [4, 5]]
    ) or torch.allclose(edge_index[1, [4, 5]], expected[1, [5, 4]])
    assert torch.allclose(
        edge_index[0, [6, 7]], expected[0, [6, 7]]
    ) or torch.allclose(edge_index[1, [6, 7]], expected[1, [7, 6]])
