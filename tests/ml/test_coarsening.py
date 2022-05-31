"""Unit tests for loss functions."""

import torch
from torch_geometric.data import Data, Batch

from graphnet.components.pool import group_by
from graphnet.models.coarsening import Coarsening


# Utility method(s)
def _get_test_data() -> Data:
    """Produce toy data for unit tests."""

    x = torch.tensor(
        [
            [1, 0, 0.1],  # 0
            [2, 1, 0.1],  # 1
            [1, 1, 0.2],  # 2
            [2, 1, 0.1],  # 1
            [1, 1, 0.2],  # 2
            [1, 0, 0.1],  # 3
            [2, 1, 0.3],  # 4
            [1, 1, 0.1],  # 5
            [2, 1, 0.3],  # 4
            [1, 1, 0.1],  # 6
        ],
        dtype=torch.float32,
    )

    batch = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            2,
        ],
        dtype=torch.int64,
    )

    attr1 = torch.tensor(
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
        dtype=torch.int32,
    )

    attr2 = torch.tensor(
        [
            1.0,
            2.0,
            2.0,
            3.0,
            4.0,
            1.0,
            4.0,
            3.0,
            1.0,
            6.0,
        ],
        dtype=torch.float32,
    )

    data = Batch(batch=batch, x=x)
    data["x0"] = x[:, 0]
    data["x1"] = x[:, 1]
    data["x2"] = x[:, 2]
    data["attr1"] = attr1
    data["attr2"] = attr2

    return data


class SimpleCoarsening(Coarsening):
    """Simple coarsening operation for the purposes of testing."""

    def _perform_clustering(self, data: Data) -> torch.LongTensor:
        """Perform clustering of nodes in `data` by assigning unique cluster indices to each."""
        return group_by(data, ["x0", "x1", "x2"])


# Unit test(s)
def test_attribute_transfer():
    """Testing the transfering of auxillary attributes during coarsening."""
    # Check(s)
    data = _get_test_data()

    # Perform coarsening
    coarsening = SimpleCoarsening(reduce="avg", transfer_attributes=False)
    pooled_data = coarsening(data)

    # Test(s)
    assert not hasattr(pooled_data, "x0")
    assert not hasattr(pooled_data, "x1")
    assert not hasattr(pooled_data, "x2")
    assert not hasattr(pooled_data, "attr1")
    assert not hasattr(pooled_data, "attr2")

    # Perform coarsening
    coarsening = SimpleCoarsening(reduce="avg", transfer_attributes=True)
    pooled_data = coarsening(data)

    # Test(s)
    assert hasattr(pooled_data, "x0")
    assert hasattr(pooled_data, "x1")
    assert hasattr(pooled_data, "x2")
    assert hasattr(pooled_data, "attr1")
    assert hasattr(pooled_data, "attr2")

    assert pooled_data.x.size(dim=0) == pooled_data["x0"].size(dim=0)
    assert pooled_data.x.size(dim=0) == pooled_data["attr1"].size(dim=0)
