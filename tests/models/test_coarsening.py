"""Unit tests for Coarsening class."""

import torch
from torch_geometric.data import Data, Batch

from graphnet.models.coarsening import AttributeCoarsening


# Utility method(s)
def _get_test_data() -> Batch:
    """Produce toy data for unit tests."""
    data1 = Data(
        x=torch.tensor(
            [
                [1, 0, 0.1],  # 0
                [2, 1, 0.1],  # 1
                [1, 1, 0.2],  # 2
                [2, 1, 0.1],  # 1
                [1, 1, 0.2],  # 2
            ],
            dtype=torch.float32,
        ),
        attr1=torch.tensor(
            [
                10,
                11,
                12,
                13,
                14,
            ],
            dtype=torch.int32,
        ),
        attr2=torch.tensor(
            [
                1.0,
                2.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=torch.float32,
        ),
    )

    data2 = Data(
        x=torch.tensor(
            [
                [1, 0, 0.1],  # 3
                [2, 1, 0.3],  # 4
                [1, 1, 0.1],  # 5
                [2, 1, 0.3],  # 4
            ],
            dtype=torch.float32,
        ),
        attr1=torch.tensor(
            [
                15,
                16,
                17,
                18,
            ],
            dtype=torch.int32,
        ),
        attr2=torch.tensor(
            [
                1.0,
                4.0,
                3.0,
                1.0,
            ],
            dtype=torch.float32,
        ),
    )

    data3 = Data(
        x=torch.tensor(
            [
                [1, 1, 0.1],  # 6
            ],
            dtype=torch.float32,
        ),
        attr1=torch.tensor(
            [
                19,
            ],
            dtype=torch.int32,
        ),
        attr2=torch.tensor(
            [
                6.0,
            ],
            dtype=torch.float32,
        ),
    )

    data_list = [data1, data2, data3]
    for ix in range(len(data_list)):
        data_list[ix]["x0"] = data_list[ix].x[:, 0]
        data_list[ix]["x1"] = data_list[ix].x[:, 1]
        data_list[ix]["x2"] = data_list[ix].x[:, 2]

    batch = Batch.from_data_list([data1, data2, data3])

    return batch


# Unit test(s)
def test_attribute_transfer() -> None:
    """Testing the transfering of auxillary attributes during coarsening."""
    # Check(s)
    data = _get_test_data()

    # Perform coarsening
    coarsening = AttributeCoarsening(
        attributes=["x0", "x1", "x2"],
        reduce="avg",
        transfer_attributes=False,
    )
    pooled_data = coarsening(data)

    # Test(s)
    assert not hasattr(pooled_data, "x0")
    assert not hasattr(pooled_data, "x1")
    assert not hasattr(pooled_data, "x2")
    assert not hasattr(pooled_data, "attr1")
    assert not hasattr(pooled_data, "attr2")

    # Perform coarsening
    coarsening = AttributeCoarsening(
        attributes=["x0", "x1", "x2"],
        reduce="avg",
        transfer_attributes=True,
    )
    pooled_data = coarsening(data)

    # Test(s)
    assert hasattr(pooled_data, "x0")
    assert hasattr(pooled_data, "x1")
    assert hasattr(pooled_data, "x2")
    assert hasattr(pooled_data, "attr1")
    assert hasattr(pooled_data, "attr2")

    assert pooled_data.x.size(dim=0) == pooled_data["x0"].size(dim=0)
    assert pooled_data.x.size(dim=0) == pooled_data["attr1"].size(dim=0)


def test_batch_reconstruction() -> None:
    """Testing the batch reconstruction."""
    # Check(s)
    data = _get_test_data()
    original_batch_idx = data.batch
    # Perform coarsening
    coarsening = coarsening = AttributeCoarsening(
        attributes=["x0", "x1", "x2"],
        reduce="avg",
        transfer_attributes=False,
    )
    pooled_data = coarsening(data)

    # Check that the number of batches is as expected.
    assert len(torch.unique(original_batch_idx)) == len(
        torch.unique(pooled_data.batch)
    )

    # Check that each event can be recovered
    assert len(pooled_data.to_data_list()) == len(
        torch.unique(original_batch_idx)
    )
    assert all([pooled_data[ix] for ix in range(len(pooled_data))])
