"""Test ERDAHosted Datasets."""

import pytest

from graphnet.models.detector.prometheus import ORCA150
from graphnet.models.graphs import KNNGraph
from graphnet.datasets import TestDataset
from graphnet.training.utils import DataLoader
from graphnet.constants import TEST_DATA_DIR


def iterate_dataloader(dataloader: DataLoader) -> None:
    """Loop through dataloader to confirm no errors occur."""
    for batch in dataloader:
        assert len(batch.x.shape) == 2
    return


@pytest.mark.parametrize("backend", ["sqlite"])
def test_erda_hosted_dataset(backend: str) -> None:
    """Test TestDataset.

       This test verifies that the data can be downloaded, unzipped,
       that the .description() call works, and that the resulting dataloaders
       functions.

    Args:
        backend: backend to use in Dataset.
    """
    graph_definition = KNNGraph(detector=ORCA150())
    data_module = TestDataset(
        download_dir=TEST_DATA_DIR,
        graph_definition=graph_definition,
        backend=backend,
        train_dataloader_kwargs={"batch_size": 3, "num_workers": 1},
    )

    data_module.description()

    iterate_dataloader(data_module.train_dataloader)
    iterate_dataloader(data_module.val_dataloader)
    return
