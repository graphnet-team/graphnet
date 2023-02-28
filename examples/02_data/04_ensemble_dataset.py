"""Example of combining multiple Datasets using EnsembleDataset."""

import time
from timer import timer
import torch.multiprocessing
import torch.utils.data
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from graphnet.constants import TEST_SQLITE_DATA
from graphnet.data import EnsembleDataset
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE


def main() -> None:
    """Read intermediate file using `Dataset` class."""
    # Construct Logger
    logger = Logger()

    # Check(s)
    pulsemap = "SRTInIcePulses"
    truth_table = "truth"
    batch_size = 5
    num_workers = 1  # 30
    wait_time = 0.00  # sec.

    # Common variables
    dataset_1 = SQLiteDataset(
        TEST_SQLITE_DATA,
        pulsemap,
        features,
        truth,
        truth_table=truth_table,
    )

    dataset_2 = SQLiteDataset(
        TEST_SQLITE_DATA,
        pulsemap,
        features,
        truth,
        truth_table=truth_table,
    )

    ensemble_dataset = EnsembleDataset(datasets=[dataset_1, dataset_2])

    logger.info(f"dataset_1 has length: {len(dataset_1)}")
    logger.info(f"dataset_2 has length: {len(dataset_2)}")
    logger.info(f"EnsembleDataset has length {len(ensemble_dataset)}")

    dataloader = torch.utils.data.DataLoader(
        ensemble_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        prefetch_factor=2,
    )

    with timer("torch dataloader"):
        for batch in tqdm(dataloader, unit=" batches", colour="green"):
            time.sleep(wait_time)

    for i in range(batch_size):
        logger.info(f"Event {i} came from {batch['dataset_path'][i]}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""
Combine multiple Datasets using EnsembleDataset.
"""
    )
    main()
