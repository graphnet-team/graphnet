"""Example for EnsembleDataset."""

from timer import timer

import time
import torch.multiprocessing
import torch.utils.data
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from graphnet.constants import TEST_SQLITE_DATA
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data import EnsembleDataset
from graphnet.utilities.logging import get_logger


logger = get_logger()

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE


def main() -> None:
    """Read intermediate file using `Dataset` class."""
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

    logger.info(f"dataset_1 have length: {len(dataset_1)}")
    logger.info(f"dataset_2 have length: {len(dataset_2)}")
    logger.info(f"EnsembleDataset have length {len(ensemble_dataset)}")

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
    main()
