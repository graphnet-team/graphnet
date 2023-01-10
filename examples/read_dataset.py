"""Example of reading events from Dataset class."""

from timer import timer

import awkward
import sqlite3
import time
import torch.multiprocessing
import torch.utils.data
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import Dataset
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.parquet.parquet_dataset import ParquetDataset
from graphnet.utilities.logging import get_logger


logger = get_logger()

torch.multiprocessing.set_sharing_strategy("file_system")

DATASET_CLASS = {
    "sqlite": SQLiteDataset,
    "parquet": ParquetDataset,
}

# Constants
# features = FEATURES.UPGRADE  # From I3FeatureExtractor
features = [  # From I3GenericExtractor
    "position__x",
    "position__y",
    "position__z",
    "time",
    "charge",
    "relative_dom_eff",
    "area",
]
truth = TRUTH.UPGRADE


def main(backend: str) -> None:
    """Read intermediate file using `Dataset` class."""
    # Check(s)
    assert backend in DATASET_CLASS

    suffix = {
        "sqlite": "db",
        "parquet": "parquet",
    }[backend]

    path = f"./temp/test_ic86/oscNext_genie_level7_v03.01_pass2.160000.000001.{suffix}"
    pulsemap = "SRTInIcePulses"
    truth_table = "truth"
    batch_size = 128
    num_workers = 30  # 30
    wait_time = 0.00  # sec.

    for table in [pulsemap, truth_table]:
        # Get column names from backend
        if backend == "sqlite":
            with sqlite3.connect(path) as conn:
                cursor = conn.execute(f"SELECT * FROM {table} LIMIT 1")
                names = list(map(lambda x: x[0], cursor.description))
        else:
            ak = awkward.from_parquet(path, lazy=True)
            names = ak[table].fields
            del ak

        # Print
        logger.info(f"Available columns in {table}")
        for name in names:
            logger.info(f"  . {name}")

    # Common variables
    dataset = DATASET_CLASS[backend](
        path,
        pulsemap,
        features,
        truth,
        truth_table=truth_table,
    )
    assert isinstance(dataset, Dataset)

    logger.info(dataset[1])
    logger.info(dataset[1].x)
    if backend == "sqlite":
        assert isinstance(dataset, SQLiteDataset)
        dataset._close_connection()  # This is necessary iff `dataset` has been indexed between instantiation and passing to `DataLoader`

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        # persistent_workers=True,
        prefetch_factor=2,
    )

    with timer("torch dataloader"):
        for batch in tqdm(dataloader, unit=" batches", colour="green"):
            time.sleep(wait_time)

    logger.info(batch)
    logger.info(batch.size())
    logger.info(batch.num_graphs)


if __name__ == "__main__":
    backend = "parquet"
    backend = "sqlite"
    main(backend)
