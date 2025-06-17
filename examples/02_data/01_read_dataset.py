"""Example of reading events from Dataset class."""

import sqlite3
import time
import torch
from torch_geometric.data.batch import Batch
from tqdm import tqdm
import pandas as pd
import os

from graphnet.constants import TEST_PARQUET_DATA, TEST_SQLITE_DATA
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import Dataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import (
    IceCubeDeepCore,
)

DATASET_CLASS = {
    "sqlite": SQLiteDataset,
    "parquet": ParquetDataset,
}

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE


def main(backend: str) -> None:
    """Read intermediate file using `Dataset` class."""
    # Construct Logger
    logger = Logger()

    # Check(s)
    assert backend in DATASET_CLASS

    path = TEST_SQLITE_DATA if backend == "sqlite" else TEST_PARQUET_DATA
    pulsemap = "SRTInIcePulses"
    truth_table = "truth"
    batch_size = 128
    num_workers = 30
    wait_time = 0.00  # sec.

    # Define data representation
    data_representation = KNNGraph(
        detector=IceCubeDeepCore(), input_feature_names=features
    )

    for table in [pulsemap, truth_table]:
        # Get column names from backend
        if backend == "sqlite":
            with sqlite3.connect(path) as conn:
                cursor = conn.execute(f"SELECT * FROM {table} LIMIT 1")
                names = list(map(lambda x: x[0], cursor.description))
        else:
            df = pd.DataFrame(os.path.join(path, f"{table}*.parquet"))
            names = df.columns.tolist()
            # ak = awkward.from_parquet(path, lazy=True)
            # names = ak[table].fields
            # del ak

        # Print
        logger.info(f"Available columns in {table}")
        for name in names:
            logger.info(f"  . {name}")

    # Common variables
    dataset = DATASET_CLASS[backend](
        path=path,
        pulsemaps=pulsemap,
        features=features,
        truth=truth,
        truth_table=truth_table,
        data_representation=data_representation,
    )
    assert isinstance(dataset, Dataset)

    logger.info(str(dataset[1]))
    logger.info(dataset[1].x)
    if backend == "sqlite":
        assert isinstance(dataset, SQLiteDataset)
        # This is necessary iff `dataset` has been indexed between
        # instantiation and passing to `DataLoader`.
        dataset._close_connection()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        prefetch_factor=2,
    )

    for batch in tqdm(dataloader, unit=" batches", colour="green"):
        time.sleep(wait_time)

    logger.info(str(batch))
    logger.info(batch.size())
    logger.info(batch.num_graphs)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Read a few events from data in an intermediate format.
"""
    )

    parser.add_argument(
        "backend",
        choices=["sqlite", "parquet"],
        default="sqlite",
        const="sqlite",
        nargs="?",
    )

    args, unknown = parser.parse_known_args()

    main(args.backend)
