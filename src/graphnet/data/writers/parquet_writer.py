"""DataConverter for the Parquet backend."""

import os
from typing import List, Dict, Any

import pandas as pd
import polars as pol
from glob import glob
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

from .graphnet_writer import GraphNeTWriter


class ParquetWriter(GraphNeTWriter):
    """Class for writing interim data format to Parquet."""

    def __init__(
        self,
        truth_table: str = "truth",
        index_column: str = "event_no",
    ) -> None:
        """Construct `ParquetWriter`.

        Args:
            truth_table: Name of the tables containing event-level truth data.
                         defaults to "truth".
            index_column: The column used for indexation.
                             Defaults to "event_no".
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        # Class variables
        self._file_extension = ".parquet"
        self._merge_dataframes = True
        self._index_column = index_column
        self._truth_table = truth_table

    # Abstract method implementation(s)
    def _save_file(
        self,
        data: Dict[str, pd.DataFrame],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save data to parquet."""
        # Check(s)

        if n_events > 0:
            for table in data.keys():
                save_path = os.path.dirname(output_file_path)
                file_name = os.path.splitext(
                    os.path.basename(output_file_path)
                )[0]

                table_dir = os.path.join(save_path, f"{table}")
                os.makedirs(table_dir, exist_ok=True)
                df = data[table].set_index(self._index_column)
                df.to_parquet(
                    os.path.join(table_dir, file_name + f"_{table}.parquet")
                )

    def merge_files(
        self,
        files: List[str],
        output_dir: str,
        events_per_batch: int = 200000,
        num_workers: int = 1,
    ) -> None:
        """Convert files into shuffled batches.

            Events will be shuffled, and the resulting batches will constitute
            random subsamples of the full dataset.

        Args:
            files: Files converted to parquet. Note this argument is ignored
                    by this method, as these files are automatically found
                    using the `output_dir`.
            output_dir: The directory to store the batched data.
            events_per_batch: Number of events in each batch.
                             Defaults to 200000.
            num_workers: Number of workers to use for merging. Defaults to 1.
        """
        # Handle inputs
        input_dir = output_dir.replace("merged", "")
        truth_dir = os.path.join(input_dir, self._truth_table)
        tables = os.listdir(input_dir)
        self._validate_inputs(
            tables=tables, input_dir=input_dir, truth_dir=truth_dir
        )

        truth_files = glob(os.path.join(truth_dir, "*.parquet"))

        # Exit if no files found
        if len(truth_files) == 0:
            self.warning(f"No files found in {truth_dir}. Exiting.")
            return

        # Produce a shuffled master-list of event_no's
        truth_meta = self._identify_events(
            index_column=self._index_column,
            truth_table=self._truth_table,
            truth_files=truth_files,
        )

        # Split event_nos into smaller batches "shards"
        shards = self._split_dataframe(
            df=truth_meta, chunk_size=events_per_batch
        )

        # Construct list of arguments for processing function
        arguments = []
        for i in range(len(shards)):
            arguments.append(
                [
                    tables,
                    shards[i],
                    input_dir,
                    i,
                    self._index_column,
                    output_dir,
                ]
            )

        # Setup map function
        if num_workers > 1:
            self.info(
                f"Processing {len(arguments)} batches using "
                f"{num_workers} cores."
            )
            pool = Pool(num_workers)
            map_func = pool.imap
        else:
            self.info(f"Processing {len(arguments)} batches in main thread.")
            map_func = map  # type: ignore

        # Process files
        for _ in map_func(
            self._process_shard,
            tqdm(arguments, unit="shard(s)", colour="green"),
        ):
            pass

    def _identify_events(
        self, index_column: str, truth_files: List[str], truth_table: str
    ) -> pd.DataFrame:
        res = pol.DataFrame()
        for truth_file in truth_files:
            df = pol.read_parquet(truth_file)
            df2 = pol.concat(
                [
                    df.select([index_column]),
                    pol.DataFrame(
                        {
                            "file_name": np.repeat(
                                truth_file.replace(
                                    f"_{truth_table}.parquet", ""
                                ).split("/")[-1],
                                len(df),
                            )
                        }
                    ).select(["file_name"]),
                ],
                how="horizontal",
            )
            res = pol.concat([res, df2])
        return res.to_pandas().sample(frac=1.0)

    def _split_dataframe(
        self, df: pd.DataFrame, chunk_size: int
    ) -> List[pd.DataFrame]:
        chunks = list()
        num_chunks = int(np.ceil(len(df) // chunk_size) + 1)
        for i in range(num_chunks):
            chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
        return chunks

    def _process_shard(self, settings: List[Any]) -> None:
        tables, splits, input_dir, batch_ids, index_column, outdir = settings
        if outdir is None:
            outdir = os.path.join(input_dir, "merged")
        if not isinstance(splits, list):
            splits = [splits]

        if not isinstance(batch_ids, list):
            batch_ids = [batch_ids]

        for batch_id, split in zip(batch_ids, splits):
            unique_files = pd.unique(split["file_name"])
            for table in tables:
                table_shards = []
                for unique_file in unique_files:
                    path = (
                        os.path.join(input_dir, table, unique_file)
                        + f"_{table}.parquet"
                    )
                    df = pd.read_parquet(path)

                    id = split[index_column][split["file_name"] == unique_file]

                    # Filter out indices that point to empty events
                    idx = [i for i in id if i in df.index]
                    table_shards.append(df.loc[idx, :])

                os.makedirs(os.path.join(outdir, table), exist_ok=True)
                if len(table_shards) > 0:
                    combined_df = pd.concat(table_shards, axis=0)
                    combined_df.to_parquet(
                        os.path.join(
                            outdir, table, f"{table}_{batch_id}.parquet"
                        )
                    )

    def _validate_inputs(
        self, tables: List[str], input_dir: str, truth_dir: str
    ) -> None:
        try:
            assert "merged" not in tables
        except AssertionError as e:
            self.error(
                f"Directory appears to already contain merged files"
                f" under {os.path.join(input_dir, 'merged')}"
            )
            raise e
        try:
            assert os.path.isdir(truth_dir)
        except AssertionError as e:
            self.error(f"Directory for truth {truth_dir} does not exist.")
            raise e
