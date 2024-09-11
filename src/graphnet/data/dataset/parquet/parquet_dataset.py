"""Base :py:class:`Dataset` class(es) used in GraphNeT."""

from typing import (
    Dict,
    List,
    Optional,
    Union,
    Any,
)

import numpy as np
import torch
import os
from torch_geometric.data import Data
import polars as pol
from polars.series.series import Series
from glob import glob
from bisect import bisect_right
from collections import OrderedDict

from graphnet.models.graphs import GraphDefinition
from graphnet.data.dataset import Dataset
from graphnet.exceptions.exceptions import ColumnMissingException


class ParquetDataset(Dataset):
    """Dataset class for Parquet-files converted with `ParquetWriter`."""

    def __init__(
        self,
        path: str,
        graph_definition: GraphDefinition,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
        cache_size: int = 1,
        labels: Optional[Dict[str, Any]] = None,
    ):
        """Construct Dataset.

            NOTE: DataLoaders using this Dataset should have
            "multiprocessing_context = 'spawn'" set to avoid thread locking.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: The batch ids to include in the dataset.
                        Defaults to None, meaning that batches are read.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection (e.g.,
                `"10000 random events ~ event_no % 5 > 0"` or `"20% random
                events ~ event_no % 5 > 0"`).
            graph_definition: Method that defines the graph representation.
            cache_size: Number of files to cache in memory.
                        Must be at least 1. Defaults to 1.
            labels: Dictionary of labels to be added to the dataset.
        """
        self._validate_selection(selection)
        # Base class constructor
        super().__init__(
            path=path,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            node_truth=node_truth,
            index_column=index_column,
            truth_table=truth_table,
            node_truth_table=node_truth_table,
            string_selection=string_selection,
            selection=selection,
            dtype=dtype,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
            seed=seed,
            graph_definition=graph_definition,
            labels=labels,
        )

        # mypy..
        assert isinstance(self._path, str)
        self._path: str = self._path
        # Member Variables
        self._cache_size = cache_size
        self._chunk_sizes = self._calculate_sizes()
        self._chunk_cumsum = np.cumsum(self._chunk_sizes)
        self._file_cache = self._initialize_file_cache(
            truth_table=truth_table,
            node_truth_table=node_truth_table,
            pulsemaps=pulsemaps,
        )
        self._string_selection = string_selection
        # Purely internal member variables
        self._missing_variables: Dict[str, List[str]] = {}
        self._remove_missing_columns()

    def _initialize_file_cache(
        self,
        truth_table: str,
        node_truth_table: Optional[str],
        pulsemaps: Union[str, List[str]],
    ) -> Dict[str, OrderedDict]:
        tables = [truth_table]
        if node_truth_table is not None:
            tables.append(node_truth_table)
        if isinstance(pulsemaps, str):
            tables.append(pulsemaps)
        elif isinstance(pulsemaps, list):
            tables.extend(pulsemaps)

        cache: Dict[str, OrderedDict] = {}
        for table in tables:
            cache[table] = OrderedDict()
        return cache

    def _validate_selection(
        self,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
    ) -> None:
        if selection is not None:
            try:
                assert not isinstance(selection, str)
            except AssertionError:
                e = AssertionError(
                    f"{self.__class__.__name__} does not support "
                    "str-selections."
                )
                raise e

    def _init(self) -> None:
        return

    def _get_event_index(self, sequential_index: int) -> int:
        event_index = self.query_table(
            table=self._truth_table,
            sequential_index=sequential_index,
            columns=[self._index_column],
        )
        return event_index

    @property
    def chunk_sizes(self) -> List[int]:
        """Return a list of the chunk sizes."""
        return self._chunk_sizes

    def __len__(self) -> int:
        """Return length of dataset, i.e. number of training examples."""
        return sum(self._chunk_sizes)

    def _get_all_indices(self) -> List[int]:
        """Return a list of all unique values in `self._index_column`."""
        files = glob(os.path.join(self._path, self._truth_table, "*.parquet"))
        return np.arange(0, len(files), 1)

    def _calculate_sizes(self) -> List[int]:
        """Calculate the number of events in each chunk."""
        sizes = []
        for chunk_id in self._indices:
            path = os.path.join(
                self._path,
                self._truth_table,
                f"{self.truth_table}_{chunk_id}.parquet",
            )
            sizes.append(len(pol.read_parquet(path)))
        return sizes

    def _get_row_idx(self, sequential_index: int) -> int:
        """Return the row index corresponding to a `sequential_index`."""
        file_idx = bisect_right(self._chunk_cumsum, sequential_index)
        if file_idx > 0:
            idx = int(sequential_index - self._chunk_cumsum[file_idx - 1])
        else:
            idx = sequential_index
        return idx

    def query_table(  # type: ignore
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """
        if isinstance(columns, str):
            columns = [columns]

        if sequential_index is None:
            file_idx = np.arange(0, len(self._chunk_cumsum), 1)
        else:
            file_idx = [bisect_right(self._chunk_cumsum, sequential_index)]

        file_indices = [self._indices[idx] for idx in file_idx]

        arrays = []
        for file_idx in file_indices:
            array = self._query_table(
                table=table,
                columns=columns,
                file_idx=file_idx,
                sequential_index=sequential_index,
                selection=selection,
            )
            arrays.append(array)
        return np.concatenate(arrays, axis=0)

    def _query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        file_idx: int,
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:

        self._load_table(table_name=table, file_idx=file_idx)
        df = self._file_cache[table][file_idx]
        if sequential_index is not None:
            row_id = self._get_row_idx(sequential_index)
        else:
            row_id = np.arange(0, len(df), 1)
        df = df[row_id]
        if len(df) > 0:
            self._raise_column_exception(
                df_columns=df.columns, columns=columns, table=table
            )
            data = df.select(columns)
            if isinstance(data[columns[0]][0], Series):
                x = [data[col][0].to_numpy().reshape(-1, 1) for col in columns]
                array = np.concatenate(x, axis=1)
            else:
                array = data.to_numpy()
        else:
            array = np.array()
        return array

    def _load_table(self, table_name: str, file_idx: int) -> None:
        """Load and possibly cache a parquet table."""
        if file_idx not in self._file_cache[table_name].keys():
            file_path = os.path.join(
                self._path, table_name, f"{table_name}_{file_idx}.parquet"
            )
            df = pol.read_parquet(file_path).sort(self._index_column)
            if (table_name in self._pulsemaps) or (
                table_name == self._node_truth_table
            ):
                if table_name == self._node_truth_table:
                    pol_columns = [pol.col(self._node_truth)]
                else:
                    pol_columns = [pol.col(feat) for feat in self._features]

                if self._string_selection:
                    pol_columns.append(pol.col(self._string_column))

                df = df.group_by(self._index_column).agg(pol_columns)

            self._file_cache[table_name][file_idx] = df.sort(
                self._index_column
            )
            n_files_cached: int = len(self._file_cache[table_name])
            if n_files_cached > self._cache_size:
                del self._file_cache[table_name][
                    list(self._file_cache[table_name].keys())[0]
                ]

    def _raise_column_exception(
        self, df_columns: List[str], columns: Union[List[str], str], table: str
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            if column not in df_columns:
                raise ColumnMissingException(f"{column} not in {table}")

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        if self._node_truth_table is not None:
            assert isinstance(self._node_truth, (list, str))  # mypy..
            node_truth = self.query_table(
                table=self._node_truth_table,
                columns=self._node_truth,
                sequential_index=sequential_index,
            )
        else:
            node_truth = None

        if self._loss_weight_table is not None:
            assert isinstance(self._loss_weight_column, str)
            loss_weight = self.query_table(
                table=self._loss_weight_table,
                columns=self._loss_weight_column,
                sequential_index=sequential_index,
            )
        else:
            loss_weight = None

        features = []
        for pulsemap in self._pulsemaps:
            features.append(
                self.query_table(
                    table=pulsemap,
                    columns=self._features,
                    sequential_index=sequential_index,
                )
            )
        features = np.concatenate(features, axis=0)

        truth = self.query_table(
            table=self._truth_table,
            columns=self._truth,
            sequential_index=sequential_index,
        )

        graph = self._create_graph(
            features=features,
            truth=truth,
            node_truth=node_truth,
            loss_weight=loss_weight,
        )
        return graph
