"""`Dataset` class(es) for reading data from LMDB databases."""

from typing import Any, Dict, List, Optional, Union
import os
import numpy as np
import lmdb
from tqdm import tqdm
from torch_geometric.data import Data
from graphnet.data.dataset.dataset import Dataset, ColumnMissingException
from graphnet.data.utilities.lmdb_utilities import (
    get_all_indices,
    get_serialization_method,
)
from graphnet.training.utils import add_custom_labels, add_truth


class LMDBDataset(Dataset):
    """Pytorch dataset for reading data from LMDB databases.

    Supports two modes:
    1. Reading raw tables and computing data representations in real-time
       (similar to SQLiteDataset)
    2. Reading pre-computed data representations directly from the database
       (skipping DataRepresentation computation)
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        graph_definition: Optional[Any] = None,
        data_representation: Optional[Any] = None,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: Any = None,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
        labels: Optional[Dict[str, Any]] = None,
        # LMDB-specific parameters
        pre_computed_representation: Optional[str] = None,
        repeat_labels_by: Optional[int] = None,
    ):
        """Construct `LMDBDataset`.

        Args:
            path: Path to the LMDB database directory(ies).
            pulsemaps: Name(s) of the pulse map series (used when reading raw
                tables, ignored when using pre-computed representations).
            features: List of columns in the input files (used when reading raw
                tables, ignored when using pre-computed representations).
            truth: List of event-level columns (used when reading raw tables,
                ignored when using pre-computed representations).
            graph_definition: Method that defines the graph representation.
                NOTE: DEPRECATED Use `data_representation` instead.
            data_representation: Method that defines the data representation.
            node_truth: List of node-level columns in the input files that
                should be added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indices to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects.
            selection: The events that should be read. This can be given either
                as list of indices (in `index_column`); or a string-based
                selection used to query the `Dataset` for events passing the
                selection.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights.
            loss_weight_default_value: Default per-event loss weight.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection.
            labels: Dictionary of labels to be added to the dataset.
            pre_computed_representation: Name of the pre-computed data
                representation to use. If None, reads raw tables and computes
                representations in real-time. If specified, extracts the
                pre-computed representation directly (by class name or key).
            repeat_labels_by: If specified, repeats the labels along the
                specified dimension.
        """
        # Store LMDB-specific parameter before calling super().__init__
        self._pre_computed_representation = pre_computed_representation
        self._deserializer: Optional[Any] = None
        self._env: Optional[lmdb.Environment] = None
        self._repeat_labels_by = repeat_labels_by
        self._tables: Optional[List[str]] = None
        # Call parent constructor
        super().__init__(
            path=path,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            graph_definition=graph_definition,
            data_representation=data_representation,
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
            labels=labels,
        )

    # Implementing abstract method(s)
    def _init(self) -> None:
        """Set internal representation needed to read data from LMDB."""
        # Check path format
        if isinstance(self._path, list):
            if len(self._path) > 1:
                raise NotImplementedError(
                    "Multiple LMDB databases not yet supported. "
                    "Please provide a single path."
                )
            self._path = self._path[0]

        assert isinstance(self._path, str)
        # LMDB databases are directories, check if path exists and directory
        if not os.path.isdir(self._path):
            raise ValueError(
                f"LMDB path `{self._path}` is not a valid directory. "
                "LMDB databases are stored as directories."
            )

        # Get deserialization method
        self._deserializer = get_serialization_method(self._path)
        if self._deserializer is None:
            raise ValueError(
                f"Could not determine deserialization method for {self._path}."
                "Database may be corrupted or use unsupported serialization."
            )

        # Initialize cache for deserialized data (single index at a time)
        self._reset_cache()

        # Set custom member variable(s) for raw table mode
        if self._pre_computed_representation is None:
            self._features_string = ", ".join(self._features)
            self._truth_string = ", ".join(self._truth)
            if self._node_truth:
                self._node_truth_string = ", ".join(self._node_truth)

    def _reset_cache(self) -> None:
        """Reset the cache."""
        self._cached_index: int = -1
        self._cached_data: Dict[str, Any] = {}

    def _post_init(self) -> None:
        """Implementation-specific code executed after the main constructor."""
        self._missing_variables: Dict[str, List[str]] = {}
        if self._pre_computed_representation is None:
            # Only check for missing columns if using raw tables
            self._remove_missing_columns()
        self._close_connection()
        if self._pre_computed_representation is not None:
            self._identify_missing_truth_labels()

    def _identify_missing_truth_labels(self) -> None:
        """Identify missing truth labels in the pre-computed representation."""
        data = self._get_pre_computed_data_representation(0)
        if self._truth_table in self._cached_data.keys():
            labels = [
                label for label in self._truth if label not in data.keys()
            ]
            self._missing_truth_labels = labels
            self.info(
                f"The following truth labels will be added to the "
                f"pre-computed representation: {self._missing_truth_labels}"
            )
        else:
            self._missing_truth_labels = []

    def _update_cache(self, sequential_index: int) -> None:
        """Update the cache with the data for the given sequential index.

        Args:
            sequential_index: Sequentially index of the event to query.
        """
        index = self._get_event_index(sequential_index)

        # Query LMDB database
        assert index is not None
        self._establish_connection()

        # Check cache first
        if self._cached_index == index and self._cached_data is not None:
            data = self._cached_data
        else:
            # Cache miss - deserialize and update cache
            assert self._env is not None
            assert self._deserializer is not None
            with self._env.begin(write=False) as txn:
                key_bytes = str(index).encode("utf-8")
                value_bytes = txn.get(key_bytes)
                if value_bytes is None:
                    raise KeyError(f"Index {index} not found in database.")

                # Deserialize data
                data = self._deserializer(value_bytes)
                # Update cache
                self._cached_index = index
                self._cached_data = data
        return

    def _get_tables(self) -> List[str]:
        """Return a list of all tables in the database."""
        if self._tables is not None:
            return self._tables
        else:
            if len(self._cached_data) == 0:
                self._update_cache(0)
            tables = list(self._cached_data.keys())
            self._reset_cache()
            self._tables = tables
            return tables

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        """Query table at a specific index, optionally with some selection.

        Args:
            table: Table name (extractor name) to query.
            columns: Columns to read out.
            sequential_index: Sequentially index of the event to query.
            selection: Selection to be imposed (not fully supported for LMDB).

        Returns:
            Numpy array containing the values in `columns`.
        """
        # Convert columns to list if string
        if isinstance(columns, str):
            columns = [columns]

        # Check if we're in the string-resolver mode
        if (sequential_index is None) and (selection is None):
            if not hasattr(self, "_indices"):
                self._indices = self._get_all_indices()

        # Check if the table is in the entry
        tables = self._get_tables()
        if table not in tables:
            raise ColumnMissingException(
                f"Table '{table}' not found in database ({tables})."
            )

        # If a sequential index is provided, load single entry into the cache
        if sequential_index is not None:
            self._update_cache(sequential_index)

            table_data = self._query_cache(table=table, columns=columns)
        else:
            # If no sequential index is provided, return all entries
            table_data = []
            self.info(
                f"Querying table '{table}' for all entries."
                " This may take a while..."
            )
            for sequential_index in tqdm(range(len(self))):
                self._update_cache(sequential_index)
                single_entry = self._query_cache(table=table, columns=columns)
                table_data.append(single_entry)
            table_data = np.concatenate(table_data, axis=0)
        return table_data

    def _query_cache(
        self, table: str, columns: Union[List[str], str]
    ) -> np.ndarray:
        """Query the cache for the table data."""
        data = self._cached_data[table]
        try:
            table_data = [
                np.array(data[column]).reshape(-1, 1) for column in columns
            ]
        except KeyError:
            missing = []
            for column in columns:
                if column not in data.keys():
                    missing.append(column)

            raise ColumnMissingException(
                f"Columns '{missing}' not found in table '{table}'."
            )

        table_data = [
            np.array(data[column]).reshape(-1, 1) for column in columns
        ]
        table_data = np.concatenate(table_data, axis=1)
        return table_data

    def _get_all_indices(self) -> List[int]:
        """Return a list of all unique values in `self._index_column`."""
        # _init() ensures self._path is a string
        assert isinstance(self._path, str)
        return get_all_indices(self._path)

    def _get_event_index(self, sequential_index: Optional[int]) -> int:
        """Return the event index corresponding to a `sequential_index`."""
        index: int = 0
        if sequential_index is not None:
            index_ = self._indices[sequential_index]
            if isinstance(index_, list):
                index_ = index_[0]
            if not isinstance(index_, int):
                index_ = int(index_)
            assert isinstance(index_, int)
            index = index_
        return index

    def _establish_connection(self) -> "LMDBDataset":
        """Make sure that an LMDB connection is open."""
        if self._env is None:
            self._env = lmdb.open(
                self._path, readonly=True, lock=False, subdir=True
            )
        return self

    def _close_connection(self) -> "LMDBDataset":
        """Make sure that no LMDB connection is open.

        This is necessary to call this before passing to `torch.DataLoader`
        such that the dataset replica on each worker is required to create
        its own connection (thereby avoiding connection sharing issues
        across processes).
        """
        if self._env is not None:
            self._env.close()
            del self._env
            self._env = None
        return self

    def _get_pre_computed_data_representation(
        self, sequential_index: int
    ) -> Data:
        """Extract pre-computed data representation from LMDB.

        Returns:
            Pre-computed graph object (torch_geometric.Data).
        """
        self._update_cache(sequential_index)
        data = self._cached_data

        if "data_representations" not in data.keys():
            raise RuntimeError(
                "Database entry does not contain pre-computed "
                "representations. Set pre_computed_representation=None "
                "to use raw tables."
            )

        representations = data["data_representations"]
        if not isinstance(representations, dict):
            raise RuntimeError(
                "Pre-computed representations are malformed. "
                "Expected dictionary of representations."
            )

        if self._pre_computed_representation not in representations:
            raise KeyError(
                f"Pre-computed representation "
                f"'{self._pre_computed_representation}' not found."
            )

        return representations[self._pre_computed_representation]

    def __getitem__(self, sequential_index: int) -> Any:
        """Return graph `Data` object at `index`.

        Overrides base class to support pre-computed representations.

        Args:
            sequential_index: Sequential index of the event.

        Returns:
            Graph object.
        """
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )

        # Store sequential index for pre-computed mode

        if self._pre_computed_representation is not None:
            data = self._get_pre_computed_data_representation(sequential_index)

            # If the user specifies missing truth labels, add them to the data
            if self._missing_truth_labels:
                truth_table = self._cached_data[self._truth_table]

                add_these = []
                for label in self._missing_truth_labels:
                    if label in truth_table.keys():
                        add_these.append(label)

                truth_dict = [
                    {label: truth_table[label]} for label in add_these
                ]
                data = add_truth(
                    data=data, truth_dicts=truth_dict, dtype=self._dtype
                )
            data = add_custom_labels(
                data=data,
                custom_label_functions=self._label_fns,
                repeat_labels_by=self._repeat_labels_by,
            )
        else:
            # Use base class implementation for raw tables
            data = super().__getitem__(sequential_index)

        return data
