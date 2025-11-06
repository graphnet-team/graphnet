"""`Dataset` class(es) for reading data from LMDB databases."""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import pandas as pd
import numpy as np
import lmdb

from torch_geometric.data import Data
from graphnet.data.dataset.dataset import Dataset, ColumnMissingException
from graphnet.data.utilities.lmdb_utilities import (
    get_all_indices,
    get_serialization_method,
)


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
        """
        # Store LMDB-specific parameter before calling super().__init__
        self._pre_computed_representation = pre_computed_representation
        self._deserializer: Optional[Any] = None
        self._env: Optional[lmdb.Environment] = None

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
        self._cached_index: Optional[int] = None
        self._cached_data: Optional[Dict[str, Any]] = None

        # Set custom member variable(s) for raw table mode
        if self._pre_computed_representation is None:
            self._features_string = ", ".join(self._features)
            self._truth_string = ", ".join(self._truth)
            if self._node_truth:
                self._node_truth_string = ", ".join(self._node_truth)

    def _post_init(self) -> None:
        """Implementation-specific code executed after the main constructor."""
        self._missing_variables: Dict[str, List[str]] = {}
        if self._pre_computed_representation is None:
            # Only check for missing columns if using raw tables
            self._remove_missing_columns()
        self._close_connection()

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> np.ndarray:
        """Query table at a specific index, optionally with some selection.

        This method is used when reading raw tables (not pre-computed
        representations).

        Args:
            table: Table name (extractor name) to query.
            columns: Columns to read out.
            sequential_index: Sequentially index of the event to query.
            selection: Selection to be imposed (not fully supported for LMDB).

        Returns:
            Array containing the values in `columns`.

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """
        if self._pre_computed_representation is not None:
            raise RuntimeError(
                "query_table() cannot be used when "
                "pre_computed_representation is set."
            )

        if isinstance(columns, str):
            columns = [columns]

        index = self._get_event_index(sequential_index)

        # Query LMDB database
        assert index is not None
        self._establish_connection()

        try:
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

            # Extract data from raw tables
            if not isinstance(data, dict):
                raise ValueError(
                    f"Unexpected data format at index {index}. "
                    "Expected dict of tables."
                )

            # Get table data
            if table not in data:
                raise ColumnMissingException(
                    f"Table '{table}' not found in database. "
                    f"Available tables: {list(data.keys())}"
                )

            table_data = data[table]

            # Convert to DataFrame if needed
            if isinstance(table_data, dict):
                # Data stored as dict (from to_dict(orient="list"))
                df = pd.DataFrame(table_data)
            elif isinstance(table_data, pd.DataFrame):
                df = table_data
            else:
                raise ValueError(f"Unexpected table data format for '{table}'")

            # Apply string selection if provided
            if selection and self._string_column in df.columns:
                # Parse selection (full SQL parsing not implemented)
                # For now, just check if string_column is in selection
                pass  # TODO: Implement proper selection parsing

            # Extract requested columns
            missing_columns = [c for c in columns if c not in df.columns]
            if missing_columns:
                raise ColumnMissingException(
                    f"Columns {missing_columns} not found in table '{table}'"
                )

            result = df[columns].values
            return result

        except KeyError as e:
            raise ColumnMissingException(str(e)) from e

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

    def _query(
        self, sequential_index: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """Query file for event features and truth information.

        Overrides base class to handle pre-computed representations.
        For pre-computed mode, returns dummy values since we skip this step.

        Args:
            sequential_index: Sequentially index of the event to query.

        Returns:
            Tuple containing pulse-level event features; event-level truth
                information; pulse-level truth information; and event-level
                loss weights, respectively.
        """
        # If using pre-computed representation, skip querying raw tables
        if self._pre_computed_representation is not None:
            # Return dummy values - these won't be used
            return (
                np.array([]).reshape(0, len(self._features)),
                np.array([]),
                None,
                None,
            )

        # Otherwise, use base class implementation
        return super()._query(sequential_index)

    def _create_graph(
        self,
        features: np.ndarray,
        truth: np.ndarray,
        node_truth: Optional[np.ndarray] = None,
        loss_weight: Optional[float] = None,
    ) -> Any:
        """Create Pytorch Data (i.e. graph) object.

        Overrides base class to support pre-computed representations.

        Args:
            features: List of tuples, containing event features (used only
                for raw table mode).
            truth: List of tuples, containing truth information (used only
                for raw table mode).
            node_truth: List of tuples, containing node-level truth (used only
                for raw table mode).
            loss_weight: A weight associated with the event (used only
                for raw table mode).

        Returns:
            Graph object (torch_geometric.Data).
        """
        # If using pre-computed representation, extract it directly
        if self._pre_computed_representation is not None:
            data = self._get_pre_computed_graph()
        else:
            data = super()._create_graph(
                features, truth, node_truth, loss_weight
            )

        return data

    def _get_pre_computed_graph(self) -> Any:
        """Extract pre-computed data representation from LMDB.

        Returns:
            Pre-computed graph object (torch_geometric.Data).
        """
        sequential_index = getattr(self, "_current_sequential_index", None)
        if sequential_index is None:
            raise RuntimeError(
                "_get_pre_computed_graph() called without sequential_index"
            )

        index = self._get_event_index(sequential_index)
        self._establish_connection()

        try:
            assert self._env is not None
            assert self._deserializer is not None
            with self._env.begin(write=False) as txn:
                key_bytes = str(index).encode("utf-8")
                value_bytes = txn.get(key_bytes)
                if value_bytes is None:
                    raise KeyError(f"Index {index} not found in database.")

                # Deserialize data
                data = self._deserializer(value_bytes)

                # Check if data contains pre-computed representations
                if (
                    not isinstance(data, dict)
                    or "data_representations" not in data
                ):
                    raise RuntimeError(
                        f"Database at index {index} does not contain "
                        "pre-computed representations. Set "
                        "pre_computed_representation=None to use raw tables."
                    )

                representations = data["data_representations"]

                # Find the requested representation
                if self._pre_computed_representation in representations:
                    graph = representations[self._pre_computed_representation]
                else:
                    raise KeyError(
                        f"Pre-computed representation "
                        f"'{self._pre_computed_representation}' not found."
                    )

            if not isinstance(graph, Data):
                raise TypeError(
                    f"Pre-computed representation is not a Data object. "
                    f"Got {type(graph)}"
                )

            return graph

        except KeyError as e:
            raise ColumnMissingException(str(e)) from e

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
        self._current_sequential_index = sequential_index

        if self._pre_computed_representation is not None:
            features, truth, node_truth, loss_weight = self._query(
                sequential_index
            )
            return self._create_graph(features, truth, node_truth, loss_weight)
        else:
            # Use base class implementation for raw tables
            return super().__getitem__(sequential_index)
