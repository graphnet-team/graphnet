"""Module defining the base `Dataset` class used in GraphNeT."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import LoggerMixin


class ColumnMissingException(Exception):
    """Exception to indicate a missing column in a dataset."""


class Dataset(ABC, torch.utils.data.Dataset, LoggerMixin):
    """Base Dataset class for reading from any intermediate file format."""

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: str = None,
        loss_weight_column: str = None,
        loss_weight_default_value: Optional[float] = None,
    ):
        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value

        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]

        self._node_truth = node_truth
        self._node_truth_table = node_truth_table

        if string_selection is not None:
            self.logger.warning(
                (
                    "String selection detected.\n",
                    f"Accepted strings: {string_selection}\n",
                    "All other strings are ignored!",
                )
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._string_selection = string_selection

        self._selection = None
        if self._string_selection:
            self._selection = f"string in {str(tuple(self._string_selection))}"

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            self.logger.warning("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            self.logger.warning("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._dtype = dtype

        self._label_fns: Dict[str, Callable[[Data], Any]] = {}

        # Implementation-specific initialisation.
        self._init()

        # Set unique indices
        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection

        # Purely internal member variables
        self._missing_variables = {}
        self._remove_missing_columns()

        # Implementation-specific post-init code.
        self._post_init()

    # Abstract method(s)
    @abstractmethod
    def _init(self):
        """Set internal representation needed to read data from input file."""

    def _post_init(self):
        """Implemenation-specific code to be run after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all available values in `self._index_column`."""

    @abstractmethod
    def _query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        index: int,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any]]:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table (str): Table to be queried.
            columns (List[str]): Columns to read out.
            index (int): Sequentially numbered index (i.e. in [0,len(self))) of
                the event to query. This _may_ differ from the indexation used
                in `self._indices`.
            selection (Optional[str], optional): Selection to be imposed before
                reading out data. Defaults to None.

        Returns:
            List[Tuple[Any]]: Returns a list of tuples containing the values in
                `columns`. If the `table` contains only scalar data for
                `columns`, a list of length 1 is returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """

    # Public method(s)
    def add_label(self, key: str, fn: Callable[[Data], Any]):
        """Add custom graph label define using function `fn`."""
        assert (
            key not in self._label_fns
        ), f"A custom label {key} has already been defined."
        self._label_fns[key] = fn

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int) -> Data:
        if not (0 <= index < len(self)):
            raise IndexError(
                f"Index {index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(index)
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        return graph

    # Internal method(s)
    def _remove_missing_columns(self):
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
        # Find missing features
        missing_features = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features = missing_features.intersection(missing)

        missing_features = list(missing_features)

        # Find missing truth variables
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        # Remove missing features
        if missing_features:
            self.logger.warning(
                "Removing the following (missing) features: "
                + ", ".join(missing_features)
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.logger.warning(
                (
                    "Removing the following (missing) truth variables: "
                    + ", ".join(missing_truth_variables)
                )
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        """Return a list missing columns in `table`."""
        for column in columns:
            try:
                self._query_table(table, [column], 0)
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)

        return self._missing_variables.get(table, [])

    def _query(
        self, index: int
    ) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        """Query file for event features and truth information

        Args:
            index (int): Sequentially numbered index (i.e. in [0,len(self))) of
                the event to query. This _may_ differ from the indexation used
                in `self._indices`.

        Returns:
            List[Tuple]: Pulse-level event features.
            List[Tuple]: Event-level truth information. List has length 1.
            List[Tuple]: Pulse-level truth information.
        """

        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self._query_table(
                pulsemap, self._features, index, self._selection
            )
            features.extend(features_pulsemap)

        truth = self._query_table(self._truth_table, self._truth, index)
        if self._node_truth:
            node_truth = self._query_table(
                self._node_truth_table,
                self._node_truth,
                index,
                self._selection,
            )
        else:
            node_truth = None

        loss_weight = None  # Default
        if self._loss_weight_column is not None:
            if self._loss_weight_table is not None:
                loss_weight = self._query_table(
                    self._loss_weight_table, self._loss_weight_column, index
                )
        return features, truth, node_truth, loss_weight

    def _create_graph(
        self,
        features: List[Tuple[Any]],
        truth: List[Tuple[Any]],
        node_truth: Optional[List[Tuple[Any]]] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e.graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight`
        attributes are not set.

        Args:
            features (list): List of tuples, containing event features.
            truth (list): List of tuples, containing truth information.
            node_truth (list): List of tuples, containing node-level truth.
            loss_weight (float): A weight associated with the event for weighing the loss.

        Returns:
            torch.Data: Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {
            key: truth[0][index] for index, key in enumerate(self._truth)
        }
        assert len(truth) == 1

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            node_truth_dict = {
                key: node_truth_array[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:, 1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)  # pylint: disable=C0103
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features[1:]

        # Add loss weight to graph.
        if loss_weight is not None and self._loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current event
            if len(loss_weight) == 0:
                if self._loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{self._loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                graph[self._loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph[self._loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)

        # Write attributes, either target labels, truth info or original
        # features.
        add_these_to_graph = [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.logger.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )

        # Additionally add original features as (static) attributes
        for index, feature in enumerate(graph.features):
            graph[feature] = graph.x[:, index].detach()

        # Add custom labels to the graph
        for key, fn in self._label_fns.items():
            graph[key] = fn(graph)

        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        abs_pid = abs(truth_dict["pid"])
        sim_type = truth_dict["sim_type"]

        labels_dict = {
            "event_no": truth_dict["event_no"],
            "muon": int(abs_pid == 13),
            "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
            "noise": int((abs_pid == 1) & (sim_type != "data")),
            "neutrino": int(
                (abs_pid != 13) & (abs_pid != 1)
            ),  # @TODO: `abs_pid in [12,14,16]`?
            "v_e": int(abs_pid == 12),
            "v_u": int(abs_pid == 14),
            "v_t": int(abs_pid == 16),
            "track": int(
                (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
            ),
            "dbang": self._get_dbang_label(truth_dict),
            "corsika": int(abs_pid > 20),
        }
        return labels_dict

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1
