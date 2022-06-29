"""Module defining the base `Dataset` class used in GraphNeT."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.utilities.logging import LoggerMixin


class Dataset(ABC, torch.utils.data.Dataset, LoggerMixin):
    """Base Dataset class for reading from any intermediate file format."""

    def __init__(
        self,
        path: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # Check(s)
        if isinstance(path, list):
            self.logger.error("Multiple folders not supported")
        assert isinstance(path, str)

        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        assert (
            node_truth is None
        ), "Argument `node_truth` is currently not supported."
        assert (
            node_truth_table is None
        ), "Argument `node_truth_table` is currently not supported."

        assert (
            string_selection is None
        ), "Argument `string_selection` is currently not supported"

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = features
        self._truth = truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._dtype = dtype

        # Implementation specific initialisation.
        self._initialise()

        # Purely internal member variables
        self._missing_variables = dict()
        self._remove_missing_columns()

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection

    # Abstract method(s)
    @abstractmethod
    def _set_input_file(self, path: str):
        """Set any internal representation needed to read from `path`."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all available values in `self._index_column`."""

    @abstractmethod
    def _query_table(
        self,
        table: str,
        columns: List[str],
        index: int,
        selection: Optional[str] = None,
    ) -> Union[List[Tuple[Any]], Tuple[Any]]:
        """Query a table at a specific index, optionally subject to some selection.

        Args:
            table (str): Table to be queried.
            columns (List[str]): Columns to read out.
            index (int): Sequentially numbered index (i.e. in [0,len(self))) of
                the event to query. This _may_ differ from the indexation used
                in `self._indices`.
            selection (Optional[str], optional): Selection to be imposed before
                reading out data. Defaults to None.

        Returns:
            Union[List[Tuple[Any]], Tuple[Any]]:  Returns a list of tuples if
                the `table` contains array-type data; otherwise returns a single
                tuple of `table` contains only scalar data.
        """

    # Public method(s)
    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i: int) -> Data:
        features, truth, node_truth = self._query(i)
        graph = self._create_graph(features, truth, node_truth)
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
                f"Removing the following (missing) features: {', '.join(missing_features)}"
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.logger.warning(
                f"Removing the following (missing) truth variables: {', '.join(missing_truth_variables)}"
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
            except ValueError:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)

        return self._missing_variables.get(table, [])

    def _query(self, index: int) -> Tuple[List[Tuple], Tuple, List[Tuple]]:
        """Query file for event features and truth information

        Args:
            index (int): Sequentially numbered index (i.e. in [0,len(self))) of
                the event to query. This _may_ differ from the indexation used
                in `self._indices`.

        Returns:
            List[Tuple]: Pulse-level event features.
            Tuple: Event-level truth information.
            List[Tuple]: Pulse-level truth information.
        """

        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self._query_table(
                pulsemap, self._features, index, self._selection
            )
            features.extend(features_pulsemap)

        truth = self._query_table(self._truth_table, self._truth, index)
        node_truth = None
        return features, truth, node_truth

    def _create_graph(self, features, truth, node_truth=None):
        """Create Pytorch Data (i.e.graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight` attributes
        are not set.

        Args:
            features (list): List of tuples, containing event features.
            truth (list): List of tuples, containing truth information.
            node_truth (list): List of tuples, containing node-level truth.

        Returns:
            torch.Data: Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {key: truth[ix] for ix, key in enumerate(self._truth)}

        # assert len(truth) == len(self._truth)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            node_truth_dict = {
                key: node_truth_array[:, ix]
                for ix, key in enumerate(self._node_truth)
            }

        # Construct graph data object
        x = torch.tensor(features, dtype=self._dtype)
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features

        # Write attributes, either target labels, truth info or original features.
        add_these_to_graph = [truth_dict]  # [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type, e.g. `str`.
                    self.logger.debug(
                        f"Could not assign `{key}` with type '{type(value).__name__}' as attribute to graph."
                    )

        return graph

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1
