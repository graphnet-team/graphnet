from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data

from graphnet.utilities.logging import LoggerMixin


# Global variables
MISSING_VARIABLES = dict()


class ParquetDataset(torch.utils.data.Dataset, LoggerMixin):
    """Pytorch dataset for reading from SQLite."""

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

        self._selection = None
        self._path = path
        self._pulsemaps = pulsemaps
        self._features = features
        self._truth = truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._dtype = dtype
        self._parquet_hook = ak.from_parquet(path)

        self.remove_missing_columns()

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection

    def __len__(self):
        return len(self._indices)

    def remove_missing_columns(self):
        missing_features = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features = missing_features.intersection(missing)
        missing_features = list(missing_features)
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        if missing_features:
            self.logger.warning(
                f"Removing the following (missing) features: {', '.join(missing_features)}"
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

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
        for column in columns:
            try:
                self._parquet_hook[table][column]
            except ValueError:
                if table not in MISSING_VARIABLES:
                    MISSING_VARIABLES[table] = []
                MISSING_VARIABLES[table].append(column)

        return MISSING_VARIABLES.get(table, [])

    def __getitem__(self, i: int) -> Data:
        features, truth, node_truth = self._query_parquet(i)
        graph = self._create_graph(features, truth, node_truth)
        return graph

    def _get_all_indices(self):
        return ak.to_numpy(
            self._parquet_hook[self._truth_table][self._index_column]
        ).tolist()

    def _query_parquet(self, i: int) -> Tuple[np.ndarray]:
        """Query Parquet file for event feature and truth information.

        Args:
            i (int): Sequentially numbered index (i.e. in [0,len(self))) of the
                event to query.

        Returns:
            list: List of tuples, containing event features.
            list: List of tuples, containing truth information.
            list: List of tuples, containing node-level truth information.
        """

        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self._query_table(
                self._features, pulsemap, i, self._selection
            )
            features.extend(features_pulsemap)

        truth = self._query_table(self._truth, self._truth_table, i)
        node_truth = None
        return features, truth, node_truth

    def _query_table(
        self,
        columns: List[str],
        table: str,
        index: int,
        selection: Optional[str] = None,
    ) -> List[Tuple]:
        """Query a table at a specific index, optionally subject to some selection."""
        # Check(s)
        assert (
            selection is None
        ), "Argument `selection` is currently not supported"

        ak_array = self._parquet_hook[table][columns][index]
        dictionary = ak_array.to_list()
        assert list(dictionary.keys()) == columns
        if all(map(np.isscalar, dictionary.values())):
            result = list(dictionary.values())
        else:
            result = list(zip(*dictionary.values()))

        return result

    def _get_dbang_label(self, truth_dict):
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1

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
