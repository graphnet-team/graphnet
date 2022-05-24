from typing import List, Optional, Union
import pandas as pd
import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data
import time


class ParquetDataset(torch.utils.data.Dataset):
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
            print("multiple folders not supported")
            assert isinstance(1, str)
        assert isinstance(path, str)

        if isinstance(pulsemaps, str):
            pulsemaps = pulsemaps

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        self._node_truth = None
        if node_truth is not None:
            print("node truth is not supported")
            assert isinstance(1, str)

        if string_selection is not None:
            print("string selection not supported")
            assert isinstance(1, str)

        self._string_selection = string_selection
        self._selection = ""
        if self._string_selection:
            self._selection = f"string in {str(tuple(self._string_selection))}"

        self._path = path
        self._pulsemaps = pulsemaps
        self._features = features
        self._truth = truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._dtype = dtype
        self._parquet_hook = ak.from_parquet(path)

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection

    def __len__(self):
        return len(self._indices)

    def _query_parquet(
        self,
        columns: Union[List, str],
        table: str,
        index: int,
        selection: Optional[str] = None,
    ):
        return ak.to_pandas(self._parquet_hook[table][[index], columns])

    def __getitem__(self, i):
        features, truth, node_truth = self._get_event_data(i)
        graph = self._create_graph(features, truth, node_truth)
        return graph

    def _get_all_indices(self):
        return ak.to_numpy(self._parquet_hook[self._index_column]).tolist()

    def _get_event_data(self, i):
        """Query SQLite database for event feature and truth information.

        Args:
            i (int): Sequentially numbered index (i.e. in [0,len(self))) of the
                event to query.

        Returns:
            list: List of tuples, containing event features.
            list: List of tuples, containing truth information.
            list: List of tuples, containing node-level truth information.
        """
        features = self._query_parquet(
            self._features, self._pulsemaps, i, self._selection
        )
        truth = self._query_parquet(self._truth, self._truth_table, i)
        node_truth = None
        return features, truth, node_truth

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
        truth_dict = {
            key: truth[key].to_numpy().tolist() for key in truth.keys()
        }

        # assert len(truth) == len(self._truth)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            node_truth_dict = {
                key: node_truth_array[:, ix]
                for ix, key in enumerate(self._node_truth)
            }

        # Construct graph data object
        x = torch.tensor(features.to_numpy(), dtype=self._dtype)
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
                    if key not in ["x", "y", "z"]:
                        graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type, e.g. `str`.
                    pass
        return graph
