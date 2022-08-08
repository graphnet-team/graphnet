from copy import deepcopy
import re
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import sqlite3
import torch
from torch_geometric.data import Data
import time

from graphnet.utilities.logging import get_logger


logger = get_logger()


# Global variables
MISSING_VARIABLES = dict()


class SQLiteDataset(torch.utils.data.Dataset):
    """Pytorch dataset for reading from SQLite."""

    def __init__(
        self,
        database: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        loss_weight_table: str = None,
        loss_weight_column: str = None,
        loss_weight_padding_value: float = 1.0,
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
    ):

        # Check(s)
        if isinstance(database, list):
            self._database_list = database
            self._all_connections_established = False
            self._all_connections = []
        else:
            self._database_list = None
            assert isinstance(database, str)
            assert database.endswith(".db")

        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        self._node_truth = None
        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]
            self._node_truth = node_truth
            self._node_truth_table = node_truth_table
            self._node_truth_string = ", ".join(self._node_truth)

        if string_selection is not None:
            logger.info(
                "WARNING - STRING SELECTION DETECTED. \n Accepted strings: %s \n all other strings are ignored!"
                % string_selection
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            print("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            print("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._string_selection = string_selection
        self._selection = ""
        if self._string_selection:
            self._selection = f"string in {str(tuple(self._string_selection))}"

        self._database = database
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._dtype = dtype

        self._loss_weight_padding_value = loss_weight_padding_value
        self._features_string = ", ".join(self._features)
        self._truth_string = ", ".join(self._truth)
        if self._database_list is not None:
            self._current_database = None
        self._conn = None  # Handle for sqlite3.connection

        self.remove_missing_columns()

        self._features_string = ", ".join(self._features)
        self._truth_string = ", ".join(self._truth)

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection
        self.close_connection()

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
            logger.warning(
                f"Removing the following (missing) features: {', '.join(missing_features)}"
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        if missing_truth_variables:
            logger.warning(
                f"Removing the following (missing) truth variables: {', '.join(missing_truth_variables)}"
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        self.establish_connection(0)
        try:
            _ = self._conn.execute(
                f"SELECT {','.join(columns)} FROM {table}"
            ).fetchall()
        except sqlite3.OperationalError as e:
            global MISSING_VARIABLES
            missing_variable = re.sub(".*: *", "", str(e))
            if table not in MISSING_VARIABLES:
                MISSING_VARIABLES[table] = []
            if missing_variable not in MISSING_VARIABLES[table]:
                logger.debug(str(e) + f" in table: {table}")
                MISSING_VARIABLES[table].append(missing_variable)
            columns = deepcopy(columns)
            columns.remove(missing_variable)
            return self._check_missing_columns(columns, table)

        self.close_connection()
        return MISSING_VARIABLES.get(table, [])

    def _query_table(
        self,
        columns: Union[List, str],
        table: str,
        index: int,
        selection: Optional[str] = None,
    ):
        """Query a table at a specific index, optionally subject to some selection."""
        # Check(s)
        if isinstance(columns, list):
            columns = ", ".join(columns)

        if not selection:  # I.e., `None` or `""`
            selection = "1=1"  # Identically true, to select all

        if self._database_list is None:
            index = self._indices[index]
        else:
            index = self._indices[index][0]

        # Query table
        result = self._conn.execute(
            f"SELECT {columns} FROM {table} WHERE {self._index_column} = {index} and {selection}"
        ).fetchall()
        return result

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth, node_truth, loss_weight = self._query_database(i)
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        return graph

    def _get_all_indices(self):
        self.establish_connection(0)
        indices = pd.read_sql_query(
            f"SELECT {self._index_column} FROM {self._truth_table}", self._conn
        )
        return indices.values.ravel().tolist()

    def _query_database(self, i):
        """Query SQLite database for event feature and truth information.

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
                self._features_string, pulsemap, i, self._selection
            )
            features.extend(features_pulsemap)
        truth = self._query_table(self._truth_string, self._truth_table, i)
        if self._node_truth:
            node_truth = self._query_table(
                self._node_truth_string,
                self._node_truth_table,
                i,
                self._selection,
            )
        else:
            node_truth = None

        if self._loss_weight_column is not None:
            if self._loss_weight_table is not None:
                loss_weight = self._query_table(
                    self._loss_weight_column, self._loss_weight_table, i
                )
        else:
            loss_weight = [self._loss_weight_padding_value]

        return features, truth, node_truth, loss_weight

    def _get_dbang_label(self, truth_dict):
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1

    def _create_graph(
        self,
        features,
        truth,
        node_truth=None,
        loss_weight=None,
    ):
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
        truth_dict = {key: truth[0][ix] for ix, key in enumerate(self._truth)}
        assert len(truth) == 1

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            node_truth_dict = {
                key: node_truth_array[:, ix]
                for ix, key in enumerate(self._node_truth)
            }

        # Unpack common variables
        abs_pid = abs(truth_dict["pid"])
        sim_type = truth_dict["sim_type"]

        labels_dict = {
            "event_no": truth_dict["event_no"],
            "muon": int(abs_pid == 13),
            "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
            "noise": int((abs_pid == 1) & (sim_type != "data")),
            "neutrino": int(
                (abs_pid != 13) & (abs_pid != 1)
            ),  # `abs_pid in [12,14,16]`?
            "v_e": int(abs_pid == 12),
            "v_u": int(abs_pid == 14),
            "v_t": int(abs_pid == 16),
            "track": int(
                (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
            ),
            "dbang": self._get_dbang_label(truth_dict),
            "corsika": int(abs_pid > 20),
        }

        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:, 1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features[1:]
        if loss_weight is not None and self._loss_weight_column is not None:
            if len(loss_weight) == 0:
                graph[self._loss_weight_column] = torch.tensor(
                    self._loss_weight_padding_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph[self._loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)

        # Write attributes, either target labels, truth info or original features.
        add_these_to_graph = [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type, e.g. `str`.
                    pass

        for ix, feature in enumerate(graph.features):
            graph[feature] = graph.x[:, ix].detach()
        return graph

    def establish_connection(self, i):
        """Make sure that a sqlite3 connection is open."""
        if self._database_list is None:
            if self._conn is None:
                self._conn = sqlite3.connect(self._database)
        else:
            if self._conn is None:
                if self._all_connections_established is False:
                    self._all_connections = []
                    for database in self._database_list:
                        con = sqlite3.connect(database)
                        self._all_connections.append(con)
                    self._all_connections_established = True
                self._conn = self._all_connections[self._indices[i][1]]
            if self._indices[i][1] != self._current_database:
                self._conn = self._all_connections[self._indices[i][1]]
                self._current_database = self._indices[i][1]
        return self

    def close_connection(self):
        """Make sure that no sqlite3 connection is open.

        This is necessary to calls this before passing to `torch.DataLoader`
        such that the dataset replica on each worker is required to create its
        own connection (thereby avoiding `sqlite3.DatabaseError: database disk
        image is malformed` errors due to inability to use sqlite3 connection
        accross processes.
        """
        if self._conn is not None:
            self._conn.close()
            del self._conn
            self._conn = None
        if self._database_list is not None:
            if self._all_connections_established:
                for con in self._all_connections:
                    con.close()
                del self._all_connections
                self._all_connections_established = False
                self._conn = None
        return self
