from typing import List, Optional, Union
import pandas as pd
import numpy as np
import sqlite3
import torch
from torch_geometric.data import Data
from graphnet.data.sqlite_dataset import SQLiteDataset

class SQLiteDatasetCleaning(SQLiteDataset):
    """Pytorch dataset for reading from SQLite and adding truth flags on pulse level
    """
    def __init__(
        self,
        database: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        index_column: str = 'event_no',
        truth_table: str = 'truth',
        truth_flag_table: str = 'SplitInIcePulses_TruthFlags',
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
            assert database.endswith('.db')

        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        self._database = database
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._truth_flag_table = truth_flag_table
        self._dtype = dtype

        self._features_string = ', '.join(self._features)
        self._truth_string = ', '.join(self._truth)
        if (self._database_list != None):
            self._current_database = None
        self._conn = None  # Handle for sqlite3.connection

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection
        self.close_connection()

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth = self._query_database(i)
        graph = self._create_graph(features, truth)
        graph = self._add_truth_flag(i, graph)
        return graph

    def _query_noise_database(self, i):
        """Query SQLite database for noise truth information.
        """
        if self._database_list == None:
            index = self._indices[i]
        else:
            index = self._indices[i][0]

        truth_flags = self._conn.execute(
            "SELECT truth_flag FROM {} WHERE {} = {}".format(
                self._truth_flag_table,
                self._index_column,
                index,
            )
        ).fetchall()

        return truth_flags

    def _add_truth_flag(self, i, graph):
        graph['truth_flag'] = torch.tensor(self._query_noise_database(i)).reshape(-1)
        return graph