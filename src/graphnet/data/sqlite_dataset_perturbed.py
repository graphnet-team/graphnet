import pandas as pd
import numpy as np
import sqlite3
import torch
from torch_geometric.data import Data
import time
from graphnet.data.sqlite_dataset import SQLiteDataset


class SQLiteDatasetPerturbed(SQLiteDataset):
    """Pytorch dataset for reading from SQLite including a perturbation step to test the stability of a trained model."""

    def __init__(
        self,
        database,
        pulsemap_table,
        features,
        truth,
        perturbation_dict,
        index_column="event_no",
        truth_table="truth",
        selection=None,
        dtype=torch.float32,
    ):

        assert isinstance(perturbation_dict, dict)
        assert len(set(perturbation_dict.keys())) == len(
            perturbation_dict.keys()
        )
        self._perturbation_dict = perturbation_dict
        super().__init__(
            database,
            pulsemap_table,
            features,
            truth,
            index_column,
            truth_table,
            selection,
            dtype,
        )
        self._perturbation_cols = [
            self._features.index(key) for key in self._perturbation_dict.keys()
        ]

    def __getitem__(self, i):
        self.establish_connection(i)
        features, truth = self._query_database(i)
        perturbed_features = self._perturb_features(features)
        graph = self._create_graph(perturbed_features, truth)
        return graph

    def _perturb_features(self, features):
        features = np.array(features)
        perturbed_features = np.random.normal(
            loc=features[:, self._perturbation_cols],
            scale=np.array(
                list(self._perturbation_dict.values()), dtype=np.float
            ),
        )
        features[:, self._perturbation_cols] = perturbed_features
        return features
