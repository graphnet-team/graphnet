from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset


class SQLiteDatasetPerturbed(SQLiteDataset):
    """Pytorch dataset for reading from SQLite including a perturbation step to test the stability of a trained model."""

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        perturbation_dict: Dict,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
    ):
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
        )

        # Custom member variables
        assert isinstance(perturbation_dict, dict)
        assert len(set(perturbation_dict.keys())) == len(
            perturbation_dict.keys()
        )
        self._perturbation_dict = perturbation_dict

        self._perturbation_cols = [
            self._features.index(key) for key in self._perturbation_dict.keys()
        ]

    def __getitem__(self, index: int) -> Data:
        if not (0 <= index < len(self)):
            raise IndexError(
                f"Index {index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth = self._query(index)
        perturbed_features = self._perturb_features(features)
        graph = self._create_graph(perturbed_features, truth, node_truth)
        return graph

    def _perturb_features(
        self, features: List[Tuple[float]]
    ) -> List[Tuple[float]]:
        features = np.array(features)
        perturbed_features = np.random.normal(
            loc=features[:, self._perturbation_cols],
            scale=np.array(
                list(self._perturbation_dict.values()), dtype=np.float
            ),
        )
        features[:, self._perturbation_cols] = perturbed_features
        return features.tolist()
