"""`Dataset` class(es) for reading perturbed data from SQLite databases."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset


class SQLiteDatasetPerturbed(SQLiteDataset):
    """Pytorch dataset for reading perturbed data from SQLite databases.

    This including a pre-processing step, where the input data is randomly
    perturbed according to given per-feature "noise" levels. This is intended
    to test the stability of a trained model under small changes to the input
    parameters.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        perturbation_dict: Dict[str, float],
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
    ):
        """Construct SQLiteDatasetPerturbed.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            perturbation_dict (Dict[str, float]): Dictionary mapping a feature
                name to a standard deviation according to which the values for
                this feature should be randomly perturbed.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: List of indicies (in `index_column`) of the events in
                the input files that should be read. Defaults to None, meaning
                that all events in the input files are read.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
        """
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
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
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

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        perturbed_features = self._perturb_features(features)
        graph = self._create_graph(
            perturbed_features, truth, node_truth, loss_weight
        )
        return graph

    def _perturb_features(
        self, features: List[Tuple[float, ...]]
    ) -> List[Tuple[float, ...]]:
        features_array = np.array(features)
        perturbed_features = np.random.normal(
            loc=features_array[:, self._perturbation_cols],
            scale=np.array(
                list(self._perturbation_dict.values()), dtype=np.float
            ),
        )
        features_array[:, self._perturbation_cols] = perturbed_features
        return features_array.tolist()
