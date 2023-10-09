"""Class(es) for building/connecting graphs."""

from typing import List, Tuple
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.graphs.utils import (
    cluster_summarize_with_percentiles,
    identify_indices,
)
from copy import deepcopy


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    def __init__(self) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @final
    def forward(
        self, x: torch.tensor, node_feature_names: List[str]
    ) -> Tuple[Data, List[str]]:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            node_feature_names: list of names for each column in ´x´.

        Returns:
            graph: a graph without edges
            new_features_name: List of new feature names.
        """
        graph, new_feature_names = self._construct_nodes(
            x=x, feature_names=node_feature_names
        )
        return graph, new_feature_names

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return self.nb_inputs

    @final
    def set_number_of_inputs(self, node_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            node_feature_names: name of each node feature column.
        """
        assert isinstance(node_feature_names, list)
        self.nb_inputs = len(node_feature_names)

    @abstractmethod
    def _construct_nodes(
        self, x: torch.tensor, feature_names: List[str]
    ) -> Data:
        """Construct nodes from raw node features ´x´.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            feature_names: List of names for reach column in `x`. Identical
            order of appearance. Length `d`.

        Returns:
            graph: graph without edges.
            new_node_features: A list of node features names.
        """


class NodesAsPulses(NodeDefinition):
    """Represent each measured pulse of Cherenkov Radiation as a node."""

    def _construct_nodes(
        self, x: torch.Tensor, feature_names: List[str]
    ) -> Data:
        return Data(x=x), feature_names


class PercentileClusters(NodeDefinition):
    """Represent nodes as clusters with percentile summary node features.

    If `cluster_on` is set to the xyz coordinates of DOMs
    e.g. `cluster_on = ['dom_x', 'dom_y', 'dom_z']`, each node will be a
    unique DOM and the pulse information (charge, time) is summarized using
    percentiles.
    """

    def __init__(
        self,
        cluster_on: List[str],
        feature_names: List[str],
        percentiles: List[int],
        add_counts: bool = True,
    ) -> None:
        """Construct `PercentileClusters`.

        Args:
            cluster_on: Names of features to create clusters from.
            feature_names: List of colum names for the input data.
                           E.g. ['dom_x', 'dom_y', 'dom_z',..]
            percentiles: List of percentiles. E.g. `[10, 50, 90]`.
            add_counts: If True, number of duplicates is added to output array.
        """
        self._cluster_on = cluster_on
        self._percentiles = percentiles
        self._add_counts = add_counts
        (
            cluster_idx,
            summ_idx,
            new_feature_names,
        ) = self._get_indices_and_feature_names(
            feature_names, self._add_counts
        )
        self._cluster_indices = cluster_idx
        self._summarization_indices = summ_idx
        self._output_feature_names = new_feature_names
        # Base class constructor
        super().__init__()

    def _get_indices_and_feature_names(
        self,
        feature_names: List[str],
        add_counts: bool,
    ) -> Tuple[List[int], List[int], List[str]]:
        cluster_idx, summ_idx, summ_names = identify_indices(
            feature_names, self._cluster_on
        )
        new_feature_names = deepcopy(self._cluster_on)
        for feature in summ_names:
            for pct in self._percentiles:
                new_feature_names.append(f"{feature}_pct{pct}")
        if add_counts:
            # add "counts" as the last feature
            new_feature_names.append("counts")
        return cluster_idx, summ_idx, new_feature_names

    def _construct_nodes(
        self, x: torch.Tensor, feature_names: List[str]
    ) -> Data:
        # Cast to Numpy
        x = x.numpy()
        # Construct clusters with percentile-summarized features
        array = cluster_summarize_with_percentiles(
            x=x,
            summarization_indices=self._summarization_indices,
            cluster_indices=self._cluster_indices,
            percentiles=self._percentiles,
            add_counts=self._add_counts,
        )

        return Data(x=torch.tensor(array)), self._output_feature_names

    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return len(self._output_feature_names)
