"""Class(es) for building/connecting graphs."""

from typing import List, Tuple, Optional, Dict
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.graphs.utils import (
    cluster_summarize_with_percentiles,
    identify_indices,
    lex_sort,
    ice_transparency,
)
from copy import deepcopy

import numpy as np


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    def __init__(
        self, input_feature_names: Optional[List[str]] = None
    ) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        if input_feature_names is not None:
            self.set_output_feature_names(
                input_feature_names=input_feature_names
            )

    @final
    def forward(self, x: torch.tensor) -> Tuple[Data, List[str]]:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            node_feature_names: list of names for each column in ´x´.

        Returns:
            graph: a graph without edges
            new_features_name: List of new feature names.
        """
        graph = self._construct_nodes(x=x)
        try:
            self._output_feature_names
        except AttributeError as e:
            self.error(
                f"""{self.__class__.__name__} was instantiated without
                       `input_feature_names` and it was not set prior to this
                       forward call. If you are using this class outside a
                       `GraphDefinition`, please instatiate
                       with `input_feature_names`."""
            )  # noqa
            raise e
        return graph, self._output_feature_names

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return len(self._output_feature_names)

    @final
    def set_number_of_inputs(self, input_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            input_feature_names: name of each input feature column.
        """
        assert isinstance(input_feature_names, list)
        self.nb_inputs = len(input_feature_names)

    @final
    def set_output_feature_names(self, input_feature_names: List[str]) -> None:
        """Set output features names as a member variable.

        Args:
            input_feature_names: List of column names of the input to the
            node definition.
        """
        self._output_feature_names = self._define_output_feature_names(
            input_feature_names
        )

    @abstractmethod
    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Construct names of output columns.

        Args:
            input_feature_names: List of column names for the input data.

        Returns:
            A list of column names for each column in
            the node definition output.
        """

    @abstractmethod
    def _construct_nodes(self, x: torch.tensor) -> Tuple[Data, List[str]]:
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

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
        return Data(x=x)


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
        percentiles: List[int],
        add_counts: bool = True,
        input_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `PercentileClusters`.

        Args:
            cluster_on: Names of features to create clusters from.
            percentiles: List of percentiles. E.g. `[10, 50, 90]`.
            add_counts: If True, number of duplicates is added to output array.
            input_feature_names: (Optional) column names for input features.
        """
        self._cluster_on = cluster_on
        self._percentiles = percentiles
        self._add_counts = add_counts
        # Base class constructor
        super().__init__(input_feature_names=input_feature_names)

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        (
            cluster_idx,
            summ_idx,
            new_feature_names,
        ) = self._get_indices_and_feature_names(
            input_feature_names, self._add_counts
        )
        self._cluster_indices = cluster_idx
        self._summarization_indices = summ_idx
        return new_feature_names

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

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        # Cast to Numpy
        x = x.numpy()
        # Construct clusters with percentile-summarized features
        if hasattr(self, "_summarization_indices"):
            array = cluster_summarize_with_percentiles(
                x=x,
                summarization_indices=self._summarization_indices,
                cluster_indices=self._cluster_indices,
                percentiles=self._percentiles,
                add_counts=self._add_counts,
            )
        else:
            self.error(
                f"""{self.__class__.__name__} was not instatiated with
                `input_feature_names` and has not been set later.
                Please instantiate this class with `input_feature_names`
                if you're using it outside `GraphDefinition`."""
            )  # noqa
            raise AttributeError

        return Data(x=torch.tensor(array))


class NodeAsDOMTimeSeries(NodeDefinition):
    """Represent each node as a DOM with time and charge time series data."""

    def __init__(
        self,
        keys: List[str] = [
            "dom_x",
            "dom_y",
            "dom_z",
            "dom_time",
            "charge",
        ],
        id_columns: List[str] = ["dom_x", "dom_y", "dom_z"],
        time_column: str = "dom_time",
        charge_column: str = "charge",
        max_activations: Optional[int] = None,
    ) -> None:
        """Construct `NodeAsDOMTimeSeries`.

        Args:
            keys: Names of features in the data (in order).
            id_columns: List of columns that uniquely identify a DOM.
            time_column: Name of time column.
            charge_column: Name of charge column.
            max_activations: Maximum number of activations to include in the time series.
        """
        self._keys = keys
        super().__init__(input_feature_names=self._keys)
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_column)
        try:
            self._charge_index: Optional[int] = self._keys.index(charge_column)
        except ValueError:
            self.warning(
                "Charge column with name {} not found. Running without.".format(
                    charge_column
                )
            )

            self._charge_index = None

        self._max_activations = max_activations

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names + ["new_node_col"]

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw node features ´x´."""
        # Cast to Numpy
        x = x.numpy()
        if x.shape[0] == 0:
            return Data(x=torch.tensor(np.column_stack([x, []])))
        # if there is no charge column add a dummy column of zeros with the same shape as the time column
        if self._charge_index is None:
            charge_index: int = len(self._keys)
            x = np.insert(x, charge_index, np.zeros(x.shape[0]), axis=1)
        else:
            charge_index = self._charge_index

        # Sort by time
        x = x[x[:, self._time_index].argsort()]
        # Undo log10 scaling so we can sum charges
        x[:, charge_index] = np.power(10, x[:, charge_index])
        # Shift time to start at 0
        x[:, self._time_index] -= np.min(x[:, self._time_index])
        # Group pulses on the same DOM
        x = lex_sort(x, self._id_columns)

        unique_sensors, counts = np.unique(
            x[:, self._id_columns], axis=0, return_counts=True
        )

        sort_this = np.concatenate(
            [unique_sensors, counts.reshape(-1, 1)], axis=1
        )
        sort_this = lex_sort(x=sort_this, cluster_columns=self._id_columns)
        unique_sensors = sort_this[:, 0 : unique_sensors.shape[1]]
        counts = sort_this[:, unique_sensors.shape[1] :].flatten().astype(int)

        new_node_col = np.zeros(x.shape[0])
        new_node_col[counts.cumsum()[:-1]] = 1
        new_node_col[0] = 1
        x = np.column_stack([x, new_node_col])

        return Data(x=torch.tensor(x))


class IceMixNodes(NodeDefinition):
    """Calculate ice properties and perform random sampling.

    Ice properties are calculated based on the z-coordinate of the pulse. For
    each event, a random sampling is performed to keep the number of pulses
    below a maximum number of pulses if n_pulses is over the limit.
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_pulses: int = 768,
        z_name: str = "dom_z",
        hlc_name: Optional[str] = "hlc",
        add_ice_properties: bool = True,
        ice_args: Dict[str, Optional[float]] = {
            "z_offset": None,
            "z_scaling": None,
        },
        sample_pulses: bool = True,
    ) -> None:
        """Construct `IceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Minimum
            required features are z coordinate and hlc column names.
            max_pulses: Maximum number of pulses to keep in the event.
            z_name: Name of the z-coordinate column.
            hlc_name: Name of the `Hard Local Coincidence Check` column.
            add_ice_properties: If True, scattering and absoption length of
            ice in IceCube are added to the feature set based on z coordinate.
            ice_args: Offset and scaling of the z coordinate in the Detector,
            to be able to make similar conversion in the ice data.
            sample_pulses: Enable sampling random pulses. If True and the
            event is longer than the max_length, they will be sampled. If
            False, then only the first max_length pulses will be selected.
        """
        if input_feature_names is None:
            input_feature_names = [
                "dom_x",
                "dom_y",
                "dom_z",
                "dom_time",
                "charge",
                "hlc",
                "rde",
            ]

        if add_ice_properties:
            if z_name not in input_feature_names:
                raise ValueError(
                    f"z name '{z_name}' not found in "
                    f"input_feature_names {input_feature_names}"
                )
            self.all_features = input_feature_names + [
                "scatt_lenght",
                "abs_lenght",
            ]
            self.f_scattering, self.f_absoprtion = ice_transparency(**ice_args)
        else:
            self.all_features = input_feature_names

        super().__init__(input_feature_names=input_feature_names)

        if hlc_name not in input_feature_names:
            self.warning(
                f"hlc name '{hlc_name}' not found in input_feature_names"
                f" '{input_feature_names}', subsampling will be random."
            )
            hlc_name = None

        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.input_feature_names = input_feature_names
        self.n_features = len(self.all_features)
        self.max_length = max_pulses
        self.z_name = z_name
        self.hlc_name = hlc_name
        self.add_ice_properties = add_ice_properties
        self.sampling_enabled = sample_pulses

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return self.all_features

    def _add_ice_properties(
        self, graph: torch.Tensor, x: torch.Tensor, ids: List[int]
    ) -> torch.Tensor:

        graph[: len(ids), -2] = torch.tensor(
            self.f_scattering(x[ids, self.feature_indexes[self.z_name]])
        )
        graph[: len(ids), -1] = torch.tensor(
            self.f_absoprtion(x[ids, self.feature_indexes[self.z_name]])
        )
        return graph

    def _pulse_sampler(
        self, x: torch.Tensor, event_length: int
    ) -> torch.Tensor:

        if event_length < self.max_length:
            ids = torch.arange(event_length)
        else:
            ids = torch.randperm(event_length)
            if self.hlc_name is not None:
                auxiliary_n = torch.nonzero(
                    x[:, self.feature_indexes[self.hlc_name]] == 0
                ).squeeze(1)
                auxiliary_p = torch.nonzero(
                    x[:, self.feature_indexes[self.hlc_name]] == 1
                ).squeeze(1)
                ids_n = ids[auxiliary_n][
                    : min(self.max_length, len(auxiliary_n))
                ]
                ids_p = ids[auxiliary_p][
                    : min(self.max_length - len(ids_n), len(auxiliary_p))
                ]

                ids = torch.cat([ids_n, ids_p]).sort().values
            else:
                ids = ids[: self.max_length]

        return ids

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:

        event_length = x.shape[0]
        if self.hlc_name is not None:
            x[:, self.feature_indexes[self.hlc_name]] = torch.logical_not(
                x[:, self.feature_indexes[self.hlc_name]]
            )  # hlc in kaggle was flipped
        if self.sampling_enabled:
            ids = self._pulse_sampler(x, event_length)
        else:
            if event_length < self.max_length:
                ids = torch.arange(event_length)
            else:
                ids = torch.arange(self.max_length)
        
        event_length = min(self.max_length, event_length)

        graph = torch.zeros([event_length, self.n_features])

        if self.add_ice_properties:
            graph = self._add_ice_properties(graph, x, ids)
            non_ice_features = self.all_features[: self.n_features - 2]
        else:
            non_ice_features = self.all_features

        for idx, feature in enumerate(non_ice_features):
            graph[:event_length, idx] = x[ids, self.feature_indexes[feature]]

        return Data(x=graph)
