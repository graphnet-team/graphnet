"""Base detector-specific `Model` class(es)."""

from abc import abstractmethod
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from graphnet.models.graph_builders import GraphBuilder
from graphnet.models import Model
from graphnet.utilities.config import save_model_config
from graphnet.utilities.decorators import final


class Detector(Model):
    """Base class for all detector-specific read-ins in graphnet."""

    @property
    @abstractmethod
    def features(self) -> List[str]:
        """List of features used/assumed by inheriting `Detector` objects."""

    @save_model_config
    def __init__(
        self, graph_builder: GraphBuilder, scalers: List[dict] = None
    ):
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member variables
        self._graph_builder = graph_builder
        self._scalers = scalers
        if self._scalers:
            self.info(
                (
                    "Will use scalers rather than standard preprocessing "
                    f"in {self.__class__.__name__}."
                )
            )

    @final
    def forward(self, data: Data) -> Data:
        """Pre-process graph `Data` features and build graph adjacency."""
        # Check(s)
        assert data.x.size()[1] == self.nb_inputs, (
            "Got graph data with incompatible size, ",
            f"{data.x.size()} vs. {self.nb_inputs} expected",
        )

        # Graph-bulding
        # @NOTE: `.clone` is necessary to avoid modifying original tensor in-place.
        data = self._graph_builder(data).clone()

        if self._scalers:
            # # Scaling individual features
            # x_numpy = data.x.detach().cpu().numpy()
            # for key, scaler in self._scalers.items():
            #     ix = self.features.index(key)
            #     data.x[:,ix] = torch.tensor(scaler.transform(x_numpy[:,ix])).type_as(data.x)

            # Scaling groups of features | @TEMP, probably
            x_numpy = data.x.detach().cpu().numpy()

            data.x[:, :3] = torch.tensor(
                self._scalers["xyz"].transform(x_numpy[:, :3])  # type: ignore[call-overload]
            ).type_as(data.x)

            data.x[:, 3:] = torch.tensor(
                self._scalers["features"].transform(x_numpy[:, 3:])  # type: ignore[call-overload]
            ).type_as(data.x)

        else:
            # Implementation-specific forward pass (e.g. preprocessing)
            data = self._forward(data)

        return data

    @abstractmethod
    def _forward(self, data: Data) -> Data:
        """Syntax like `.forward`, for implentation in inheriting classes."""

    @property
    def nb_inputs(self) -> int:
        """Return number of input features."""
        return len(self.features)

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return self.nb_inputs

    def _validate_features(self, data: Data) -> None:
        if isinstance(data, Batch):
            # `data.features` is "transposed" and each list element contains only duplicate entries.

            if (
                len(data.features[0]) == data.num_graphs
                and len(set(data.features[0])) == 1
            ):
                data_features = [features[0] for features in data.features]

            # `data.features` is not "transposed" and each list element
            # contains the original features.
            else:
                data_features = data.features[0]
        else:
            data_features = data.features
        assert (
            data_features == self.features
        ), f"Features on Data and Detector differ: {data_features} vs. {self.features}"
