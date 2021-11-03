from abc import abstractmethod
from typing import List, final

import torch
from torch.nn import Module
from torch_geometric.data import Data

from gnn_reco.models.graph_builders import GraphBuilder

class Detector(Module):
    """Base class for all detector-specific read-ins in gnn_reco."""

    @property
    @abstractmethod
    def features(self) -> List[str]:
        """List of features used/assumed by inheriting `Detector` objects."""
        pass

    def __init__(self, graph_builder: GraphBuilder, scalers: List[dict] = None):
        # Base class constructor
        super().__init__()

        # Member variables
        self._graph_builder = graph_builder
        self._scalers = scalers
        if self._scalers:
            print(f"Will use scalers rather than standard preprocessing in {self.__class__.__name__}.")

    @final
    def forward(self, data: Data) -> Data:
        # Convenience variables
        assert data.x.size()[1] == self.nb_inputs, f"Got graph data with incompatible size, {data.x.size()} vs. {self.nb_inputs} expected"
        
        # Graph-bulding
        data = self._graph_builder(data).clone()  # `.clone` is necessary to avoid modifying original tensor in-place

        if self._scalers:
            # # Scaling individual features
            # x_numpy = data.x.detach().cpu().numpy()
            # for key, scaler in self._scalers.items():
            #     ix = self.features.index(key)
            #     data.x[:,ix] = torch.tensor(scaler.transform(x_numpy[:,ix])).type_as(data.x)
            
            # Scaling groups of features | @TEMP, probably
            x_numpy = data.x.detach().cpu().numpy()
            data.x[:,:3] = torch.tensor(self._scalers['xyz'].transform(x_numpy[:,:3])).type_as(data.x)
            data.x[:,3:] = torch.tensor(self._scalers['features'].transform(x_numpy[:,3:])).type_as(data.x)
        
        else:
            # Implementation-specific forward pass (e.g. preprocessing)
            data = self._forward(data)

        return data

    @abstractmethod
    def _forward(self, data: Data) -> Data:
        """Same syntax as `.forward` for implentation in inheriting classes."""
        pass

    @property
    def nb_inputs(self) -> int:
        return len(self.features)

    @property
    def nb_outputs(self) -> int:
        """This the default, but may be overridden by specific inheriting classes."""
        return self.nb_inputs