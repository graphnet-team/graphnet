from abc import abstractmethod
from typing import List

from torch.nn import Module

from gnn_reco.models.graph_builders import GraphBuilder

class Detector(Module):
    """Base class for all detector-specific read-ins in gnn_reco."""

    @property
    @abstractmethod
    def features(self) -> List[str]:
        """List of features used/assumed by inheriting `Detector` objects."""
        pass

    def __init__(self, graph_builder: GraphBuilder):
        # Base class constructor
        super().__init__()

        # Member variables
        self._graph_builder = graph_builder

    @property
    def nb_inputs(self):
        return len(self.features)

    @property
    def nb_outputs(self):
        """This the default, but may be overridden by specific inheriting classes."""
        return self.nb_inputs