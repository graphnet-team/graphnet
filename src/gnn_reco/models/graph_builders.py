from abc import ABC, abstractmethod
from typing import List

from torch_geometric.nn import knn_graph
from torch_geometric.data import Data


class GraphBuilder(ABC):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def __call__ (self, data: Data) -> Data:
        pass


class KNNGraphBuilder(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph adjacency according to the k-nearest neighbours."""
    def __init__ (
        self,
        nb_nearest_neighbours: int,
        columns: List[int] = None,
        device: str = None,
    ):
        # Check(s)
        if columns is None:
            columns = [0,1,2]

        # Member variable(s)
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns
        self._device = device

    def __call__ (self, data: Data) -> Data:
        # Constructs the adjacency matrix from the raw, DOM-level data and returns this matrix
        if data.edge_index is not None:
            print("WARNING: GraphBuilder received graph with pre-existing structure. ",
                  "Will overwrite.")

        data.edge_index = knn_graph(
            data.x[:, self._columns],
            self._nb_nearest_neighbours,
            data.batch,
        ).to(self._device)

        return data


#class EuclideanGraphBuilder(GraphBuilder):
#    ...


#class MinkowskiGraphBuilder(GraphBuilder):
#    ...
