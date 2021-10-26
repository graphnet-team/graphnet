from abc import ABC, abstractmethod
from typing import List

from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

class GraphBuilder(ABC):
    @abstractmethod
    def __call__ (self, data: Data) -> Data:
        pass

class KNNGraphBuilder(GraphBuilder):
    def __init__ (self, nb_nearest_neighbours: int, columns: List[int] = [0,1,2], device: str = None):
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns
        self._device = device

    def __call__ (self, data: Data) -> Data:
        # Constructs the adjacency matrix from the raw, DOM-level data and returns this matrix
        if data.edge_index is not None:
            print("WARNING: GraphBuilder received graph with pre-existing structure. Will overwrite.")
        x, batch = data.x, data.batch
        edge_index = knn_graph(x[:, self._columns], self._nb_nearest_neighbours, batch).to(self._device)
        data.edge_index = edge_index
        return data

#class EuclideanGraphBuilder(GraphBuilder):
#    ...

#class MinkowskiGraphBuilder(GraphBuilder):
#    ...