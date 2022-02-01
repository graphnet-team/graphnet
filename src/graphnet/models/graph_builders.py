from abc import ABC, abstractmethod
from typing import List

import torch
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


class EuclideanGraphBuilder(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph adjacency """
    def __init__ (
        self,
        sigma: float, #retains/requires grad? here?
        columns: List[int] = None,
        device: str = None,
    ):
        # Check(s) 
        if columns is None:
            columns = [0,1,2]

        # Member variable(s)
        self._sigma = torch.tensor(sigma, requires_grad=True)
        self._columns = columns
        self._device = device

    def __call__ (self, data: Data) -> Data:
        # Constructs the adjacency matrix from the raw, DOM-level data and returns this matrix
        if data.edge_index is not None:
            print("WARNING: GraphBuilder received graph with pre-existing structure. ",
                  "Will overwrite.")

        data.edge_index, data.edge_weight = euclidean_graph_builder(
            data.x[:, self._columns],
            self._sigma,
            #data.batch,
        ).to(self._device)

        return data


def euclidean_graph_builder(coords, sigma):

    def g_kernel(sigma,arr1,arr2):
        return torch.exp((-0.5 * torch.linalg.norm(arr1-arr2)**2)/sigma**2)
    
    n_doms = coords.shape[0]
    affinity_mat = -1*torch.ones((n_doms,n_doms))
    weighted_adj_mat = -1*torch.ones((n_doms,n_doms))

    for i in range(n_doms):
        for j in range(n_doms):
            affinity_mat[i,j] = g_kernel(sigma,coords[i,:],coords[j,:])
        for k in range(n_doms):
            weighted_adj_mat[i,k] = torch.exp(affinity_mat[i,k])/torch.sum(torch.exp(affinity_mat[i,:]))

    adj_list = []
    edge_weights = []
    for row in range(n_doms): 
        for col in range(n_doms):
            adj_list.append([row,col])
            edge_weights.append(weighted_adj_mat[row,col])        

    return torch.tensor(adj_list).t().contiguous(),torch.tensor(edge_weights)

#class EuclideanGraphBuilder(GraphBuilder):
#    ...


#class MinkowskiGraphBuilder(GraphBuilder):
#    ...
