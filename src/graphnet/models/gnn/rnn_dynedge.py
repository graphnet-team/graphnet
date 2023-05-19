"""RNN_DynEdge model implementation."""
from typing import List, Optional, Tuple, Union

import torch
from graphnet.models.gnn.gnn import GNN
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.node_rnn import Node_RNN

from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class RNN_DynEdge(GNN):
    """The RNN_DynEdge model class."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        RNN_layers: int = 1,
        RNN_hidden_size: int = 64,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
    ):
        """Initialize the RNN_DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_neighbours (int, optional): Number of neighbours to consider.
                Defaults to 8.
            RNN_layers (int, optional): Number of RNN layers.
                Defaults to 1.
            RNN_hidden_size (int, optional): Size of the hidden state of the RNN. Also determines the size of the output of the RNN.
                Defaults to 64.
            features_subset (Optional[Union[List[int], slice]], optional): Subset of features to use.
            dynedge_layer_sizes (Optional[List[Tuple[int, ...]]], optional): List of tuples of integers representing the sizes of the hidden layers of the DynEdge model.
            post_processing_layer_sizes (Optional[List[int]], optional): List of integers representing the sizes of the hidden layers of the post-processing model.
            readout_layer_sizes (Optional[List[int]], optional): List of integers representing the sizes of the hidden layers of the readout model.
            global_pooling_schemes (Optional[Union[str, List[str]]], optional): Pooling schemes to use. Defaults to None.
            add_global_variables_after_pooling (bool, optional): Whether to add global variables after pooling. Defaults to False.
        """
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = nb_inputs
        self._RNN_layers = RNN_layers
        self._RNN_hidden_size = RNN_hidden_size

        self._feautres_subset = features_subset
        self._dynedge_layer_sizes = dynedge_layer_sizes
        self._post_processing_layer_sizes = post_processing_layer_sizes
        self._global_pooling_schemes = global_pooling_schemes
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
            ]
        self._readout_layer_sizes = readout_layer_sizes

        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        self._rnn = Node_RNN(
            num_layers=self._RNN_layers,
            nb_inputs=self._nb_inputs,
            hidden_size=self._RNN_hidden_size,
            nb_neighbours=self._nb_neighbours,
        )

        self._dynedge = DynEdge(
            nb_inputs=self._RNN_hidden_size + 5,
            nb_neighbours=self._nb_neighbours,
            dynedge_layer_sizes=self._dynedge_layer_sizes,
            post_processing_layer_sizes=self._post_processing_layer_sizes,
            readout_layer_sizes=self._readout_layer_sizes,
            global_pooling_schemes=self._global_pooling_schemes,
            add_global_variables_after_pooling=self._add_global_variables_after_pooling,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass of the RNN and DynEdge models."""
        data = self._rnn(data)
        readout = self._dynedge(data)

        return readout
