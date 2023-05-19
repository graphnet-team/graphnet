"""Implementation of the NodeTimeRNN model.

(cannot be used as a standalone model)
"""
import torch

from graphnet.models.gnn.gnn import GNN
from graphnet.models.components.DOM_handling import (
    append_dom_id,
    DOM_time_series_to_pack_sequence,
    DOM_to_time_series,
    knn_graph_ignore,
)
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class Node_RNN(GNN):
    """Implementation of the RNN model architecture.

    The model takes as input the typical DOM data format and transforms it into
    a time series of DOM activations pr. DOM. before applying a RNN layer and
    outputting the an RNN output for each DOM.
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        hidden_size: int,
        num_layers: int,
        nb_neighbours: int,
    ) -> None:
        """Construct `NodeTimeRNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            nb_neighbours: Number of neighbours to use when reconstructing the graph representation.
        """
        self._nb_neighbours = nb_neighbours
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = nb_inputs
        super().__init__(nb_inputs, hidden_size + 5)

        self._rnn = torch.nn.RNN(
            num_layers=self._num_layers,
            input_size=self._nb_inputs,
            hidden_size=self._hidden_size,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass to the GNN."""
        x, batch = data.x, data.batch

        x = append_dom_id(x, batch, device=self.device)  # add dom id to data
        x, batch = DOM_to_time_series(
            x, batch
        )  # create time series for each dom
        x, xyztt = DOM_time_series_to_pack_sequence(
            x, device=self.device
        )  # pack time series into pack sequence
        x = self._rnn(x)[-1][0]  # apply rnn layer

        x = torch.hstack(
            [xyztt, x]
        )  # reintroduce x/y/z-coordinates and time of first/mean activation for each DOM

        batch, sort_index = batch.sort()
        data.x = x[sort_index]
        data.batch = batch
        edge_index = knn_graph_ignore(
            x=data.x,
            k=self._nb_neighbours,
            batch=data.batch,
        ).to(self.device)
        data.edge_index = edge_index
        return data
