"""Implementation of the NodeTimeRNN model.

(cannot be used as a standalone model)
"""

import torch

from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn_graph
from typing import List, Optional


from graphnet.models.components.embedding import SinusoidalPosEmb


class Node_RNN(GNN):
    """Implementation of the Node RNN model architecture.

    The model takes as input the typical DOM data format and transforms it into
    a time series of DOM activations pr. DOM. before applying a RNN layer and
    outputting the an RNN output for each DOM. This model is in its current
    state not intended to be used as a standalone model. Furthermore, it needs
    to be used with a time-series dataset object, where the last column in x is
    a special column that is used to seperate the activation into time series
    per dom per batch.
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        hidden_size: int,
        num_layers: int,
        time_series_columns: List[int],
        nb_neighbours: int = 8,
        features_subset: Optional[List[int]] = None,
        dropout: float = 0.5,
        embedding_dim: int = 0,
    ) -> None:
        """Construct `Node_RNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden
                layers.
            num_layers: Number of layers in the RNN.
            time_series_columns: The indices of the input data that should be
                treated as time series data. The first index should be
                the charge column.
            nb_neighbours: Number of neighbours to use when reconstructing the
                graph representation. Defaults to 8.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2,3]
            dropout: Dropout fraction to use in the RNN. Defaults to 0.5.
                embedding_dim: Embedding dimension of the RNN.
                Defaults to no embedding.
            embedding_dim: Dimension of the embedding. Defaults to 0.
        """
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._time_series_columns = time_series_columns
        self._nb_neighbors = nb_neighbours
        self._features_subset = features_subset
        self._embedding_dim = embedding_dim
        self._nb_inputs = nb_inputs

        super().__init__(nb_inputs, hidden_size + 5)

        if self._embedding_dim != 0:
            self._nb_inputs = self._embedding_dim * nb_inputs

        self._rnn = torch.nn.GRU(
            num_layers=self._num_layers,
            input_size=self._nb_inputs,
            hidden_size=self._hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self._emb = SinusoidalPosEmb(dim=self._embedding_dim)

    def clean_up_data_object(self, data: Data) -> Data:
        """Update the feature names of the data object.

        Args:
            data: The input data object.
        """
        # old features removing the new_node column
        old_features = data.features[0][:-1]
        new_features = old_features + [
            "rnn_out_" + str(i) for i in range(self._hidden_size)
        ]
        data.features = [new_features] * len(data.features)
        for i, name in enumerate(old_features):
            data[name] = data.x[i]
        return data

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass to the GNN."""
        # cutter = data.cutter.cumsum(0)[:-1]
        # Optional embedding of the time and charge time series data.
        x = data.x
        time_series = x[:, self._time_series_columns]
        if self._embedding_dim != 0:
            time_series = self._emb(time_series * 4096).reshape(
                (
                    time_series.shape[0],
                    self._embedding_dim * time_series.shape[-1],
                )
            )
        # Create the dom + batch unique splitter from the new_node_col
        splitter = x[:, -1].argwhere()[1:].flatten().cpu()
        time_series = time_series.tensor_split(splitter)
        # apply RNN per DOM irrespective of batch and return the final state.
        time_series = torch.nn.utils.rnn.pack_sequence(
            time_series, enforce_sorted=False
        )
        rnn_out = self._rnn(time_series)[-1][0]
        # prepare node level features
        charge = data.x[:, self._time_series_columns[0]].tensor_split(splitter)
        charge = torch.tensor(
            [
                torch.asinh(5 * torch.sum(node_charges) / 5)
                for node_charges in charge
            ]
        )
        batch = data.batch[x[:, -1].bool()]
        x = x[x[:, -1].bool()][:, :-1]
        x[:, self._time_series_columns[0]] = charge

        # combine the RNN output with the DOM summary features
        data.x = torch.hstack([x, rnn_out])
        # correct the batches
        data.batch = batch
        data = self.clean_up_data_object(data)
        # Recompute adjacency
        data.edge_index = knn_graph(
            x=x[:, self._features_subset],
            k=self._nb_neighbors,
            batch=batch,
        ).to(self.device)

        return data
