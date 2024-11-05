"""RNN_DynEdge model implementation."""

from typing import List, Optional, Tuple

import torch
from graphnet.models.gnn.gnn import GNN
from graphnet.models.gnn.dynedge_kaggle_tito import DynEdgeTITO
from graphnet.models.rnn.node_rnn import Node_RNN

from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class RNN_TITO(GNN):
    """The RNN_TITO model class.

    Combines the Node_RNN and DynEdgeTITO models, intended for data with large
    amount of DOM activations per event. This model works only with non-
    standard dataset specific to the Node_RNN model see Node_RNN for more
    details.
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        time_series_columns: List[int],
        *,
        nb_neighbours: int = 8,
        rnn_layers: int = 2,
        rnn_hidden_size: int = 64,
        rnn_dropout: float = 0.5,
        features_subset: Optional[List[int]] = None,
        dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: List[str] = ["max"],
        embedding_dim: Optional[int] = None,
        n_head: int = 16,
        use_global_features: bool = True,
        use_post_processing_layers: bool = True,
    ):
        """Initialize the RNN_DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            time_series_columns (List[int]): The indices of the input data that
                should be treated as time series data.
                The first index should be the charge column.
            nb_neighbours (int, optional): Number of neighbours to consider.
                Defaults to 8.
            rnn_layers (int, optional): Number of RNN layers.
                Defaults to 1.
            rnn_hidden_size (int, optional): Size of the hidden state of the
                RNN. Also determines the size of the output of the RNN.
                Defaults to 64.
            rnn_dropout (float, optional): Dropout to use in the RNN.
                Defaults  to 0.5.
            features_subset (List[int], optional): The subset of latent
                features on each node that are used as metric dimensions when
                performing the k-nearest neighbours clustering.
                Defaults to [0,1,2,3]
            dyntrans_layer_sizes (List[Tuple[int, ...]], optional): List of
                tuples representing the sizes of the hidden layers of
                the DynTrans model.
            post_processing_layer_sizes (List[int], optional): List of
                integers representing the sizes of the hidden layers of the
                post-processing model.
            readout_layer_sizes (List[int], optional): List of integers
                representing the sizes of the hidden layers of the
                readout model.
            global_pooling_schemes (Union[str, List[str]], optional): Pooling
                schemes to use. Defaults to None.
            embedding_dim (int, optional): Embedding dimension of the RNN.
                Defaults to None ie. no embedding.
            n_head (int, optional): Number of heads to use in the DynTrans
                model. Defaults to 16.
            use_global_features (bool, optional): Whether to use global
                features after pooling. Defaults to True.
            use_post_processing_layers (bool, optional): Whether to use
                post-processing layers after the DynTrans layers.
                Defaults to True.
        """
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = nb_inputs
        self._rnn_layers = rnn_layers
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_dropout = rnn_dropout
        self._embedding_dim = embedding_dim
        self._n_head = n_head
        self._use_global_features = use_global_features
        self._use_post_processing_layers = use_post_processing_layers

        self._features_subset = features_subset
        if dyntrans_layer_sizes is None:
            dyntrans_layer_sizes = [
                (256, 256),
                (256, 256),
                (256, 256),
                (256, 256),
            ]
        else:
            dyntrans_layer_sizes = [
                tuple(layer_sizes) for layer_sizes in dyntrans_layer_sizes
            ]

        self._dyntrans_layer_sizes = dyntrans_layer_sizes
        self._post_processing_layer_sizes = post_processing_layer_sizes
        self._global_pooling_schemes = global_pooling_schemes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                256,
                128,
            ]
        self._readout_layer_sizes = readout_layer_sizes

        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        self._rnn = Node_RNN(
            nb_inputs=2,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_layers,
            time_series_columns=time_series_columns,
            nb_neighbours=self._nb_neighbours,
            features_subset=self._features_subset,
            dropout=self._rnn_dropout,
            embedding_dim=self._embedding_dim,
        )

        self._dynedge_tito = DynEdgeTITO(
            nb_inputs=self._rnn_hidden_size + 5,
            dyntrans_layer_sizes=self._dyntrans_layer_sizes,
            features_subset=self._features_subset,
            global_pooling_schemes=self._global_pooling_schemes,
            use_global_features=self._use_global_features,
            use_post_processing_layers=self._use_post_processing_layers,
            post_processing_layer_sizes=self._post_processing_layer_sizes,
            readout_layer_sizes=self._readout_layer_sizes,
            n_head=self._n_head,
            nb_neighbours=self._nb_neighbours,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass of the RNN and tito model."""
        data = self._rnn(data)
        readout = self._dynedge_tito(data)

        return readout
