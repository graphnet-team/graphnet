"""Implementation of the ParticleNet GNN model architecture."""
from typing import List, Optional, Callable, Tuple, Union

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynEdgeConv
from graphnet.models.gnn.gnn import GNN

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class ParticleNeT(GNN):
    """ParticleNeT (dynamical edge convolutional) model.

    Inspired by: https://arxiv.org/abs/1902.08570
    """

    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 16,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynamic: bool = True,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
        ],
        readout_layer_sizes: Optional[List[int]] = [256],
        global_pooling_schemes: Optional[Union[str, List[str]]] = "mean",
        activation_layer: Optional[str] = "relu",
        add_batchnorm_layer: bool = True,
        dropout_readout: float = 0.1,
        skip_readout: bool = False,
    ):
        """Construct `ParticleNeT`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynamic: wether or not update the edges after every `DynEdgeConv`
                block.
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-three tuples means that all `DynEdgeConv` layers contain
                a three-layer MLP.
                Defaults to [(64, 64, 64), (128, 128, 128), (256, 256, 256)].
            readout_layer_sizes: Hidden layer size in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer in the model, it yields the output of the `DynEdge`
                model. Defaults to [256,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
                Default to "mean".
            activation_layer: The activation function to use in the model.
                Default to "relu".
            add_batchnorm_layer: Whether to add a batch normalization layer
                after each linear layer. Default to True.
            dropout_readout: Dropout value to use in the readout layer(s).
                Default to 0.1.
            skip_readout: Whether to skip the readout layer(s). If `True`, the
                output of the last DynEdgeConv block is returned directly.
        """
        # Latent feature subset for computing nearest neighbours in model
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (64, 64, 64),
                (
                    128,
                    128,
                    128,
                ),
                (
                    256,
                    256,
                    256,
                ),
            ]

        dynedge_layer_sizes_check = []
        for sizes in dynedge_layer_sizes:
            if isinstance(sizes, list):
                sizes = tuple(sizes)
            dynedge_layer_sizes_check.append(sizes)

        assert isinstance(dynedge_layer_sizes_check, list)
        assert len(dynedge_layer_sizes_check)
        assert all(
            isinstance(sizes, tuple) for sizes in dynedge_layer_sizes_check
        )
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes_check)
        assert all(
            all(size > 0 for size in sizes)
            for sizes in dynedge_layer_sizes_check
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes_check

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                256,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if activation_layer is None or activation_layer.lower() == "relu":
            activation_layer = torch.nn.ReLU()
        elif activation_layer.lower() == "gelu":
            activation_layer = torch.nn.GELU()
        else:
            raise ValueError(
                f"Activation layer {activation_layer} not supported."
            )

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = activation_layer
        self._nb_inputs = nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset
        self._dynamic = dynamic
        self._add_batchnorm_layer = add_batchnorm_layer
        self._dropout_readout = dropout_readout
        self._skip_readout = skip_readout

        self._construct_layers()

    # Builds the network
    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(torch.nn.Linear(nb_in, nb_out))
                if self._add_batchnorm_layer:
                    layers.append(torch.nn.BatchNorm1d(nb_out))
                layers.append(self._activation)

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="mean",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = nb_out * nb_poolings

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)
            readout_layers.append(torch.nn.Dropout(self._dropout_readout))

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # DynEdge-convolutions
        for conv_layer in self._conv_layers:
            if self._dynamic:
                x, edge_index = conv_layer(x, edge_index, batch)
            else:
                x, _ = conv_layer(x, edge_index, batch)

        # Read-out
        if not self._skip_readout:
            # (Optional) Global pooling
            if self._global_pooling_schemes:
                x = self._global_pooling(x, batch=batch)

            # Read-out
            x = self._readout(x)

        return x
