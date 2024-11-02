"""Implementation of DynEdge architecture used in.

                    IceCube - Neutrinos in Deep Ice
Reconstruct the direction of neutrinos from the Universe to the South Pole

Kaggle competition.

Solution by TITO.
"""

from typing import List, Tuple, Optional

import torch
from torch import Tensor, LongTensor

from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynTrans
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdgeTITO(GNN):
    """DynEdgeTITO (dynamical edge convolutional with Transformer) model."""

    def __init__(
        self,
        nb_inputs: int,
        features_subset: Optional[List[int]] = None,
        dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        global_pooling_schemes: List[str] = ["max"],
        use_global_features: bool = True,
        use_post_processing_layers: bool = True,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        n_head: int = 8,
        nb_neighbours: int = 8,
    ):
        """Construct `DynEdgeTITO`.

        Args:
            nb_inputs: Number of input features on each node.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2,3].
            dyntrans_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynTrans` layer.
                Defaults to [(256, 256), (256, 256), (256, 256), (256, 256)].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            use_global_features: Whether to use global features after pooling.
            use_post_processing_layers: Whether to use post-processing layers
                after the `DynTrans` layers.
            post_processing_layer_sizes: The layer sizes used in the
                post-processing layers. Defaults to [336, 256].
            readout_layer_sizes: The layer sizes used in the readout layers.
                Defaults to [256, 128].
            n_head: The number of heads to use in the `DynTrans` layer.
            nb_neighbours: The number of neighbours to use in the `DynTrans`
                layer.
        """
        # DynTrans layer sizes
        if dyntrans_layer_sizes is None:
            dyntrans_layer_sizes = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

        assert isinstance(dyntrans_layer_sizes, list)
        assert len(dyntrans_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dyntrans_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dyntrans_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dyntrans_layer_sizes
        )

        self._dyntrans_layer_sizes = dyntrans_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                256,
                128,
            ]

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

        assert self._global_pooling_schemes, (
            "No global pooling schemes were request, so cannot add global"
            " variables after pooling."
        )

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.LeakyReLU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset or [0, 1, 2, 3]
        self._use_global_features = use_global_features
        self._use_post_processing_layers = use_post_processing_layers
        self._n_head = n_head
        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dyntrans_layer_sizes:
            conv_layer = DynTrans(
                [nb_latent_features] + list(sizes),
                aggr="max",
                features_subset=self._features_subset,
                n_head=self._n_head,
            )
            self._conv_layers.append(conv_layer)
            nb_latent_features = sizes[-1]

        if self._use_post_processing_layers:
            post_processing_layers = []
            layer_sizes = [nb_latent_features] + list(
                self._post_processing_layer_sizes
            )
            for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
                post_processing_layers.append(self._activation)
            last_posting_layer_output_dim = nb_out

            self._post_processing = torch.nn.Sequential(
                *post_processing_layers
            )
        else:
            last_posting_layer_output_dim = nb_latent_features

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = last_posting_layer_output_dim * nb_poolings
        if self._use_global_features:
            nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

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

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self._use_global_features:
            global_variables = self._calculate_global_variables(
                x,
                edge_index,
                batch,
                torch.log10(data.n_pulses),
            )

        # DynEdge-convolutions
        for conv_layer in self._conv_layers:
            x = conv_layer(x, edge_index, batch)

        # Post-processing
        if self._use_post_processing_layers:
            x = self._post_processing(x)

        x = self._global_pooling(x, batch=batch)
        if self._use_global_features:
            x = torch.cat(
                [
                    x,
                    global_variables,
                ],
                dim=1,
            )

        # Read-out
        x = self._readout(x)

        return x
