"""Implementation of the DynEdge GNN model architecture.

[Description of what this architecture does.]

Author: Rasmus Oersoe
Email: ###@###.###
"""
from multiprocessing import pool
from pickle import GLOBAL
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from graphnet.components.layers import DynEdgeConv

from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdge(GNN):
    def __init__(
        self,
        nb_inputs,
        *,
        nb_neighbours: Optional[int] = 8,
        features_subset: Optional[List[int]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
    ):
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    256,
                    336,
                ),
                (
                    256,
                    336,
                ),
                (
                    256,
                    336,
                ),
            ]

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
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

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )

        # Base class constructor
        super().__init__(nb_inputs, self._layer_sizes[-1])

        # Common layer(s)
        self._activation = torch.nn.LeakyReLU()
        nb_global_variables = 5 + nb_inputs

        # Convolutional operations
        nb_input_features = nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += nb_global_variables

        self._conv_layers = []
        for sizes in self._dynedge_layer_sizes:
            layers = []
            for nb_in, nb_out in zip([nb_input_features] + sizes[:-1], sizes):
                layers.append(torch.nn.Linear(nb_in, nb_out))
                layers.append(self._activation)

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="add",
                nb_neighbors=nb_neighbours,
                features_subset=features_subset,
            )
            self._conv_layers.append(conv_layer)

        # Post-processing operations
        nb_latent_features = (
            nb_out * len(self._dynedge_layer_sizes) + nb_input_features
        )

        post_processing_layers = []
        for nb_in, nb_out in zip(
            [nb_latent_features] + self._readout_layer_sizes[:-1],
            self._readout_layer_sizes,
        ):
            post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
            post_processing_layers.append(self._activation)

        self._post_processing = torch.nn.Sequential(*post_processing_layers)

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes) if global_pooling_schemes else 1
        )
        nb_latent_features = nb_out * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent_features += nb_global_variables

        readout_layers = []
        for nb_in, nb_out in zip(
            [nb_latent_features] + self._readout_layer_sizes[:-1],
            self._readout_layer_sizes,
        ):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, batch=batch, dim=0)
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
        """Calculate global variables"""

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
        """Model forward pass.

        Args:
            data (Data): Graph of input features.

        Returns:
            Tensor: Model output.
        """

        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        global_variables = self._calculate_global_variables(
            x,
            edge_index,
            batch,
            torch.log10(data.n_pulses),
        )

        # Distribute global variables out to each node
        if not self._add_global_variables_after_pooling:
            distribute = (
                batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
            ).type(torch.float)

            global_variables_distributed = torch.sum(
                distribute.unsqueeze(dim=2)
                * global_variables.unsqueeze(dim=0),
                dim=1,
            )

            x = torch.cat((x, global_variables_distributed), dim=1)

        # DynEdge-convolutions
        skip_connections = [x]
        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        # Skip-cat
        x = torch.cat(skip_connections, dim=1)

        # Post-processing
        x = self._post_processing(x)

        # (Optional) Global pooling
        if self._global_pooling_schemes:
            x = self._global_pooling(x, batch=batch)
            if self._add_global_variables_after_pooling:
                x = torch.cat(
                    [
                        x,
                        global_variables.unsqueeze(1),
                    ],
                    dim=1,
                )

        # Read-out
        x = self._readout(x)

        return x
