"""Implementation of DynEdge architecture used in.

                    IceCube - Neutrinos in Deep Ice
Reconstruct the direction of neutrinos from the Universe to the South Pole

Kaggle competition.

Solution by TITO.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, LongTensor
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynTrans
from graphnet.utilities.config import save_model_config
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdgeTITO(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
        use_global_variables: bool = True,
        serial_connection: bool = True,
        use_postprocessing_layers: bool = True,
        use_tranformer_in_last: bool = True,
        dropout: float = 0.0,
    ):
        """Construct `DynEdge`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
            use_global_variables: Whether to use global variables in the GNN.
            serial_connection: Whether to use a serial connection in the GNN.
            use_postprocessing_layers: Whether to use post-processing layers.
            use_tranformer_in_last: Whether to use a transformer convolution
            after the DynEdge convolutional layers.
            dropout: Dropout probability inside 'DynTrans' layer.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 4)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
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
                256,
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
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.LeakyReLU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset
        self._serial_connection = serial_connection
        self._use_postprocessing_layers = use_postprocessing_layers
        self._use_global_variables = use_global_variables
        self._use_tranformer_in_last = use_tranformer_in_last
        self._dropout = dropout

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if self._use_global_variables:
            if not self._add_global_variables_after_pooling:
                nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            conv_layer = DynTrans(
                [nb_latent_features] + list(sizes),
                aggr="max",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
                dropout=self._dropout,
                serial_connection=self._serial_connection,
            )
            self._conv_layers.append(conv_layer)
            nb_latent_features = sizes[-1]

        # Post-processing operations
        if self._serial_connection:
            nb_latent_features = self._dynedge_layer_sizes[-1][-1]
        else:
            nb_latent_features = (
                sum(sizes[-1] for sizes in self._dynedge_layer_sizes)
                + nb_input_features
            )

        if self._use_postprocessing_layers:
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
        if self._use_global_variables:
            if self._add_global_variables_after_pooling:
                nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)

        # Transformer layer(s)
        if self._use_tranformer_in_last:
            encoder_layer = TransformerEncoderLayer(
                d_model=last_posting_layer_output_dim,
                nhead=8,
                batch_first=True,
                dropout=self._dropout,
                norm_first=False,
            )
            self._transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers=self._use_tranformer_in_last
            )

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

        if self._use_global_variables:
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
        if self._serial_connection:
            for conv_layer in self._conv_layers:
                x, _edge_index = conv_layer(x, data.edge_index, batch)
        else:
            skip_connections = [x]
            for conv_layer in self._conv_layers:
                x, _edge_index = conv_layer(x, data.edge_index, batch)
                skip_connections.append(x)

            # Skip-cat
            x = torch.cat(skip_connections, dim=1)

        # Transformer convolutions
        if self._use_tranformer_in_last:
            x, mask = to_dense_batch(x, batch)
            x = self._transformer_encoder(x, src_key_padding_mask=mask)
            x = x[mask]

        # Post-processing
        if self._use_postprocessing_layers:
            x = self._post_processing(x)

        # (Optional) Global pooling
        if self._use_global_variables:
            x = self._global_pooling(x, batch=batch)
            if self._add_global_variables_after_pooling:
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
