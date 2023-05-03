"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union, List, Tuple

import torch
from torch.functional import Tensor
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch_geometric.utils import to_dense_batch
from pytorch_lightning import LightningModule


class DynEdgeConv(EdgeConv, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(self.device)

        return x, edge_index


class EdgeConv0(MessagePassing, LightningModule):
    """Implementation of EdgeConv0 layer used in TITO solution for.

    'IceCube - Neutrinos in Deep' kaggle competition.
    """

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        **kwargs: Any,
    ):
        """Construct `EdgeConv0`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv0`.
            aggr: Aggregation method to be used with `EdgeConv0`.
            **kwargs: Additional features to be passed to `EdgeConv0`.
        """
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the module."""
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """Forward pass."""
        if isinstance(x, Tensor):
            x = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        """EdgeConv0 message passing."""
        return self.nn(torch.cat([x_i, x_j - x_i, x_j], dim=-1))  # edgeConv0

    def __repr__(self) -> str:
        """Print out module name."""
        return f"{self.__class__.__name__}(nn={self.nn})"


class EdgeConv1(MessagePassing, LightningModule):
    """Implementation of EdgeConv1 layer used in TITO solution for.

    'IceCube - Neutrinos in Deep' kaggle competition.
    """

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        node_edge_feat_ratio: float = 0.7,
        **kwargs: Any,
    ):
        """Construct `EdgeConv1`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv1`.
            aggr: Aggregation method to be used with `EdgeConv1`.
            node_edge_feat_ratio: Ratio of edge features used.
            **kwargs: Additional features to be passed to `EdgeConv1`.
        """
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.node_edge_feat_ratio = node_edge_feat_ratio

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the module."""
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """Forward pass."""
        if isinstance(x, Tensor):
            x = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        """EdgeConv1 message passing.

        Usually, the number of edge features is 20 or more, and when it is less
        than that, it is the first layer and has four attributes: XYZT.
        """
        edge_cnt = round(x_i.shape[-1] * self.node_edge_feat_ratio)
        if edge_cnt < 20:
            edge_cnt = 4
        edge_ij = torch.cat(
            [(x_j - x_i)[:, :edge_cnt], x_j[:, edge_cnt:]], axis=-1
        )
        return self.nn(torch.cat([x_i, edge_ij], dim=-1))  # edgeConv1

    def __repr__(self) -> str:
        """Print out module name."""
        return f"{self.__class__.__name__}(nn={self.nn})"


class DynTrans(EdgeConv0, LightningModule):
    """Implementation of dynTrans1 layer used in TITO solution for.

    'IceCube - Neutrinos in Deep' kaggle competition.
    """

    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        use_transformer: bool = True,
        dropout: float = 0.0,
        serial_connection: bool = True,
        **kwargs: Any,
    ):
        """Construct `DynTrans`.

        Args:
            nn: The MLP/torch.Module to be used within the `DynTrans`.
            layer_sizes: List of layer sizes to be used in `DynTrans`.
            aggr: Aggregation method to be used with `DynTrans`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            use_transformer: Use of the Transformer layer in 'DynTrans'.
            dropout: Dropout rate to be used in `DynTrans`.
            serial_connection: Use of serial connection in `DynTrans`.
            **kwargs: Additional features to be passed to `DynTrans`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        if layer_sizes is None:
            layer_sizes = [256, 256, 256]
        layers = []
        for ix, (nb_in, nb_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            if ix == 0:
                nb_in *= 3  # edgeConv1
            layers.append(torch.nn.Linear(nb_in, nb_out))
            layers.append(torch.nn.LeakyReLU())
        d_model = nb_out

        # Base class constructor
        super().__init__(nn=torch.nn.Sequential(*layers), aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset
        self.serial_connection = serial_connection
        self.use_trans_in_dyn1 = use_transformer

        self.norm_first = False

        self.norm1 = LayerNorm(d_model, eps=1e-5)  # lNorm

        # Transformer layer(s)
        if use_transformer:
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True,
                dropout=dropout,
                norm_first=self.norm_first,
            )
            self._transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers=1
            )

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        if self.norm_first:
            x = self.norm1(x)  # lNorm

        x_out = super().forward(x, edge_index)

        if x_out.shape[-1] == x.shape[-1] and self.serial_connection:
            x = x + x_out
        else:
            x = x_out

        if not self.norm_first:
            x = self.norm1(x)  # lNorm

        # Recompute adjacency
        edge_index = None

        # Transformer layer
        if self.use_trans_in_dyn1:
            x, mask = to_dense_batch(x, batch)
            x = self._transformer_encoder(x, src_key_padding_mask=~mask)
            x = x[mask]

        return x, edge_index