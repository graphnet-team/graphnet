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


class EdgeConvTito(MessagePassing, LightningModule):
    """Implementation of EdgeConvTito layer used in TITO solution for.

    'IceCube - Neutrinos in Deep' kaggle competition.
    """

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        **kwargs: Any,
    ):
        """Construct `EdgeConvTito`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConvTito`.
            aggr: Aggregation method to be used with `EdgeConvTito`.
            **kwargs: Additional features to be passed to `EdgeConvTito`.
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
        """Edgeconvtito message passing."""
        return self.nn(
            torch.cat([x_i, x_j - x_i, x_j], dim=-1)
        )  # EdgeConvTito

    def __repr__(self) -> str:
        """Print out module name."""
        return f"{self.__class__.__name__}(nn={self.nn})"


class DynTrans(EdgeConvTito, LightningModule):
    """Implementation of dynTrans1 layer used in TITO solution for.

    'IceCube - Neutrinos in Deep' kaggle competition.
    """

    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        aggr: str = "max",
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        n_head: int = 8,
        **kwargs: Any,
    ):
        """Construct `DynTrans`.

        Args:
            nn: The MLP/torch.Module to be used within the `DynTrans`.
            layer_sizes: List of layer sizes to be used in `DynTrans`.
            aggr: Aggregation method to be used with `DynTrans`.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            n_head: Number of heads to be used in the multiheadattention models.
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
        self.features_subset = features_subset

        self.norm1 = LayerNorm(d_model, eps=1e-5)  # lNorm

        # Transformer layer(s)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True,
            norm_first=False,
        )
        self._transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=1
        )

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        x_out = super().forward(x, edge_index)

        if x_out.shape[-1] == x.shape[-1]:
            x = x + x_out
        else:
            x = x_out

        x = self.norm1(x)  # lNorm

        # Transformer layer
        x, mask = to_dense_batch(x, batch)
        x = self._transformer_encoder(x, src_key_padding_mask=~mask)
        x = x[mask]

        return x
