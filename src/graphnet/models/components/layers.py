"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union, List

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import (
    knn_graph,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from torch.nn.functional import linear
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_batch, softmax
from torch_scatter import scatter

from pytorch_lightning import LightningModule

from graphnet.models.utils import get_log_deg


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
            layer_sizes: List of layer sizes to be used in `DynTrans`.
            aggr: Aggregation method to be used with `DynTrans`.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            n_head: Number of heads to be used in the multiheadattention
                models.
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
            layers.append(nn.Linear(nb_in, nb_out))
            layers.append(nn.LeakyReLU())
        d_model = nb_out

        # Base class constructor
        super().__init__(nn=nn.Sequential(*layers), aggr=aggr, **kwargs)

        # Additional member variables
        self.features_subset = features_subset

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)  # lNorm

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


class DropPath(LightningModule):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(
        self,
        drop_prob: float = 0.0,
    ):
        """Construct `DropPath`.

        Args:
            drop_prob: Probability of dropping a path during training.
                If 0.0, no paths are dropped. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        """Return extra representation of the module."""
        return "p={}".format(self.drop_prob)


class Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features "
                f"{in_features} instead"
            )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.output_projection = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x


class Block_rel(LightningModule):
    """Implementation of BEiTv2 Block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Block_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key,
                and value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key
                matrices in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the
                `Attention_rel` layer.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = Attention_rel(
            input_dim,
            num_heads,
            attn_drop=attn_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rel_pos_bias: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    kv,
                    kv,
                    rel_pos_bias=rel_pos_bias,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.gamma_1
                * self.drop_path(
                    self.attn(
                        xn,
                        kv,
                        kv,
                        rel_pos_bias=rel_pos_bias,
                        key_padding_mask=key_padding_mask,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Attention_rel(LightningModule):
    """Attention mechanism with relative position bias."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Attention_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            qk_scale: a scaling factor that multiplies the dot product of query
                and key vectors. Defaults to None. If None, computed as
                :math: `head_dim^(-1/2)`.
            attn_drop: the dropout probability for the attention weights.
                Defaults to 0.0.
            proj_drop: the dropout probability for the output of the attention
                module. Defaults to 0.0.
            attn_head_dim: the feature dimensionality of each attention head.
                Defaults to None. If None, computed as `dim // num_heads`.
        """
        if input_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got input_dim={input_dim} and num_heads={num_heads} instead"
            )

        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or input_dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rel_pos_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        batch_size, event_length, _ = q.shape

        q = linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(batch_size, event_length, self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        k = linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(batch_size, k.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        v = linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(batch_size, v.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            )
            bias[
                torch.max(
                    key_padding_mask[:, None, :], key_padding_mask[:, :, None]
                )
                < 0
            ] = 0
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(batch_size, event_length, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(LightningModule):
    """Transformer block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """Construct 'Block'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the
                `MultiheadAttention` layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `MultiheadAttention`
                layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class GritSparseMHA(LightningModule):
    """Proposed Attention Computation for GRIT.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/layer/grit_layer.py
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        use_bias: bool,
        clamp: float = 5.0,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
        edge_enhance: bool = True,
    ):
        """Construct 'GritSparseMHA'.

        Args:
            in_dim: Dimension of the input tensor.
            out_dim: Dimension of the output tensor.
            num_heads: Number of attention heads.
            use_bias: Apply bias the key and value linear layers.
            clamp: Clamp the absolute value of the attention scores to a value.
            dropout: Dropout layer probability.
            activation: Uninstantiated activation function.
                E.g. `torch.nn.ReLU`
            edge_enhance: Applies learnable weight matrix with node-pair in
                output node calculation.
        """
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(
            torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True
        )
        nn.init.xavier_normal_(self.Aw)

        # TODO: Better activation function handling -PW
        self.activation = activation()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.VeRow)

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        Q_x = self.Q(data.x)
        K_x = self.K(data.x)
        V_x = self.V(data.x)

        if data.get("edge_attr", None) is not None:
            E = self.E(data.edge_attr)
        else:
            E = None

        Q_x = Q_x.view(-1, self.num_heads, self.out_dim)
        K_x = K_x.view(-1, self.num_heads, self.out_dim)
        V_x = V_x.view(-1, self.num_heads, self.out_dim)

        # Applying Eq. 2 of the GRIT paper:
        src = K_x[data.edge_index[0]]  # (num relative) x num_heads x out_dim
        dest = Q_x[data.edge_index[1]]  # (num relative) x num_heads x out_dim
        score = src + dest  # element-wise multiplication
        if E is not None:
            E = E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = E[:, :, : self.out_dim], E[:, :, self.out_dim :]
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(
                torch.relu(-score)
            )
            score = score + E_b

        score = self.activation(score)
        e_t = score  # ehat_ij

        # Output edge
        if E is not None:
            wE = score.flatten(1)

        # Complete attention calculation
        score = torch.einsum("ehd, dhc->ehc", score, self.Aw)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        score = softmax(score, index=data.edge_index[1]).to(
            dtype=data.x.dtype
        )  # (num relative) x num_heads x 1
        score = self.dropout(score)

        # Aggregate with Attn-Score
        V_x_weighted = (
            V_x[data.edge_index[0]] * score
        )  # (num relative) x num_heads x out_dim
        wV = torch.zeros_like(
            V_x, dtype=score.dtype
        )  # (num nodes in batch) x num_heads x out_dim
        scatter(V_x_weighted, data.edge_index[1], dim=0, out=wV, reduce="add")

        # Adds the second term (W_Ev ehhat_ij) in the last line of Eq. 2
        if self.edge_enhance and E is not None:
            rowV = scatter(
                e_t * score, data.edge_index[1], dim=0, reduce="add"
            )
            rowV = torch.einsum("nhd, dhc -> nhc", rowV, self.VeRow)
            wV = wV + rowV

        return wV, wE


class GritTransformerLayer(LightningModule):
    """Proposed Transformer Layer for GRIT.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/layer/grit_layer.py
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        norm: nn.Module = nn.BatchNorm1d,
        residual: bool = True,
        deg_scaler: bool = True,
        activation: nn.Module = nn.ReLU,
        norm_edges: bool = True,
        update_edges: bool = True,
        batch_norm_momentum: float = 0.1,
        batch_norm_runner: bool = True,
        rezero: bool = False,
        enable_edge_transform: bool = True,
        attn_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_clamp: float = 5.0,
        attn_activation: nn.Module = nn.ReLU,
        attn_edge_enhance: bool = True,
    ):
        """Construct 'GritTransformerLayer'.

        Args:
            in_dim: Dimension of the input tensor.
            out_dim: Dimension of theo output tensor.
            num_heads: Number of attention heads.
            dropout: Dropout layer probability.
            norm: Uninstantiated normalization layer.
                Must be either `torch.nn.BatchNorm1d` or `torch.nn.LayerNorm`.
            residual: Apply residual connections.
            deg_scaler: Apply degree scaling after MHA.
            activation: Uninstantiated activation function.
                E.g. `torch.nn.ReLU`
            norm_edges: Apply normalization to edges.
            update_edges: Update edges after layer.
            batch_norm_momentum: Momentum of batch normalization.
            batch_norm_runner: Track running stats of batch normalization.
            rezero: Apply learnable scaling parameters.
            enable_edge_transform: Apply a FC to edges at the start
                of the layer.
            attn_bias: Add bias to keys and values in MHA block.
            attn_dropout: Attention droput.
            attn_clamp: Clamp absolute value of attention scores to a value.
            attn_activation: Uninstantiated activation function for MHA block.
                E.g. `torch.nn.ReLU`
            attn_edge_enhance: Applies learnable weight matrix with node-pair
                in output node calculation in MHA block.
        """
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.update_edges = update_edges
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_runner = batch_norm_runner
        self.rezero = rezero
        self.deg_scaler = deg_scaler
        self.activation = activation()

        self.attention = GritSparseMHA(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=attn_bias,
            dropout=attn_dropout,
            clamp=attn_clamp,
            activation=attn_activation,
            edge_enhance=attn_edge_enhance,
        )

        self.fc1_x = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if enable_edge_transform:
            self.fc1_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.fc1_e = nn.Identity()

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(
                torch.zeros(1, out_dim // num_heads * num_heads, 2)
            )
            nn.init.xavier_normal_(self.deg_coef)

        if norm == nn.LayerNorm:
            self.norm1_x = norm(out_dim)
            self.norm1_e = self.norm(out_dim) if norm_edges else nn.Identity()
        elif norm == nn.BatchNorm1d:
            self.norm1_x = norm(
                out_dim,
                track_running_stats=self.batch_norm_runner,
                eps=1e-5,
                momentum=self.batch_norm_momentum,
            )
            self.norm1_e = (
                norm(
                    out_dim,
                    track_running_stats=self.batch_norm_runner,
                    eps=1e-5,
                    momentum=self.batch_norm_momentum,
                )
                if norm_edges
                else nn.Identity()
            )
        else:  # TODO: Maybe just set this to nn.Identity. -PW
            raise ValueError(
                "GritTransformerLayer normalization layer must be 'LayerNorm' \
                    or 'BatchNorm1d'!"
            )

        # FFN for x
        self.FFN_x_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_x_layer2 = nn.Linear(out_dim * 2, out_dim)

        if norm == nn.LayerNorm:
            self.norm2_x = norm(out_dim)
        elif norm == nn.BatchNorm1d:
            self.norm2_x = norm(
                out_dim,
                track_running_stats=self.batch_norm_runner,
                eps=1e-5,
                momentum=self.batch_norm_momentum,
            )

        if self.rezero:  # Learnable scaling parameters
            self.alpha1_x = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_x = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

        self.dropout1 = nn.Dropout(dropout)  # Post-attention dropout on x
        self.dropout2 = nn.Dropout(dropout)  # Post-attention dropout on e
        self.dropout3 = nn.Dropout(dropout)  # Post-FFN dropout on x

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        x = data.x
        num_nodes = data.num_nodes
        log_deg = get_log_deg(data)

        x_attn_residual = x  # for first residual connection
        e_values_in = data.get("edge_attr", None)
        e = None

        # Attention outputs
        x_attn_out, e_attn_out = self.attention(data)

        x = x_attn_out.view(num_nodes, -1)
        x = self.dropout1(x)

        # Apply degree scaler if enabled
        if self.deg_scaler:
            x = torch.stack([x, x * log_deg], dim=-1)
            x = (x * self.deg_coef).sum(dim=-1)

        x = self.fc1_x(x)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = self.dropout2(e)
            e = self.fc1_e(e)

        if self.residual:
            if self.rezero:
                x = x * self.alpha1_x
            x = x_attn_residual + x

            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_values_in

        x = self.norm1_x(x)
        if e is not None:
            e = self.norm1_e(e)

        # FFN for x
        x_ffn_residual = x  # Residual over the FFN
        x = self.FFN_x_layer1(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.FFN_x_layer2(x)

        if self.residual:
            if self.rezero:
                x = x * self.alpha2_x
            x = x_ffn_residual + x  # residual connection

        x = self.norm2_x(x)

        data.x = x
        if self.update_edges:
            data.edge_attr = e
        else:
            data.edge_attr = e_values_in

        return data


# TODO: This is a prediction head... we probably want only the graph stuff here
# and let the Tasks handle the last layer. -PW
class SANGraphHead(LightningModule):
    """SAN prediction head for graph prediction tasks.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/head/san_graph.py
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int = 1,
        L: int = 2,
        activation: nn.Module = nn.ReLU,
        pooling: str = "mean",
    ):
        """Construct `SANGraphHead`.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            L: Number of hidden layers.
            activation: Uninstantiated activation function.
                E.g. `torch.nn.ReLU`
            pooling: Node-wise pooling operation. Either "mean" or "add".
        """
        super().__init__()
        if pooling == "mean":
            self.pooling_fun = global_mean_pool
        elif pooling == "add":
            self.pooling_fun = global_add_pool
        else:
            raise RuntimeError("Currently supports only 'add' or 'mean'.")

        fc_layers = [
            nn.Linear(dim_in // 2**n, dim_in // 2 ** (n + 1), bias=True)
            for n in range(L)
        ]
        assert dim_in // 2**L >= dim_out, "Too much dim reduction!"
        fc_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.L = L
        self.activation = activation()
        self.dim_out = dim_out

    def forward(self, data: Data) -> Tensor:
        """Forward Pass."""
        graph_emb = self.pooling_fun(data.x, data.batch)
        for i in range(self.L):
            graph_emb = self.fc_layers[i](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.fc_layers[self.L](graph_emb)
        # Original code applied a final linear layer to project to dim_out,
        # but we will let the Task layer do that.
        return graph_emb
