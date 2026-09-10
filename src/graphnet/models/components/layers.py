"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import (
    Any,
    Callable,
    cast,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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
from torch_geometric.utils import degree

from graphnet.models.utils import flex_attention


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

    def forward(
        self,
        x: Tensor,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor. On the padded path this is `[B, ...]` and one
                decision is drawn per row of the batch dimension.
            doc_id: Optional `[N]` event index per token, for a packed
                `[1, N, D]` input where the batch dimension no longer
                separates events. One decision is then drawn per event and
                gathered to its tokens, matching the per-event granularity of
                the padded path.
            num_docs: Number of distinct events in `doc_id`. Required when
                `doc_id` is given.

        Returns:
            Tensor of the same shape as `x`.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        if doc_id is None:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0:
                random_tensor.div_(keep_prob)
            return x * random_tensor

        assert num_docs is not None, "`num_docs` is required with `doc_id`"
        event_mask = x.new_empty((num_docs, 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            event_mask = event_mask.div(keep_prob)
        token_mask = event_mask.index_select(0, doc_id).unsqueeze(0)
        return x * token_mask

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
        else:
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
        log_deg = torch.log10(
            degree(data.edge_index[0], num_nodes=num_nodes, dtype=data.x.dtype)
            + 1
        )
        log_deg = log_deg.view(data.num_nodes, 1)

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


class RMSNorm(LightningModule):
    """Root-mean-square layer normalization.

    Equivalent to `torch.nn.RMSNorm`, but the weight is cast to the input
    dtype before the call. The standard-library module keeps its weight in
    float32, which forces an unfused path under autocast; casting lets the
    fused bfloat16/float16 kernel fire instead.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Construct `RMSNorm`.

        Args:
            dim: Size of the normalized (last) dimension.
            eps: Term added to the denominator for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize `x` over its last dimension."""
        return torch.nn.functional.rms_norm(
            x, self.normalized_shape, self.weight.to(x.dtype), self.eps
        )


class SwiGLU(LightningModule):
    """Feed-forward block with a SwiGLU gate.

    The two input projections are fused into a single `Linear`, which is
    both faster and keeps the parameter count identical to the unfused form.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Construct `SwiGLU`.

        Args:
            dim: Input and output dimension.
            hidden_dim: Inner dimension of the gated projection.
            dropout: Dropout applied to the block output.
            bias: Whether the projections carry a bias.
        """
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Xavier-uniform weights and zero biases."""
        for module in (self.w13, self.w2):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward transform."""
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.dropout(self.w2(torch.nn.functional.silu(a) * b))


class RoPE4D(LightningModule):
    """Rotary position embedding over four coordinates `(x, y, z, t)`.

    Implements the standard axis-aligned 4D construction: each coordinate
    axis is given its own rotation planes, so the four generators stay
    linearly independent. Planes are allocated one per axis first and then
    round-robin, which is what guarantees that independence for any even
    `dim >= 8`.

    Pair `j` rotates dimensions `(j, dim / 2 + j)` via a contiguous
    `chunk`/`cat` on the last axis; each side of the rotation is then a
    contiguous half, which is markedly faster than a strided
    `(2j, 2j + 1)` layout.

    See https://arxiv.org/abs/2504.06308.
    """

    freqs: Tensor
    coord_select: Tensor

    def __init__(
        self,
        dim: int,
        scales: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        base: int = 10000,
    ):
        """Construct `RoPE4D`.

        Args:
            dim: Head dimension. Must be even and at least 8.
            scales: Per-axis frequency scale for `(x, y, z, t)`, in radians
                per coordinate unit.
            base: Ratio between the lowest and highest frequency in each
                axis' band, i.e. frequencies span `scale / base` to `scale`.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE4D dim must be even (got {dim})")
        if dim < 8:
            raise ValueError(
                f"RoPE4D requires dim >= 8 for 4 axes (got {dim})"
            )
        if len(scales) != 4:
            raise ValueError(f"RoPE4D expects 4 scales (got {len(scales)})")
        self.dim = dim
        self.scales = tuple(scales)
        self.base = base

        num_planes = dim // 2

        # One plane per axis first (ensures linear independence), then
        # distribute what is left round-robin.
        allocation = [1, 1, 1, 1]
        for i in range(num_planes - 4):
            allocation[i % 4] += 1

        all_freqs = [
            self._build_freqs(n_planes, scale)
            for n_planes, scale in zip(allocation, self.scales)
        ]
        self.register_buffer("freqs", torch.cat(all_freqs))  # (dim / 2,)

        coord_select: List[int] = []
        for axis_idx, n_planes in enumerate(allocation):
            coord_select.extend([axis_idx] * n_planes)
        self.register_buffer(
            "coord_select", torch.tensor(coord_select, dtype=torch.long)
        )

    def _build_freqs(self, num_bands: int, scale: float) -> Tensor:
        """Build log-spaced frequency bands for one axis."""
        if num_bands == 1:
            return torch.tensor([1.0 / self.base]) * scale
        exponents = torch.arange(num_bands, dtype=torch.float32) / (
            num_bands - 1
        )
        freqs = (1.0 / self.base) * (self.base**exponents)
        return freqs * scale

    def compute_tables(
        self, coords: Tensor, dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tensor, Tensor]:
        """Precompute the `(cos, sin)` rotation tables for `coords`.

        Every layer of an encoder stack sees the same coordinates and the
        same RoPE configuration, so the tables are computed once and shared.

        Args:
            coords: `[B, S, 4]` coordinates.
            dtype: Optional dtype to cast the tables to. Trig is always
                evaluated in float32 for accuracy; casting once here lets the
                per-layer rotation stay in the model dtype.

        Returns:
            A `(cos, sin)` pair, each of shape `[B, 1, S, dim / 2]`.
        """
        coords_f = coords if coords.dtype == torch.float32 else coords.float()
        coord_per_plane = coords_f.index_select(
            dim=-1, index=self.coord_select
        )
        angles = (coord_per_plane * self.freqs).unsqueeze(1)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        if dtype is not None and dtype != cos_a.dtype:
            cos_a = cos_a.to(dtype)
            sin_a = sin_a.to(dtype)
        return cos_a, sin_a

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        tables: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Rotate queries or keys according to their coordinates.

        The rotation is `(a, b) -> (a cos t - b sin t, a sin t + b cos t)`
        for pair `j = (j, dim / 2 + j)`.

        Args:
            x: `[B, H, S, dim]` queries or keys.
            coords: `[B, S, 4]` coordinates `(x, y, z, t)`.
            tables: Optional precomputed `(cos, sin)` pair from
                :meth:`compute_tables`, avoiding recomputation of the
                transcendentals for every layer and for both Q and K.

        Returns:
            The rotated tensor, shaped like `x`.
        """
        if tables is None:
            tables = self.compute_tables(coords, dtype=x.dtype)
        cos_a, sin_a = tables

        # Fast path: same-dtype tables, so stay entirely in x's dtype.
        if cos_a.dtype == x.dtype:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat(
                (x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a), dim=-1
            )

        # Mixed-dtype fallback: do the maths in float32 and cast back.
        x_f = x if x.dtype == torch.float32 else x.float()
        x1, x2 = x_f.chunk(2, dim=-1)
        out = torch.cat(
            (x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a), dim=-1
        )
        return out if out.dtype == x.dtype else out.to(x.dtype)


class AttentionPool(LightningModule):
    """Pool a sequence to one vector, attending from a learned query."""

    def __init__(self, dim: int):
        """Construct `AttentionPool`.

        Args:
            dim: Feature dimension of the sequence and of the output.
        """
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * (dim**-0.5))
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pool `x` to a single vector per batch element.

        Args:
            x: `[B, S, D]` sequence.
            mask: Optional bool `[B, S]`; True marks a valid element.

        Returns:
            `[B, D]` pooled features, zero for fully-masked rows.
        """
        batch_size, _, n_features = x.shape
        q = self.q.expand(batch_size, -1, -1)
        k, v = self.kv(x).chunk(2, dim=-1)

        attn = torch.bmm(q, k.transpose(1, 2)) * (n_features**-0.5)
        if mask is not None:
            # A finite fill rather than -inf: a fully-masked row (a zero-hit
            # event) must soft-max to finite garbage, which is zeroed below.
            # -inf would yield NaN and poison the whole batch.
            attn = attn.masked_fill(
                ~mask.unsqueeze(1), torch.finfo(attn.dtype).min
            )
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).squeeze(1)
        out = self.proj(out)
        if mask is not None:
            out = out * mask.any(dim=1, keepdim=True).to(out.dtype)
        return out


class NeptuneTransformerEncoderLayer(LightningModule):
    """Pre-norm transformer encoder layer with 4D rotary attention.

    Used by :class:`~graphnet.models.transformer.neptune.Neptune`. Compared
    with a vanilla encoder layer it uses RMSNorm instead of LayerNorm, a
    fused QKV projection, per-head query/key normalization before the rotary
    embedding, LayerScale on both residual branches, and a SwiGLU
    feed-forward network.

    Attention-matrix dropout is deliberately unused: `flex_attention` cannot
    express it, so omitting it keeps the packed and padded attention paths
    regularized identically. Residual/feed-forward dropout and stochastic
    depth still apply.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        ff_bias: bool = False,
        qk_norm: bool = True,
        rope_scales: Sequence[float] = (180.0, 180.0, 180.0, 40.0),
        rope_base: int = 60,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
    ):
        """Construct `NeptuneTransformerEncoderLayer`.

        Args:
            d_model: Token dimension.
            nhead: Number of attention heads. `d_model / nhead` must be even
                and at least 8, as required by :class:`RoPE4D`.
            dim_feedforward: Inner dimension of the SwiGLU network.
            dropout: Dropout on the residual and feed-forward branches.
            layer_norm_eps: Epsilon of the RMSNorm layers.
            bias: Whether the attention projections carry a bias.
            ff_bias: Whether the feed-forward projections carry a bias.
            qk_norm: Whether to RMS-normalize queries and keys per head
                before applying the rotary embedding.
            rope_scales: Per-axis rotary frequency scales for `(x, y, z, t)`.
            rope_base: Rotary frequency span, see :class:`RoPE4D`.
            drop_path_rate: Stochastic-depth rate for this layer.
            layerscale_init: Initial value of the LayerScale parameters.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE4D (got {self.head_dim})"
            )

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._initialize_weights()

        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)

        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=ff_bias)
        self.rope = RoPE4D(
            dim=self.head_dim, scales=rope_scales, base=rope_base
        )

        # LayerScale: learnable per-channel gain on each residual branch.
        self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def _initialize_weights(self) -> None:
        """Apply Xavier-uniform weights and zero biases."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        block_mask: Any,
    ) -> Tensor:
        """Dispatch to packed block-diagonal flex attention or padded SDPA."""
        if block_mask is not None:
            return flex_attention(q, k, v, block_mask=block_mask)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    def prepare_attention_mask(
        self, key_padding_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """Convert a key-padding mask to the boolean SDPA convention.

        Args:
            key_padding_mask: Bool `[B, S]` where True marks *padding*, as in
                `torch.nn.MultiheadAttention`.
            device: Device to build the mask on.

        Returns:
            Bool `[B, 1, 1, S]` where True marks positions that *may* be
            attended, or None if no mask was given.
        """
        if key_padding_mask is None:
            return None
        allow = ~key_padding_mask.to(torch.bool).to(device)
        # A fully-padded row (a zero-hit event) would soft-max over all -inf
        # and NaN-poison the batch; let it attend everywhere instead. Pooling
        # zeroes those rows afterwards, matching the packed path.
        allow = allow | ~allow.any(dim=-1, keepdim=True)
        return allow.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        rope_tables: Optional[Tuple[Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Apply one encoder layer.

        Args:
            src: `[B, S, d_model]` token features.
            centroids: `[B, S, 4]` token coordinates driving the rotary
                embedding.
            src_key_padding_mask: Bool `[B, S]` where True marks padding.
                Only used when neither `attn_mask` nor `block_mask` is given.
            rope_tables: Optional shared `(cos, sin)` tables.
            attn_mask: Optional prebuilt boolean SDPA mask.
            block_mask: Optional `flex_attention` `BlockMask` selecting the
                packed path.
            doc_id: Optional `[N]` event index per token, on the packed path.
            num_docs: Number of events in `doc_id`.

        Returns:
            `[B, S, d_model]` updated token features.
        """
        batch_size, seq_length, _ = src.shape

        x = src
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, centroids, tables=rope_tables)
        k = self.rope(k, centroids, tables=rope_tables)

        # Use the mask prebuilt by the encoder when available; otherwise
        # build one here for single-layer callers. Skipped entirely on the
        # packed path, which masks via `block_mask`.
        if attn_mask is None and block_mask is None:
            attn_mask = self.prepare_attention_mask(
                src_key_padding_mask, q.device
            )

        attn_output = self._attn(q, k, v, attn_mask, block_mask)
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)
        attn_output = attn_output.contiguous().view(
            batch_size, seq_length, self.d_model
        )
        attn_output = self.out_proj(attn_output)

        x = x + self.drop_path1(
            self.dropout(self.gamma_1 * attn_output), doc_id, num_docs
        )

        x_norm = self.norm2(x)
        ff_output = self.ffn(x_norm)
        x = x + self.drop_path2(self.gamma_2 * ff_output, doc_id, num_docs)

        return x


class NeptuneTransformerEncoder(LightningModule):
    """Stack of :class:`NeptuneTransformerEncoderLayer`.

    Layers are built by a factory so each depth gets its own stochastic-depth
    rate, ramped linearly from 0 to `drop_path_rate`. Because every layer
    shares the same coordinates and rotary configuration, the `(cos, sin)`
    tables and the attention mask are built once here and threaded through
    the stack.
    """

    def __init__(
        self,
        layer_factory: Callable[[float], NeptuneTransformerEncoderLayer],
        num_layers: int,
        norm: Optional[LightningModule] = None,
        drop_path_rate: float = 0.0,
    ):
        """Construct `NeptuneTransformerEncoder`.

        Args:
            layer_factory: Callable mapping a stochastic-depth rate to a
                fresh :class:`NeptuneTransformerEncoderLayer`.
            num_layers: Number of layers in the stack.
            norm: Optional module applied to the stack output.
            drop_path_rate: Stochastic-depth rate of the final layer; earlier
                layers are scaled linearly towards zero.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_layers == 1:
            drop_rates = [drop_path_rate]
        else:
            drop_rates = [
                drop_path_rate * float(i) / (num_layers - 1)
                for i in range(num_layers)
            ]
        self.layers = nn.ModuleList(
            [layer_factory(rate) for rate in drop_rates]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Run the encoder stack.

        Args:
            src: `[B, S, d_model]` token features.
            centroids: `[B, S, 4]` token coordinates.
            src_key_padding_mask: Bool `[B, S]` where True marks padding.
            block_mask: Optional `flex_attention` `BlockMask` for the packed
                path.
            doc_id: Optional `[N]` event index per token, on the packed path.
            num_docs: Number of events in `doc_id`.

        Returns:
            `[B, S, d_model]` encoded tokens.
        """
        first = cast(NeptuneTransformerEncoderLayer, self.layers[0])
        # Pre-cast the tables to src.dtype so the per-layer rotation skips a
        # float32 round-trip on bfloat16/float16 paths.
        rope_tables = first.rope.compute_tables(centroids, dtype=src.dtype)
        attn_mask = (
            first.prepare_attention_mask(src_key_padding_mask, src.device)
            if block_mask is None
            else None
        )

        output = src
        for layer in self.layers:
            output = layer(
                output,
                centroids,
                rope_tables=rope_tables,
                attn_mask=attn_mask,
                block_mask=block_mask,
                doc_id=doc_id,
                num_docs=num_docs,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
