
"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union, List, Tuple

import torch
from torch.functional import Tensor
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_batch
from pytorch_lightning import LightningModule
from timm.models.layers import drop_path
import math

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
    """DropPath regularization module for neural networks."""
    def __init__(
        self, 
        drop_prob: Optional[float] = None,
    ):
        """
        Construct `DropPath`.

        Args:
            drop_prob: Probability of dropping a path during training.
                If None, no paths are dropped. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """Return extra representation of the module."""
        return "p={}".format(self.drop_prob)


class Mlp(LightningModule):
    """
    Multi-Layer Perceptron (MLP) module.
    """
    def __init__(
        self,
        in_features: int = None,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: Optional[nn.Module] = nn.GELU,
        dropout_prob: Optional[float] = 0.0,
    ):
        """
        Construct `Mlp`.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features={in_features} instead"
            )
        super().__init__()
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

class SinusoidalPosEmb(LightningModule):
    def __init__(
        self, 
        dim: int = 16, 
        n_freq: int = 10000,
        scaled: bool = False,
    ):
        """
        Construct `SinusoidalPosEmb`.

        This module generates sinusoidal positional embeddings to be 
        added to input sequences.

        Args:
            dim: Embedding dimension.
            n_freq: Number of frequencies.
            scaled: Whether or not to scale the embeddings.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        self.dim = dim
        self.n_freq = n_freq

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.n_freq) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale

class FourierEncoder(LightningModule):
    def __init__(
        self, 
        base_dim: int = 128, 
        output_dim: int = 384,
        scaled: bool = False,
    ):
        """
        Construct `FourierEncoder`.

        This module incorporates sinusoidal positional embeddings and auxiliary embeddings
        to process input sequences and produce meaningful representations.

        Args:
            base_dim: Dimensionality of the base sinusoidal positional embeddings.
            output_dim: Output dimensionality of the final projection.
            scaled: Whether or not to scale the embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=base_dim, scaled=scaled)
        self.aux_emb = nn.Embedding(2, base_dim // 2)
        self.sin_emb2 = SinusoidalPosEmb(dim=base_dim // 2, scaled=scaled)
        self.projection = nn.Sequential(
            nn.Linear(6 * base_dim, 6 * base_dim),
            nn.LayerNorm(6 * base_dim),
            nn.GELU(),
            nn.Linear(6 * base_dim, output_dim),
        )

    def forward(
        self, 
        x: Tensor, 
        n_pulses: Tensor,
        #Lmax: Optional[int] = None
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(n_pulses.to(dtype=x.dtype))
        x = torch.cat(
            [
                self.sin_emb(4096 * x[:,:,:3]).flatten(-2), #pos
                self.sin_emb(1024 * x[:,:,4]),              #charge
                self.sin_emb(4096 * x[:,:,3]),              #time
                self.aux_emb(x[:,:,5].long()),                     #auxiliary
                self.sin_emb2(length).unsqueeze(1).expand(-1, max(n_pulses), -1),
            ],
            -1,
        )
        x = self.projection(x)
        return x


class SpacetimeEncoder(LightningModule):
    def __init__(
        self, 
        base_dim: int = 32,
    ):
        """
        Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events and
        generates sinusoidal positional embeddings to be added to input sequences.

        Args:
            base_dim: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=base_dim)
        self.projection = nn.Linear(base_dim, base_dim)

    def forward(
        self,
        x: Tensor,
        #Lmax: Optional[int] = None,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:,:,:3]
        time = x[:,:,3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1) - (
            (time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)
        ).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(torch.abs(spacetime_interval))
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        return rel_attn, sin_emb

# BEiTv2 block
class Block_rel(LightningModule):
    """Implementation of BEiTv2 Block.
    """
    def __init__(
        self,
        dim: int = None,
        num_heads: int = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_head_dim: int = None,
    ):     
        """
        Construct 'Block_rel'.

        Args:
            dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel` layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to the
                input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key, and 
                value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key matrices
                in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_dropt: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the output 
                of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2` 
                parameters if not `None`.
            act_layer: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the 
                `Attention_rel` layer.   
        """
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_rel(
            dim, num_heads, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_head_dim=attn_head_dim
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: Tensor, key_padding_mask=None, rel_pos_bias=None, kv=None):
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
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int = None,
    ):
        """
        Args:
            dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            qk_scale: a scaling factor that multiplies the dot product of query 
                and key vectors. Defaults to None. If None, computed as
                :math: `\sqrt{1/head_dim}`
            attn_drop: the dropout probability for the attention weights.
                Defaults to 0.0.
            proj_drop: the dropout probability for the output of the attention
                module. Defaults to 0.0.
            attn_head_dim: the feature dimensionality of each attention head.
                Defaults to None. If None, computed as `dim // num_heads`.
        """
        if dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got dim={dim} and num_heads={num_heads} instead"
            )
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, rel_pos_bias=None, key_padding_mask=None):
        """Forward pass."""
        B, N, C = q.shape

        q = linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(B, k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(B, v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

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
            bias = torch.min(key_padding_mask[:, None, :], key_padding_mask[:, :, None])
            bias[
                torch.max(key_padding_mask[:, None, :], key_padding_mask[:, :, None])
                < 0
            ] = 0
            # print(bias.shape,bias.min(),bias.max())
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(LightningModule):
    def __init__(
        self,
        dim: int = None,
        num_heads: int = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Construct 'Block'.

        Args:
            dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `MultiheadAttention`
                layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to the
                input size in the `Mlp` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_dropt: Dropout probability to use in the `MultiheadAttention` layer.
            drop_path: Probability of applying drop path regularization to the output 
                of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2` 
                parameters if not `None`.
            act_layer: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the 
                `MultiheadAttention` layer.   
        """     
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
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