"""Layers for the GRIT graph-transformer architecture."""

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter
from pytorch_lightning import LightningModule


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
