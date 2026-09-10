"""Layers for the GRIT graph-transformer architecture."""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import softmax
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
