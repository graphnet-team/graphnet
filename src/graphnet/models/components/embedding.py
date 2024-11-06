"""Classes for performing embedding of input data."""
import torch
import torch.nn as nn
from torch.functional import Tensor

from typing import Optional

from pytorch_lightning import LightningModule


class SinusoidalPosEmb(LightningModule):
    """Sinusoidal positional embeddings module.

    This module is from the kaggle competition 2nd place solution (see
    arXiv:2310.15674): It performs what is called Fourier encoding or it's used
    in the Attention is all you need arXiv:1706.03762. It can be seen as a soft
    digitization of the input data
    """

    def __init__(
        self,
        dim: int = 16,
        n_freq: int = 10000,
        scaled: bool = False,
    ):
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            n_freq: Number of frequencies.
            scaled: Whether or not to scale the output.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim has to be even. Got: {dim}")
        self.scale = (
            nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        )
        self.dim = dim
        self.n_freq = torch.Tensor([n_freq])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        device = x.device
        half_dim = self.dim / 2
        emb = torch.log(self.n_freq.to(device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb * self.scale


class FourierEncoder(LightningModule):
    """Fourier encoder module.

    This module incorporates sinusoidal positional embeddings and auxiliary
    embeddings to process input sequences and produce meaningful
    representations. The features x, y, z and time are mandatory, while charge
    and auxiliary are optional. Please use the mapping to ensure correct
    fourier encoding.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        mapping: list = [0, 1, 2, 3, 4, 5],
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, `mlp_dim` is set automatically as multiples of
                `seq_length` (in consistent with the 2nd place solution),
                depending on `n_features`.
            output_dim: Dimension of the output (I.e. number of columns).
            scaled: Whether or not to scale the embeddings.
            mapping: Mapping of the data to [x,y,z,time,charge,auxiliary].
                Use None for missing features.
        """
        super().__init__()
        self.mapping_str = ["x", "y", "z", "time", "charge", "auxiliary"]
        self.mapping = mapping
        self.n_features = len([i for i in mapping if i is not None])
        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        assert len(mapping) == 6, (
            "Fourier mapping must have 6 elements."
            "Use None for missing features."
        )
        assert all(
            isinstance(i, int) or i is None for i in mapping
        ), "Use int or None in fourier mapping."

        if any([i is None for i in mapping[:4]]):
            missing = [
                self.mapping_str[i] for i in range(4) if mapping[i] is None
            ]
            raise ValueError(
                f"x, y, z and time of the DOM are required."
                f"{missing} missing in mapping."
            )
        elif self.n_features == 6:
            self.aux_emb = nn.Embedding(2, seq_length // 2)
            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((self.n_features + 0.5) * seq_length)

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        mapping_max = max(i for i in self.mapping if i is not None) + 1
        if mapping_max > x.shape[2]:
            raise IndexError(
                f"Fourier mapping does not fit given data."
                f"Feature space of data is too small (size {x.shape[2]}),"
                f"given fourier mapping requires at least {mapping_max}."
            )

        length = torch.log10(seq_length.to(dtype=x.dtype))

        # Position
        embeddings = [
            self.sin_emb(4096 * x[:, :, self.mapping[:3]]).flatten(-2)
        ]

        # Charge
        if self.n_features >= 5:
            embeddings.append(self.sin_emb(1024 * x[:, :, self.mapping[4]]))

        # Time
        embeddings.append(self.sin_emb(4096 * x[:, :, self.mapping[3]]))

        # Auxiliary
        if self.n_features == 6:
            embeddings.append(self.aux_emb(x[:, :, self.mapping[5]].long()))

        # Length
        embeddings.append(
            self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1)
        )

        x = torch.cat(embeddings, -1)
        x = self.mlp(x)

        return x


class SpacetimeEncoder(LightningModule):
    """Spacetime encoder module."""

    def __init__(
        self,
        seq_length: int = 32,
    ):
        """Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events
        and generates sinusoidal positional embeddings to be added to input
        sequences.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)

    def forward(
        self,
        x: Tensor,
        # Lmax: Optional[int] = None,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:, :, :3]
        time = x[:, :, 3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(
            -1
        ) - ((time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        return rel_attn
