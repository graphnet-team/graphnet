"""Classes for performing embedding of input data."""
import torch
import torch.nn as nn
from torch.functional import Tensor

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
    representations.
    """

    def __init__(
        self,
        seq_length: int = 128,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            output_dim: Output dimensionality of the final projection.
            scaled: Whether or not to scale the embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.aux_emb = nn.Embedding(2, seq_length // 2)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)
        hidden_dim = 6 * seq_length if n_features >= 5 else int(5.5 * seq_length)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(seq_length.to(dtype=x.dtype))
        b = [
            self.sin_emb(4096 * x[:, :, :3]).flatten(-2),  # Position
            self.sin_emb(1024 * x[:, :, 4]),  # Charge
            self.sin_emb(4096 * x[:, :, 3]),  # Time
        ]

        if self.n_features >= 5:
            b.append(self.aux_emb(x[:, :, 5].long()))  # Auxiliary
        elif self.n_features < 4:
            raise ValueError(f"At least x_dom, y_dom, z_dom, charge and "
                             f"dom_time are required. Got only "
                             f"{self.n_features} features.") 
        b.append(self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1))  # Length
        
        x = torch.cat(b, -1)
        x = self.projection(x)
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
