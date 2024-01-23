"""Classes for performing embedding of input data."""
import torch


class SinusoidalPosEmb(torch.nn.Module):
    """Sinusoidal positional embedding layer."""

    def __init__(self, dim: int = 16, M: int = 10000) -> None:
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            M: Number of frequencies.
        """
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable forward pass to the layer."""
        device = x.device
        half_dim = self.dim
        emb = torch.log(torch.tensor(self.M, device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
