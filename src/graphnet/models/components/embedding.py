"""Classes for performing embedding of input data."""
import torch


class SinusoidalPosEmb(torch.nn.Module):
    """Sinusoidal positional embedding layer.

    This module is from the kaggle competition 2nd place solution (see
    arXiv:2310.15674): It performs what is called Fourier encoding or it's used
    in the Attention is all you need arXiv:1706.03762. It can be seen as a soft
    digitization of the input data
    """

    def __init__(self, dim: int = 16, m: int = 10000) -> None:
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            m: Number of frequencies.
        """
        super().__init__()
        self.dim = dim
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable forward pass to the layer."""
        device = x.device
        half_dim = self.dim
        emb = torch.log(torch.tensor(self.m, device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
