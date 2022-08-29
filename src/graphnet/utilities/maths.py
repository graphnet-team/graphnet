"""Collection of assorted "maths-like" functions."""

import torch


def eps_like(tensor: torch.Tensor) -> torch.Tensor:
    """Return `eps` matching `tensor`'s dtype."""
    return torch.finfo(tensor.dtype).eps
