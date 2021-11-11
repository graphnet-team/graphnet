
from torch import Tensor, finfo

def eps_like(x: Tensor) -> Tensor:
    """Return `eps` matching `x`'s dtype."""
    return finfo(x.dtype).eps
    