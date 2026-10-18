"""Deprecated module path for `DeepIce`.

`DeepIce` is a transformer model and now lives in
`graphnet.models.transformer`. This module re-exports it so existing imports
keep working, and warns on import that the path has moved.
"""

from warnings import warn

from graphnet.models.transformer.deepice import DeepIce

warn(
    "`graphnet.models.gnn.icemix` is deprecated; import `DeepIce` from "
    "`graphnet.models.transformer` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DeepIce"]
