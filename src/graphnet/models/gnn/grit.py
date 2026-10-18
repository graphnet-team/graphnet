"""Deprecated module path for `GRIT`.

`GRIT` is a graph-transformer model and now lives in
`graphnet.models.transformer`. This module re-exports it so existing imports
keep working, and warns on import that the path has moved.
"""

from warnings import warn

from graphnet.models.transformer.grit import GRIT

warn(
    "`graphnet.models.gnn.grit` is deprecated; import `GRIT` from "
    "`graphnet.models.transformer` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GRIT"]
