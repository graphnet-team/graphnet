"""LMDB-specific utility functions for use in `graphnet.data`."""

from typing import Any, Callable, Optional

import lmdb


def _resolve_deserializer(
    serialization_method: str,
) -> Optional[Callable[[bytes], Any]]:
    """Resolve deserialization callable from serialization method name.

    Args:
        serialization_method: Name of the serialization method
            (e.g., "pickle", "json", "msgpack", "dill").

    Returns:
        Deserialization callable that takes bytes and returns the deserialized
        object, or None if the method is not supported or is custom.
    """
    if serialization_method == "pickle":
        import pickle

        return pickle.loads
    if serialization_method == "json":
        import json

        return lambda data: json.loads(data.decode("utf-8"))
    if serialization_method == "msgpack":
        try:
            import msgpack  # type: ignore

            return lambda data: msgpack.unpackb(data, raw=False)
        except ImportError:
            raise ImportError("msgpack is not installed.")
    if serialization_method == "dill":
        try:
            import dill  # type: ignore

            return dill.loads
        except ImportError:
            raise ImportError("dill is not installed.")
    # For "__custom__" or unknown methods, return None
    return None


def get_serialization_method_name(lmdb_path: str) -> Optional[str]:
    """Retrieve the serialization method name for an LMDB database.

    Args:
        lmdb_path: Path to the LMDB database directory.

    Returns:
        The serialization method name
        (e.g., "pickle", "json", "msgpack", "dill")
        or None if the metadata is not found or the database cannot be opened.
    """
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=True)
        with env.begin(write=False) as txn:
            metadata_key = b"__meta_serialization__"
            metadata_value = txn.get(metadata_key)
            if metadata_value is not None:
                return metadata_value.decode("utf-8")
        env.close()
    except Exception:
        pass
    return None


def get_serialization_method(
    lmdb_path: str,
) -> Optional[Callable[[bytes], Any]]:
    """Retrieve the deserialization callable for an LMDB database.

    Args:
        lmdb_path: Path to the LMDB database directory.

    Returns:
        Deserialization callable that takes bytes and returns the deserialized
        object, or None if the metadata is not found, the database cannot be
        opened, or the serialization method is custom/unsupported.
    """
    method_name = get_serialization_method_name(lmdb_path)
    if method_name is not None:
        return _resolve_deserializer(method_name)
    return None
