"""LMDB-specific utility functions for use in `graphnet.data`."""

from typing import Optional

import lmdb


def get_serialization_method(lmdb_path: str) -> Optional[str]:
    """Retrieve the serialization method used for an LMDB database.

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
