"""Synthetic IC86 pixel tables for CNN grid tests (no parquet)."""

import numpy as np


def ic86_full_detector_pixel_table() -> np.ndarray:
    """Return all IceCube-86 DOM rows with redundant id columns.

    Columns are ``string``, ``dom_number``, ``redundant_string``,
    ``redundant_dom_number`` (``redundant_*`` copy ``string`` / ``dom_number``),
    in row-major order over strings ``1..86`` and DOMs ``1..60``.

    Shape is ``(5160, 4)`` with ``float32`` values, matching the tables that
    were previously loaded from packaged parquet.
    """
    s = np.repeat(np.arange(1, 87, dtype=np.float32), 60)
    d = np.tile(np.arange(1, 61, dtype=np.float32), 86)
    return np.column_stack([s, d, s.copy(), d.copy()])


IC86_TEST_PIXEL_COLUMNS = [
    "string",
    "dom_number",
    "redundant_string",
    "redundant_dom_number",
]
