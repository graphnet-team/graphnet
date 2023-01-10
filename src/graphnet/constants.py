"""Global constants for use across `graphnet`."""

import os.path

GRAPHNET_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TEST_DATA_DIR = os.path.abspath(os.path.join(GRAPHNET_ROOT_DIR, "test_data"))
