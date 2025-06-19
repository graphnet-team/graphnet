"""Global constants for use across `graphnet`."""

import os.path

GRAPHNET_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Data directory
DATA_DIR = os.path.join(GRAPHNET_ROOT_DIR, "data")

# Test data
TEST_DATA_DIR = os.path.join(DATA_DIR, "tests")
TEST_OUTPUT_DIR = os.path.join(TEST_DATA_DIR, "output")

_test_dataset_name = "oscNext_genie_level7_v02"
_test_dataset_file = f"{_test_dataset_name}_first_5_frames"
TEST_SQLITE_DATA = os.path.join(
    TEST_DATA_DIR, "sqlite", _test_dataset_name, f"{_test_dataset_file}.db"
)
TEST_PARQUET_DATA = os.path.join(
    TEST_DATA_DIR, "parquet", _test_dataset_name, "merged"
)
TEST_IMAGE_DIR = os.path.join(TEST_DATA_DIR, "images")
TEST_IC86MAIN_IMAGE = os.path.join(TEST_IMAGE_DIR, "IC86main_array_test.npy")
TEST_IC86LOWERDC_IMAGE = os.path.join(
    TEST_IMAGE_DIR, "IC86lower_deepcore_test.npy"
)
TEST_IC86UPPERDC_IMAGE = os.path.join(
    TEST_IMAGE_DIR, "IC86upper_deepcore_test.npy"
)

# Example data
EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "examples")
EXAMPLE_OUTPUT_DIR = os.path.join(EXAMPLE_DATA_DIR, "output")

# Configuration files
CONFIG_DIR = os.path.join(GRAPHNET_ROOT_DIR, "configs")
DATASETS_CONFIG_DIR = os.path.join(CONFIG_DIR, "datasets")
MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "models")

# Pretrained models /icecube/upgrade/QUESO
PRETRAINED_MODEL_DIR = os.path.join(
    GRAPHNET_ROOT_DIR, "src", "graphnet", "models", "pretrained"
)

# Geometry Tables
GEOMETRY_TABLE_DIR = os.path.join(DATA_DIR, "geometry_tables")
ICECUBE_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "icecube")
PROMETHEUS_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "prometheus")
LIQUIDO_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "liquid-o")
MAGIC_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "magic")

# Image Mapping Tables
IMAGE_MAPPING_TABLE_DIR = os.path.join(DATA_DIR, "image_mapping_tables")
IC86_CNN_MAPPING = os.path.join(
    IMAGE_MAPPING_TABLE_DIR, "IC86_CNN_mapping.parquet"
)
