"""Contains pre-converted datasets ready for training."""

from .test_dataset import TestDataset
from .prometheus_datasets import TRIDENTSmall, BaikalGVDSmall, PONESmall
from .snowstorm_dataset import SnowStormDataset
from .hexagon_ice_le_dataset import (
    HexagonIceLEDataset,
    FEATURES_HEXAGON_ICE_LE,
    TRUTH_HEXAGON_ICE_LE,
)
