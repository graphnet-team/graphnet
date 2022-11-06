"""Example of comparing the result of converting to SQLite and Parquet."""

import logging
import os

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite import (
    SQLiteDataConverter,
    SQLiteDataset,
)
from graphnet.data.parquet import ParquetDataConverter, ParquetDataset
from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3RetroExtractor,
)
from graphnet.utilities.logging import get_logger

logger = get_logger(level=logging.INFO)

TEST_DATA_DIR = os.path.abspath("./test_data/")
PULSEMAP = "SRTInIcePulses"


def convert_data() -> None:
    """Convert I3 files to SQLite and Parquet."""
    # Configuration
    paths = TEST_DATA_DIR
    gcd_rescue = os.path.join(
        TEST_DATA_DIR,
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
    )
    opt = dict(
        extractors=[
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86(PULSEMAP),
        ],
        outdir=TEST_DATA_DIR,
        gcd_rescue=gcd_rescue,
        workers=10,
    )

    # Run data converters.
    SQLiteDataConverter(**opt)(paths)  # type: ignore[arg-type]
    ParquetDataConverter(**opt)(paths)  # type: ignore[arg-type]


def load_data() -> None:
    """Load converted data and compare.."""
    filename = "oscNext_genie_level7_v03.01_pass2.160000.000001"

    opt = dict(
        pulsemaps=PULSEMAP,
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
    )

    data_sqlite = SQLiteDataset(
        os.path.join(TEST_DATA_DIR, filename + ".db"),
        **opt,  # type: ignore[arg-type]
    )
    data_parquet = ParquetDataset(
        os.path.join(TEST_DATA_DIR, filename + ".parquet"),
        **opt,  # type: ignore[arg-type]
    )

    logger.info(len(data_sqlite))
    logger.info(len(data_parquet))

    print(data_sqlite[0])
    print(data_parquet[0])

    print(data_sqlite[0].x == data_parquet[0].x)


if __name__ == "__main__":
    convert_data()
    load_data()
