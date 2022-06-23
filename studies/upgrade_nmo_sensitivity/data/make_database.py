"""Making SQLite database for neutrino mass ordering studies for IceCube-Upgrade."""

import os

from graphnet.data.extractors import (
    I3FeatureExtractorIceCubeUpgrade,
    I3TruthExtractor,
)
from graphnet.data.sqlite_dataconverter import SQLiteDataConverter


def make_database():
    """Main script function."""
    basedir = "/groups/icecube/asogaard/data/jvmead/ug_sim/genie2/genie/step4"
    paths = [
        basedir,
    ]
    gcd_rescue = os.path.join(
        basedir,
        "gcd/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2",
    )
    outdir = "/groups/icecube/asogaard/data/sqlite"
    db_name = "upgrade_genie_step4_june2022"
    workers = 20

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3FeatureExtractorIceCubeUpgrade(
                "SplitInIcePulses_GraphSage_Pulses"
            ),
        ],
        outdir,
        gcd_rescue,
        db_name=db_name,
        workers=workers,
    )
    converter(paths)


if __name__ == "__main__":
    make_database()
