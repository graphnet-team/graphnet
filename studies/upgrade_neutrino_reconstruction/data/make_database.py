"""Minimum working example (MWE) to use SQLiteDataConverter
"""

from graphnet.data.i3extractor import (
    I3FeatureExtractorIceCubeUpgrade,
    I3TruthExtractor,
)
from graphnet.data.sqlite_dataconverter import SQLiteDataConverter


def make_database():
    """Main script function."""
    paths = [
        "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4/step4"
    ]
    gcd_rescue = "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4/gcd/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
    outdir = "/groups/hep/pcs557/GNNReco/data/databases"
    db_name = "dev_step4_preselection_december"
    workers = 45

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3FeatureExtractorIceCubeUpgrade("SplitInIcePulsesCleaned"),
            I3FeatureExtractorIceCubeUpgrade("SplitInIcePulsesSRT"),
            I3FeatureExtractorIceCubeUpgrade("SplitInIcePulsesTWSRT"),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_DEgg"
            ),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_mDOM"
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
