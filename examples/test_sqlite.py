"""Minimum working example (MWE) to use SQLiteDataConverter
"""

from gnn_reco.data.sqlite_dataconverter import SQLiteDataConverter


def main():
    """Main script function."""
    paths = ['/groups/icecube/asogaard/data/i3/i3_to_sqlite_workshop_test/level7_v02.00']
    pulsemap = 'SRTInIcePulses'
    gcd_rescue = '/groups/icecube/stuttard/data/oscNext/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
    outdir = '/groups/icecube/asogaard/temp/sqlite_test'
    db_name = 'gnn-reco_test'
    workers = 2

    converter = SQLiteDataConverter(outdir, pulsemap, gcd_rescue, db_name=db_name, workers=workers)
    converter(paths)

if __name__ == '__main__':
    main()
