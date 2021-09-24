"""Minimum working example (MWE) to use SQLiteDataConverter
"""

from gnn_reco.data.sqlite_dataconverter import SQLiteDataConverter


def main():
    """Main script function."""
    paths = ['/groups/icecube/asogaard/data/i3/i3_to_sqlite_workshop_test/level7_v02.00']
    mode  = 'oscNext'
    pulsemap = 'SRTInIcePulses'
    gcd_rescue = '/groups/icecube/stuttard/data/oscNext/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
    outdir = '/groups/icecube/asogaard/temp/sqlite_test'
    db_name = 'gnn-reco_test'
    workers = 2

    SQLiteDataConverter(paths, mode, pulsemap, gcd_rescue, outdir, db_name, workers)

if __name__ == '__main__':
    main()
