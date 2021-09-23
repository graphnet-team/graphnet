from sqlite_dataconverter import SQLiteDataConverter
import sqlite3
import pandas as pd


paths = ['/groups/hep/pcs557/i3_workspace/data/oscNext/workshop_test/level7_v02.00']
mode  = 'oscNext'
pulsemap = 'SRTInIcePulses'
gcd_rescue = '/groups/icecube/stuttard/data/oscNext/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
outdir = '/groups/hep/pcs557/i3_workspace/data/oscNext/workshop_test'
db_name = 'gnn-reco_test'
workers = 2



SQLiteDataConverter(paths, mode, pulsemap, gcd_rescue, outdir, db_name, workers)

