import os
import pandas as pd
import sqlite3
from copy import deepcopy


def check_db_size(db):
    max_size = 5000000
    with sqlite3.connect(db) as con:
        query = "select event_no from truth"
        events = pd.read_sql(query, con)
    if len(events) > max_size:
        events = events.sample(max_size)
    return events


def fit_scaler(db, features, truth, pulsemap):
    features = deepcopy(features)
    truth = deepcopy(truth)
    # features.remove('event_no')
    # truth.remove('event_no')
    truth = ", ".join(truth)
    features = ", ".join(features)

    outdir = "/".join(db.split("/")[:-2])
    print(os.path.exists(outdir + "/meta/transformers.pkl"))
    if os.path.exists(outdir + "/meta/transformers.pkl"):
        comb_scalers = pd.read_pickle(outdir + "/meta/transformers.pkl")
    # else:
    #     truths = ['energy', 'position_x', 'position_y', 'position_z', 'azimuth', 'zenith']
    #     events = check_db_size(db)
    #     print('Fitting to %s'%pulsemap)
    #     with sqlite3.connect(db) as con:
    #         query = 'select %s from %s where event_no in %s'%(features,pulsemap, str(tuple(events['event_no'])))
    #         feature_data = pd.read_sql(query,con)
    #         scaler = RobustScaler()
    #         feature_scaler= scaler.fit(feature_data)
    #     truth_scalers = {}
    #     for truth in truths:
    #         print('Fitting to %s'%truth)
    #         with sqlite3.connect(db) as con:
    #             query = 'select %s from truth'%truth
    #             truth_data = pd.read_sql(query,con)
    #         scaler = RobustScaler()
    #         if truth == 'energy':
    #             truth_scalers[truth] = scaler.fit(np.array(np.log10(truth_data[truth])).reshape(-1,1))
    #         else:
    #             truth_scalers[truth] = scaler.fit(np.array(truth_data[truth]).reshape(-1,1))

    #     comb_scalers = {'truth': truth_scalers, 'input': feature_scaler}
    #     os.makedirs(outdir + '/meta', exist_ok= True)
    #     with open(outdir + '/meta/transformersv2.pkl','wb') as handle:
    #         pickle.dump(comb_scalers,handle,protocol = pickle.HIGHEST_PROTOCOL)
    return comb_scalers
