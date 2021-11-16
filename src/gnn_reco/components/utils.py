import os
import torch
import pandas as pd
import sqlite3
from copy import deepcopy
import pickle
import numpy as np
from tqdm import tqdm

from torch_geometric.data.batch import Batch
from sklearn.model_selection import train_test_split

from gnn_reco.data.sqlite_dataset import SQLiteDataset
from gnn_reco.models import Model

# @TODO >>> RESOLVE DUPLICATION WRT. src/gnn_reco/models/training/{callbacks,trainers,utils}.py
def make_train_validation_dataloader(db, selection, pulsemap, batch_size, FEATURES, TRUTH, num_workers, database_indices = None, persistent_workers = True, seed = 42):
    rng = np.random.RandomState(seed=seed)
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame({'event_no':selection, 'db':database_indices})
        shuffled_df = df_for_shuffle.sample(frac = 1, random_state=rng)
        training_df, validation_df = train_test_split(shuffled_df, test_size=0.33, random_state=rng)
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
    else:
        training_selection, validation_selection = train_test_split(selection, test_size=0.33, random_state=rng)
    training_dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= training_selection)
    training_dataset.close_connection()
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                            collate_fn=Batch.from_data_list,persistent_workers=persistent_workers,prefetch_factor=2)

    validation_dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= validation_selection)
    validation_dataset.close_connection()
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                            collate_fn=Batch.from_data_list,persistent_workers=persistent_workers,prefetch_factor=2)
    return training_dataloader, validation_dataloader, {'valid_selection':validation_selection, 'training_selection':training_selection}

def make_dataloader(db, selection, pulsemap, batch_size, FEATURES, TRUTH, num_workers, database_indices = None, persistent_workers = True):
    dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= selection)
    dataset.close_connection()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                            collate_fn=Batch.from_data_list,persistent_workers=persistent_workers,prefetch_factor=2)
    return dataloader
def save_results(db, tag, results, archive,model):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    os.makedirs(path, exist_ok = True)
    results.to_csv(path + '/results.csv')
    torch.save(model.cpu().state_dict(), path + '/' + tag + '.pkl')
    print('Results saved at: \n %s'%path)
    return
# @TODO <<< RESOLVE DUPLICATION WRT. src/gnn_reco/models/training/{callbacks,trainers,utils}.py

def load_model(db, tag, archive, detector, gnn, task, device):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device=device,
    )
    model.load_state_dict(torch.load(path + '/' + tag + '.pkl'))
    model.eval()
    return model

def check_db_size(db):
    max_size = 5000000
    with sqlite3.connect(db) as con:
        query = 'select event_no from truth'
        events =  pd.read_sql(query,con)
    if len(events) > max_size:
        events = events.sample(max_size)
    return events        

def fit_scaler(db, features, truth, pulsemap):
    features = deepcopy(features)
    truth = deepcopy(truth)
    #features.remove('event_no')
    #truth.remove('event_no')
    truth =  ', '.join(truth)
    features = ', '.join(features)

    outdir = '/'.join(db.split('/')[:-2]) 
    print(os.path.exists(outdir + '/meta/transformers.pkl'))
    if os.path.exists(outdir + '/meta/transformers.pkl'):
        comb_scalers = pd.read_pickle(outdir + '/meta/transformers.pkl')
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
