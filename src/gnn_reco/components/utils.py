import os
import pandas as pd
import sqlite3
import torch

def save_results(db, tag, results, archive,model):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    os.makedirs(path, exist_ok = True)
    results.to_csv(path + '/results.csv')
    torch.save(model.cpu(), path + '/' + tag + '.pkl')
    print('Results saved at: \n %s'%path)
    return

def check_db_size(db):
    max_size = 5000000
    with sqlite3.connect(db) as con:
        query = 'select event_no from truth'
        events =  pd.read_sql(query,con)
    if len(events) > max_size:
        events = events.sample(max_size)
    return events        
