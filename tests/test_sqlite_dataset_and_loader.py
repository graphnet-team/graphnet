from gnn_reco.data.sqlite_dataset import SQLiteDataset
from gnn_reco.data.sqlite_dataloader import SQLiteDataLoader

import time

db = '/groups/hep/pcs557/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'

pulsemap = 'SRTTWOfflinePulsesDC'

selection = '/groups/hep/pcs557/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/selections/track_cascade_even_all.csv'

batch_size=1024

dataset = SQLiteDataset('asd', db, selection, pulsemap, batch_size)

#print(dir(dataset))
train_loader = SQLiteDataLoader(dataset, batch_size, num_workers = 30)


start_time = time.time()
for i in range(len(train_loader)):
    print('%s / %s'%(i, len(train_loader)))
    next(iter(train_loader))
print(time.time() - start_time)