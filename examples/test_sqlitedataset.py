import logging
import torch
import torch.utils.data
import time
from timer import timer
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from gnn_reco.data.sqlite_dataset import SQLiteDataset


# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')


# Constants
FEATURES = [
    'event_no', 
    'dom_x', 
    'dom_y', 
    'dom_z', 
    'dom_time', 
    'charge', 
    'rde', 
    'pmt_area',
]
FEATURES_STRING = ', '.join(FEATURES)

TRUTH = [
    'event_no', 
    'energy', 
    'position_x', 
    'position_y', 
    'position_z', 
    'azimuth', 
    'zenith', 
    'pid', 
    'elasticity', 
    'sim_type', 
    'interaction_type',
]
TRUTH_STRING = ', '.join(TRUTH)

db = '/groups/icecube/leonbozi/datafromrasmus/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'
pulsemap = 'SRTTWOfflinePulsesDC'
batch_size = 1024
num_workers = 30
wait_time = 0.00  # sec.

# Common variables
dataset = SQLiteDataset(
    db, 
    pulsemap, 
    FEATURES, 
    TRUTH, 
)

print(dataset[0])
dataset.close_connection()  # This is necessary iff `dataset` has been indexed between instantiation and passing to `DataLoader`

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers, 
    collate_fn=Batch.from_data_list,
    persistent_workers=True,
    prefetch_factor=2,
)

with timer("torch dataloader"):
    for batch in tqdm(dataloader, unit=' batches'):
        time.sleep(wait_time)
        
print(batch)
print(batch.size(), batch.num_graphs)
