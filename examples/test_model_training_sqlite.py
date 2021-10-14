import logging
from numpy.lib.function_base import select
import torch
import torch.utils.data
from timer import timer
from torch_geometric.data.batch import Batch
from tqdm import tqdm
from gnn_reco.models.dynedge import dynedge_angle_xfeats
from gnn_reco.components.loss_functions import VonMisesSineCosineLoss

from gnn_reco.data.sqlite_dataset import SQLiteDataset
from sklearn.model_selection import train_test_split
from gnn_reco.components.utils import Trainer
from gnn_reco.components.utils import Predictor
from gnn_reco.components.utils import PiecewiseLinearScheduler
from gnn_reco.data.utils import get_even_neutrino_indicies

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')


# Constants
FEATURES = ['event_no', 'dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area']
FEATURES_STRING = ', '.join(FEATURES)

TRUTH = ['event_no', 'energy', 'position_x', 'position_y', 'position_z', 'azimuth', 'zenith', 'pid', 'elasticity', 'sim_type', 'interaction_type']
TRUTH_STRING = ', '.join(TRUTH)

db = '/groups/icecube/leonbozi/datafromrasmus/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'
pulsemap = 'SRTTWOfflinePulsesDC'
batch_size = 1024
num_workers = 10
device = 'cuda:1'
target = 'zenith'
n_epochs = 30
patience = 5
# Common variables

selection = get_even_neutrino_indicies(db)[0:100000]

training_selection, validation_selection = train_test_split(selection, test_size=0.33, random_state=42)

training_dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= training_selection)
training_dataset.close_connection()
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                        collate_fn=Batch.from_data_list,persistent_workers=True,prefetch_factor=2)

validation_dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= validation_selection)
validation_dataset.close_connection()
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                        collate_fn=Batch.from_data_list,persistent_workers=True,prefetch_factor=2)


model = dynedge_angle_xfeats(k = 8, device = device, n_outputs= 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,eps = 1e-3)
scheduler = PiecewiseLinearScheduler(training_dataset, start_lr = 1e-5, max_lr= 1e-3, end_lr = 1e-5, max_epochs= n_epochs)
loss_func = VonMisesSineCosineLoss

trainer = Trainer(training_dataloader = training_dataloader, validation_dataloader= validation_dataloader, 
                  optimizer = optimizer, n_epochs = n_epochs, loss_func = loss_func, target = target, 
                  device = device, scheduler = scheduler, patience= patience)

trained_model = trainer(model)

predictor = Predictor(dataloader = validation_dataloader, target = target, device = device, output_column_names= ['sine', 'cosine', 'k'])

results = predictor(trained_model)

print(results)