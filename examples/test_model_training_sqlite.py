import logging
import torch
import torch.utils.data
from timer import timer
from gnn_reco.models.dynedge import Dynedge
from gnn_reco.models.convnet import ConvNet
from gnn_reco.components.loss_functions import vonmises_sinecosine_loss
from gnn_reco.components.loss_functions import log_cosh
from gnn_reco.components.utils import Trainer
from gnn_reco.components.utils import Predictor
from gnn_reco.components.utils import PiecewiseLinearScheduler
from gnn_reco.data.utils import get_even_neutrino_indicies
from gnn_reco.components.utils import make_train_validation_dataloader
from gnn_reco.components.utils import save_results
from gnn_reco.components.utils import fit_scaler
# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
FEATURES = ['event_no', 'dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area']
FEATURES_STRING = ', '.join(FEATURES)

TRUTH = ['event_no', 'energy', 'position_x', 'position_y', 'position_z', 'azimuth', 'zenith', 'pid', 'elasticity', 'sim_type', 'interaction_type']
TRUTH_STRING = ', '.join(TRUTH)

db = '/groups/hep/pcs557/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'
pulsemap = 'SRTTWOfflinePulsesDC'
batch_size = 1024
num_workers = 15
device = 'cuda:0'
target = 'energy'
n_epochs = 30
patience = 5
archive = '/groups/hep/pcs557/phd/results'
scalers = fit_scaler(db, FEATURES, TRUTH, pulsemap)
# Common variables

selection = get_even_neutrino_indicies(db)[0:100000]

training_dataloader, validation_dataloader = make_train_validation_dataloader(db, selection, pulsemap, batch_size, FEATURES, TRUTH, num_workers)

#model = ConvNet(n_features = 7, n_labels = 1, knn_cols = [0,1,2,3], scalers = scalers,target = target, device = device).to(device)
model = Dynedge(k = 8, device = device, n_outputs= 1, scalers = scalers, target = target).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,eps = 1e-3)
scheduler = PiecewiseLinearScheduler(training_dataloader, start_lr = 1e-5, max_lr= 1e-3, end_lr = 1e-5, max_epochs= n_epochs)
loss_func = log_cosh

trainer = Trainer(training_dataloader = training_dataloader, validation_dataloader= validation_dataloader, 
                  optimizer = optimizer, n_epochs = n_epochs, loss_func = loss_func, target = target, 
                  device = device, scheduler = scheduler, patience= patience)

trained_model = trainer(model)

predictor = Predictor(dataloader = validation_dataloader, target = target, device = device, output_column_names= [target + '_pred'])
results = predictor(trained_model)
save_results(db, 'dynedge_energy', results,archive, trained_model)


