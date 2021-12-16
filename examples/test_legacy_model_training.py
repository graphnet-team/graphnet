import logging
from timer import timer
import torch
import torch.utils.data

from graphnet.components.utils import fit_scaler
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices
from graphnet.legacy.original import (
    Dynedge, 
    ConvNet,
    vonmises_sinecosine_loss,
    log_cosh,
    Trainer,
    Predictor,
)
from graphnet.models.training.callbacks import PiecewiseLinearScheduler
from graphnet.models.training.utils import make_train_validation_dataloader, save_results

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86

# Main function definition
def main():

    # Configuration
    db = '/groups/icecube/leonbozi/datafromrasmus/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'
    pulsemap = 'SRTTWOfflinePulsesDC'
    batch_size = 1024
    num_workers = 15
    device = 'cuda:1'
    target = 'zenith'
    n_epochs = 30
    patience = 5
    archive = '/groups/icecube/asogaard/gnn/results/legacy/original'
    scalers = fit_scaler(db, features, truth, pulsemap)

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(db)
    train_selection = train_selection[0:100000]
    
    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db, 
        train_selection,
        pulsemap,
        features,
        truth,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    #model = ConvNet(n_features = 7, n_labels = 1, knn_cols = [0,1,2,3], scalers = scalers,target = target, device = device).to(device)
    model = Dynedge(k = 8, device = device, n_outputs= 3, scalers = scalers, target = target).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,eps = 1e-3)
    scheduler = PiecewiseLinearScheduler(training_dataloader, start_lr = 1e-5, max_lr= 1e-3, end_lr = 1e-5, max_epochs= n_epochs)
    loss_func = vonmises_sinecosine_loss

    trainer = Trainer(training_dataloader = training_dataloader, validation_dataloader= validation_dataloader, 
                    optimizer = optimizer, n_epochs = n_epochs, loss_func = loss_func, target = target, 
                    device = device, scheduler = scheduler, patience= patience)

    trained_model = trainer(model)

    predictor = Predictor(dataloader = validation_dataloader, target = target, device = device, output_column_names= [target + '_pred', target + '_var'])
    results = predictor(trained_model)
    save_results(db, 'dynedge_zenith_legacy_original', results, archive, trained_model)

# Main function call
if __name__ == "__main__":
    main()
