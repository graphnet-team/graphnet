import numpy as np
from timer import timer
import logging

import torch

from gnn_reco.components.loss_functions import  VonMisesFisher2DLoss
from gnn_reco.components.utils import fit_scaler
from gnn_reco.data.constants import FEATURES, TRUTH
from gnn_reco.data.utils import get_equal_proportion_neutrino_indices
from gnn_reco.models import Model
from gnn_reco.models.detector import IceCubeDeepCore
from gnn_reco.models.gnn import DynEdge, ConvNet
from gnn_reco.models.graph_builders import KNNGraphBuilder
from gnn_reco.models.task.reconstruction import AzimuthReconstructionWithKappa, ZenithReconstructionWithKappa
from gnn_reco.models.training.callbacks import PiecewiseLinearScheduler
from gnn_reco.models.training.trainers import Trainer, Predictor
from gnn_reco.models.training.utils import make_dataloader, make_train_validation_dataloader, save_results
from gnn_reco.legacy.original import (
    Dynedge as LegacyDynedge,  
    vonmises_sinecosine_loss,
    log_cosh,
    Trainer as LegacyTrainer,
    Predictor as LegacyPredictor,
)

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86


def train_legacy_model(training_dataloader, validation_dataloader, test_dataloader,target, n_epochs, patience, scalers, device, db, archive):
    model = LegacyDynedge(k = 8, device = device, n_outputs= 3, scalers = scalers, target = target).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,eps = 1e-3)
    scheduler = PiecewiseLinearScheduler(training_dataloader, start_lr = 1e-5, max_lr= 1e-3, end_lr = 1e-5, max_epochs= n_epochs)
    loss_func = vonmises_sinecosine_loss

    trainer = LegacyTrainer(training_dataloader = training_dataloader, validation_dataloader= validation_dataloader, 
                    optimizer = optimizer, n_epochs = n_epochs, loss_func = loss_func, target = target, 
                    device = device, scheduler = scheduler, patience= patience)

    trained_model = trainer(model)

    predictor = LegacyPredictor(dataloader = test_dataloader, target = target, device = device, output_column_names= [target + '_pred', target + '_var'])
    results = predictor(trained_model)
    save_results(db, f'dynedge_{target}_legacy', results, archive, trained_model)


def train_new_model(training_dataloader, validation_dataloader, test_dataloader, target, n_epochs, patience, scalers, device, db, archive):
    
    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
        scalers=scalers['input']['SRTTWOfflinePulsesDC'],
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )
    task_ = {
        'zenith': ZenithReconstructionWithKappa,
        'azimuth': AzimuthReconstructionWithKappa,
    }[target]
    task = task_(
        hidden_size=gnn.nb_outputs, 
        target_label=target, 
        loss_function=VonMisesFisher2DLoss(),
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device=device,
    )

    # Training it
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-03)
    scheduler = PiecewiseLinearScheduler(training_dataloader, start_lr=1e-5, max_lr=1e-3, end_lr=1e-5, max_epochs=n_epochs)

    trainer = Trainer(
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader, 
        optimizer=optimizer,
        n_epochs=n_epochs, 
        scheduler=scheduler,
        patience=patience,
    )

    try:
        trainer(model)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")
        pass

    # Running inference
    predictor = Predictor(
        dataloader=test_dataloader, 
        target=target, 
        device=device, 
        output_column_names=[target + '_pred', target + '_kappa'],
    )
    results = predictor(model)
    save_results(db, f'dynedge_{target}', results,archive, model)


# Main function definition
def main(target):

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuraiton
    db = '/groups/hep/pcs557/GNNReco/data/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db'
    pulsemap = 'SRTTWOfflinePulsesDC'
    batch_size = 1024
    num_workers = 20
    device = 'cuda:0'
    n_epochs = 30
    patience = 5
    archive = '/groups/hep/pcs557/phd/paper/vonmisesfisher'

    # Scalers
    scalers = fit_scaler(db, features, truth, pulsemap)

    # Common variables
    train_selection, test_selection = get_equal_proportion_neutrino_indices(db)
    train_selection = train_selection
    
    test_dataloader = make_dataloader(db = db, pulsemap = pulsemap, features = features, truth = truth, batch_size = batch_size, shuffle= True, selection = test_selection, num_workers = num_workers)
    training_dataloader, validation_dataloader = make_train_validation_dataloader(db =  db, selection = train_selection, pulsemap = pulsemap, features = features, truth = truth, batch_size= batch_size, num_workers = num_workers) 
    
    args = (
        training_dataloader,
        validation_dataloader,
        test_dataloader,
        target,
        n_epochs,
        patience,
        scalers,
        device,
        db,
        archive,
    )

    #train_legacy_model(*args)
    train_new_model(*args)
    
    
# Main function call
if __name__ == "__main__":
    main('azimuth')
    main('zenith')
