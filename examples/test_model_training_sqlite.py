from timer import timer
import logging

import torch
from gnn_reco.data.sqlite_dataset import SQLiteDataset

from gnn_reco.models.detector.icecube86 import IceCube86
from gnn_reco.models.graph_builders import KNNGraphBuilder
from gnn_reco.models import Model
from gnn_reco.models.task.reconstruction import AngularReconstruction, EnergyReconstruction

from gnn_reco.components.loss_functions import LogCosh, LogCoshOfLogTransformed
from gnn_reco.models.training.trainers import Trainer, Predictor
from gnn_reco.models.training.callbacks import PiecewiseLinearScheduler
from gnn_reco.models.training.utils import make_train_validation_dataloader, save_results
from gnn_reco.data.constants import FEATURES, TRUTH
from gnn_reco.data.utils import get_even_neutrino_indicies
from gnn_reco.models.gnn import DynEdge, ConvNet

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86

# Main function definition
def main():

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuraiton
    db = '/groups/icecube/leonbozi/datafromrasmus/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'
    pulsemap = 'SRTTWOfflinePulsesDC'
    batch_size = 1024
    num_workers = 10
    device = 'cuda:1'
    target = 'energy'
    n_epochs = 30
    patience = 5
    archive = '/groups/icecube/asogaard/gnn/results'
    
    # Common variables
    selection = get_even_neutrino_indicies(db)[0:100000]

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db, 
        selection, 
        pulsemap, 
        batch_size, 
        features, 
        truth, 
        num_workers,
    )

    # Building model
    detector = IceCube86(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        nb_outputs=16,
    )
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs, 
        target_label=target, 
        loss_function=LogCoshOfLogTransformed(),
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-03)
    scheduler = PiecewiseLinearScheduler(training_dataloader, start_lr=1e-5, max_lr=1e-3, end_lr=1e-5, max_epochs=n_epochs)
    
    trainer = Trainer(
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader, 
        optimizer=optimizer, n_epochs=n_epochs, 
        scheduler=scheduler,
        patience=patience,
    )

    try:
        trained_model = trainer(model)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")
        pass
    
    predictor = Predictor(dataloader=validation_dataloader, target=target, device=device, output_column_names=[target + '_pred', target + '_uncert'])
    results = predictor(trained_model)
    save_results(db, 'dynedge_energy', results,archive, trained_model)

# Main function call
if __name__ == "__main__":
    main()
