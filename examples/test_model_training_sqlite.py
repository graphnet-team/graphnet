from timer import timer
import logging

import torch

from graphnet.components.loss_functions import  LogCoshLoss, VonMisesFisher2DLoss
from graphnet.components.utils import fit_scaler
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices
from graphnet.legacy.callbacks import PiecewiseLinearScheduler
from graphnet.legacy.trainers import Trainer, Predictor
from graphnet.legacy.model import Model
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge, ConvNet
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
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

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuraiton
    db = '/groups/icecube/leonbozi/datafromrasmus/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db'
    pulsemap = 'SRTTWOfflinePulsesDC'
    batch_size = 256
    num_workers = 10
    device = 'cuda:0'
    target = 'energy'
    n_epochs = 5
    patience = 5
    archive = '/groups/icecube/asogaard/gnn/results'

    # Scalers
    scalers = fit_scaler(db, features, truth, pulsemap)

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(db)
    train_selection = train_selection[0:50000]

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db,
        train_selection,
        pulsemap,
        features,
        truth,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Building model
    detector = IceCube86(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
        scalers=scalers['input']['SRTTWOfflinePulsesDC'],
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        target_label=target,
        loss_function=LogCoshLoss(
            transform_prediction_and_target=torch.log10,
        ),
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, eps=1e-03)
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

    predictor = Predictor(
        dataloader=validation_dataloader,
        target=target,
        device=device,
        output_column_names=[target + '_pred'],
    )
    model._tasks[0].inference = True
    results = predictor(model)
    save_results(db, 'dynedge_energy', results,archive, model)

# Main function call
if __name__ == "__main__":
    main()
