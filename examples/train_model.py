import logging
import numpy as np
import pandas as pd
from timer import timer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import  LogCoshLoss, VonMisesFisher2DLoss
from graphnet.components.utils import fit_scaler
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge, ConvNet
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import get_predictions, make_train_validation_dataloader, save_results

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
    gpus = [0]
    target = 'energy'
    n_epochs = 5
    patience = 5
    archive = '/groups/icecube/asogaard/gnn/results'

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
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        target_label=target,
        loss_function=LogCoshLoss(),
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * n_epochs],
            'factors': [1e-2, 1, 1e-02],
        },
        scheduler_config={
            'interval': 'step',
        },
     )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        gpus=gpus,
        max_epochs=n_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [target + '_pred'],
        [target, 'event_no'],
    )

    save_results(db, 'dynedge_energy_pytorch_lightning', results,archive, model)

# Main function call
if __name__ == "__main__":
    main()
