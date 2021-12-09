import logging
import numpy as np
import pandas as pd
from timer import timer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.optim.adam import Adam
from torch.utils.data import dataloader

from gnn_reco.components.loss_functions import  LogCoshLoss, VonMisesFisher2DLoss
from gnn_reco.components.utils import fit_scaler
from gnn_reco.data.constants import FEATURES, TRUTH
from gnn_reco.data.utils import get_desired_event_numbers
from gnn_reco.models import Model
from gnn_reco.models.detector.icecube import IceCubeUpgrade
from gnn_reco.models.gnn import DynEdge, ConvNet
from gnn_reco.models.graph_builders import KNNGraphBuilder
from gnn_reco.models.task.reconstruction import EnergyReconstruction
from gnn_reco.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from gnn_reco.models.training.utils import get_predictions, make_train_validation_dataloader, save_results

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

# Main function definition
def main():

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuraiton
    db = '/groups/icecube/asogaard/temp/sqlite_test_upgrade/data_test/data/data_test.db'
    pulsemap = 'I3RecoPulseSeriesMapRFCleaned_mDOM'
    batch_size = 128
    num_workers = 10
    gpus = [0]
    target = 'energy'
    n_epochs = 30
    patience = 5
    archive = '/groups/icecube/asogaard/gnn/results'

    # Common variables
    train_selection = get_desired_event_numbers(db, 1000000, fraction_nu_e=1.)

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
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
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

    save_results(db, 'test_upgrade_mDOM_energy', results, archive, model)

# Main function call
if __name__ == "__main__":
    main()
