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
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge, ConvNet
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction, ZenithReconstructionWithKappa
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import get_predictions, make_train_validation_dataloader, save_results

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

# Main function definition
def main():

    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuraiton
    db = '/groups/icecube/asogaard/data/sqlite/dev_upgrade_step4_preselection_decemberv2/data/dev_upgrade_step4_preselection_decemberv2.db'
    pulsemaps = ['I3RecoPulseSeriesMapRFCleaned_mDOM', 'IceCubePulsesTWSRT']
    batch_size = 1024
    num_workers = 30
    gpus = [0]
    target = 'zenith'
    n_epochs = 30
    patience = 5
    archive = '/groups/icecube/asogaard/gnn/results'

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(db)
    #train_selection = train_selection[0:50000]

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db,
        train_selection,
        pulsemaps,
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
    #task = EnergyReconstruction(
    #    hidden_size=gnn.nb_outputs,
    #    target_label=target,
    #    loss_function=LogCoshLoss(
    #        transform_prediction_and_target=torch.log10,
    #    ),
    #)
    task = ZenithReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_label=target,
        loss_function=VonMisesFisher2DLoss(),
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

    model.save("test_upgrade_mc_mdom_zenith_regression.pth")

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [target + '_pred', target + '_kappa'],
        [target, 'event_no'],
    )

    save_results(db, 'test_upgrade_mc_mdom_zenith_regression', results, archive, model)

# Main function call
if __name__ == "__main__":
    main()
