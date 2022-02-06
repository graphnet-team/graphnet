"""NB: Need to be updated to use the transform-functionality in Task.
"""
import dill
import os
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import  BinaryCrossEntropyLoss, LogCoshLoss, VonMisesFisher2DLoss, XYZWithMaxScaling
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge_V2
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction, ZenithReconstructionWithKappa, XYZReconstruction, BinaryClassificationTask
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import get_predictions, make_train_validation_dataloader, save_results, Predictor

# Configurations
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

# Utility methods
def load_model(run_name, device, archive):
    model = torch.load(os.path.join(archive, f"{run_name}.pth"), pickle_module=dill)
    model.to('cuda:%s'%device[0])
    return model

def remove_log10(x, target):
    x[target + '_pred'] = 10**x[target + '_pred']
    return x

def rescale_XYZ(x, target):
    x['position_x_pred'] = 764.431509*x['position_x_pred']
    x['position_y_pred'] = 785.041607*x['position_y_pred']
    x['position_z_pred'] = 1083.249944*x['position_z_pred']
    return x


def train_and_predict_on_validation_set(target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device, patience = 5):

    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project=f"upgrade-{target}-new-noise-model-GraphSAGE-cleaned",
        entity="graphnet-team",
        save_dir='./wandb/',
        log_model=True,
    )

    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass

    print(f"features: {features}")
    print(f"truth: {truth}")

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db = database,
        selection =  selection,
        pulsemaps = pulsemap,
        features = features,
        truth = truth,
        batch_size = batch_size,
        num_workers=num_workers,
    )

    # Building model
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge_V2(
        nb_inputs=detector.nb_outputs,
    )
    if target == 'zenith':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=VonMisesFisher2DLoss(),
        )
    elif target == 'energy':
        task = EnergyReconstruction(hidden_size=gnn.nb_outputs, target_labels=target, loss_function=LogCoshLoss())
    elif target == 'track':
        task = BinaryClassificationTask(hidden_size=gnn.nb_outputs,target_labels=target,loss_function=BinaryCrossEntropyLoss())
    elif isinstance(target, list):
        task = XYZReconstruction(hidden_size=gnn.nb_outputs, target_labels=target, loss_function=XYZWithMaxScaling())
    else:
        print('task not found')

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
        default_root_dir=archive,
        gpus=device,
        max_epochs=n_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")
        pass

    # Saving model
    model.save(os.path.join(archive, f"{run_name}.pth"))
    model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))

def predict(target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device):
    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass
    model = load_model(run_name, device, archive)

    device = 'cuda:%s'%device[0]
    _, validation_dataloader = make_train_validation_dataloader(
        db = database,
        selection =  selection,
        pulsemaps = pulsemap,
        features = features,
        truth = truth,
        batch_size = batch_size,
        num_workers=num_workers,
    )
    if target in ['zenith', 'azimuth']:
        predictor_valid = Predictor(
            dataloader=validation_dataloader,
            target=target,
            device=device,
            output_column_names=[target + '_pred', target + '_kappa'],
        )
    if target in ['track', 'neutrino']:
        predictor_valid = Predictor(
            dataloader=validation_dataloader,
            target=target,
            device=device,
            output_column_names=[target + '_pred'],
        )
    if target == 'energy':
        predictor_valid = Predictor(
            dataloader=validation_dataloader,
            target=target,
            device=device,
            output_column_names=[target + '_pred'],
            post_processing_method= remove_log10,
        )
    if isinstance(target, list):
        predictor_valid = Predictor(
            dataloader=validation_dataloader,
            target=target,
            device=device,
            output_column_names=['position_x_pred','position_y_pred','position_z_pred'],#,'interaction_time_pred'],
            post_processing_method= rescale_XYZ,
        )

    results = predictor_valid(model)

    save_results(database, run_name, results, archive,model)

# Main function call
if __name__ == "__main__":

    # Run management
    archive = "/lustre/hpc/icecube/asogaard/gnn/results"
    targets = [['position_x', 'position_y', 'position_z']] # 'track', 'zenith', 'energy', 'XYZ',
    batch_size = 256
    database ='/lustre/hpc/icecube/asogaard/data/sqlite/dev_step4_numu_140022_second_run/data/dev_step4_numu_140022_second_run.db'
    device = [0]
    n_epochs = 30
    num_workers = 40
    patience = 5
    pulsemap = 'SplitInIcePulses_GraphSage_Pulses'

    # Common variables
    for target in targets:
        if target == 'track':
            selection = pd.read_csv('/lustre/hpc/icecube/asogaard/data/sqlite/dev_step4_numu_140022_second_run/selection/even_track_cascade_over20pulses.csv')['event_no'].values.ravel().tolist()
        else:
            selection = pd.read_csv('/lustre/hpc/icecube/asogaard/data/sqlite/dev_step4_numu_140022_second_run/selection/over20pulses.csv')['event_no'].values.ravel().tolist()

        if isinstance(target, list):
            target_name = 'XYZ'
        else:
            target_name = target

        run_name = "upgrade_{}_regression_GraphSagePulses".format(target_name)

        train_and_predict_on_validation_set(target, selection, database, pulsemap, batch_size, num_workers, n_epochs, device, patience)
        predict(target, selection, database, pulsemap, batch_size, num_workers, n_epochs, device)
