import os.path
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import  LogCoshLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import get_predictions, make_train_validation_dataloader, save_results

from CLI.CLI_train_model_example import *

# Configurations
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE

# Main function definition
def main():
    
    #parser = argparse.ArgumentParser(description="parameters for training example")
    #parser.add_argument("--database", help="path, name and format of database.", type=str, required=True)
    #parser.add_argument("--pulsemap", help="", type=str, required=False, default='SRTTWOfflinePulsesDC')
    #parser.add_argument("--batch", help="batch size of training", type=int, required=False, default=512)
    #parser.add_argument("--workers", help="number of workers", required=False, default=10)
    #parser.add_argument("--gpu", help="Choose gpu to use [1/2]; default is None.", type=int, required=False, default=None)
    #parser.add_argument("--target", help="reconstruction target; energy, ...", type=str, required=False, default='energy')
    #parser.add_argument("--epochs", help="number of epochs to use.", type=int, required=False, default=5)
    #parser.add_argument("--patience", help="???", type=int, required=False, default=5)
    #parser.add_argument("--output", help="output path.", type=str, required=True)
    #args = parser.parse_args()

    savedir = args.output+'/wandb/'
    os.makedirs(savedir, exist_ok=True)
    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project="example-script",
        entity="graphnet-team",
        save_dir=savedir,
        log_model=False,
    )

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuration
    config = {
        "db": args.database,
        "pulsemap": args.pulsemap,
        "batch_size": args.batch,
        "num_workers": args.workers,
        "gpus": [args.gpu],
        "target": args.target,
        "n_epochs": args.epochs,
        "patience": args.patience,
    }
    
    archive = args.output
    os.makedirs(archive, exist_ok=True)

    run_name = "dynedge_{}_example".format(config["target"])

    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    train_selection = train_selection[0:50000]

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        config["db"],
        train_selection,
        config["pulsemap"],
        features,
        truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=torch.log10,
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * config["n_epochs"]],
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
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        gpus=config["gpus"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
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
        [config["target"] + '_pred'],
        [config["target"], 'event_no'],
    )

    save_results(config["db"], run_name, results, archive, model)

# Main function call
if __name__ == "__main__":
    main()