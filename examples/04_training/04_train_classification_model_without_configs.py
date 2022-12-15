"""Example of training Model."""

import os
from typing import cast, Dict, Any
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.nn.functional import softmax

from graphnet.training.loss_functions import CrossEntropyLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
)

from graphnet.utilities.logging import get_logger

logger = get_logger()

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="example-script",
    entity="graphnet-team",
    save_dir=WANDB_DIR,
    log_model=True,
)


def train(config: Dict[str, Any]) -> None:
    """Train model with configuration given by `config`."""
    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables; equal distribution of pid.
    train_selection = get_desired_event_numbers(
        database=cast(str, config["db"]),
        desired_size=cast(int, config["event_numbers"]),
        fraction_noise=float(1 / 3),
        fraction_muon=float(1 / 3),
        fraction_nu_e=float(1 / 9),
        fraction_nu_mu=float(1 / 9),
        fraction_nu_tau=float(1 / 9),
    )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        cast(str, config["db"]),
        train_selection,
        cast(str, config["pulsemap"]),
        features,
        truth,
        batch_size=cast(int, config["batch_size"]),
        num_workers=cast(int, config["num_workers"]),
    )

    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = MulticlassClassificationTask(
        nb_outputs=len(np.unique(list(config["class_options"].values()))),
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=CrossEntropyLoss(options=config["class_options"]),
        transform_inference=lambda x: softmax(x, dim=-1),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-04, "eps": 1e-03},
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [
            cast(str, config["target"]) + "_noise_pred",
            cast(str, config["target"]) + "_muon_pred",
            cast(str, config["target"]) + "_neutrino_pred",
        ],
        additional_attributes=[cast(str, config["target"]), "event_no"],
    )

    save_results(
        cast(str, config["db"]),
        config["run_name"],
        results,
        config["archive"],
        model,
    )


def main() -> None:
    """Run example."""
    # transformation of target to a given class integer
    class_options = {
        1: 0,
        -1: 0,
        13: 1,
        -13: 1,
        12: 2,
        -12: 2,
        14: 2,
        -14: 2,
        16: 2,
        -16: 2,
    }

    target = "pid"

    archive = "/groups/icecube/petersen/GraphNetDatabaseRepository/example_results/train_classification_model"
    run_name = "dynedge_{}_example".format(target)

    # Configuration
    config = {
        "db": "/groups/icecube/petersen/GraphNetDatabaseRepository/Leon2022_DataAndMC_CSVandDB_StoppedMuons/last_one_lvl3MC.db",
        "pulsemap": "SRTInIcePulses",
        "batch_size": 512,
        "num_workers": 10,
        "accelerator": "gpu",
        "devices": [1],
        "target": target,
        "event_numbers": 1000,
        "class_options": class_options,
        "n_epochs": 10,
        "patience": 5,
        "archive": archive,
        "run_name": run_name,
    }

    train(config)


if __name__ == "__main__":
    main()
