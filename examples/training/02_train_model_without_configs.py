"""Example of training Model."""

import os
from typing import cast

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.training.loss_functions import LogCoshLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_equal_proportion_neutrino_indices,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
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


def main() -> None:
    """Run example."""
    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config = {
        "db": "/groups/icecube/asogaard/data/sqlite/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db",
        "pulsemap": "SRTTWOfflinePulsesDC",
        "batch_size": 512,
        "num_workers": 10,
        "accelerator": "gpu",
        "devices": [0],
        "target": "energy",
        "n_epochs": 5,
        "patience": 5,
    }
    archive = "/groups/icecube/asogaard/gnn/results/"
    run_name = "dynedge_{}_example".format(config["target"])

    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(
        cast(str, config["db"])
    )
    train_selection = train_selection[0:50000]

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
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=torch.log10,
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * cast(int, config["n_epochs"]),
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
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
        [cast(str, config["target"]) + "_pred"],
        additional_attributes=[cast(str, config["target"]), "event_no"],
    )

    save_results(cast(str, config["db"]), run_name, results, archive, model)


if __name__ == "__main__":
    main()
