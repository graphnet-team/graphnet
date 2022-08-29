"""Training GNN models for neutrino mass ordering studies for IceCube-Upgrade."""

import argparse
import sqlite3
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import (
    MSELoss,
    LogCoshLoss,
    VonMisesFisher2DLoss,
)
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge_V2
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    EnergyReconstruction,
    InelasticityReconstruction,
    ZenithReconstructionWithKappa,
)
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import (
    get_predictions,
    make_dataloader,
    save_results,
)
from graphnet.utilities.logging import get_logger

logger = get_logger()


# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

TARGETS = ["zenith", "energy", "energy_track", "inelasticity"]


# Main function definition
def main(target: str):

    assert target in TARGETS

    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project=f"upgrade-nmo-sensitivity-{target}",
        entity="graphnet-team",
        save_dir="./wandb/",
        log_model=True,
    )

    # Configuration
    config = {
        "db": "/groups/icecube/asogaard/data/sqlite/upgrade_genie_step4_june2022/data/upgrade_genie_step4_june2022.db",
        "pulsemaps": [
            "SplitInIcePulses_GraphSage_Pulses",
        ],
        "batch_size": 256,
        "num_workers": 30,
        "accelerator": "gpu",
        "devices": [0],
        "target": target,
        "n_epochs": 50,
        "patience": 5,
        "gnn/type": "DynEdge_V2",
        "gnn/size_scale": 3,
    }

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Run management
    archive = "/groups/icecube/asogaard/gnn/results/upgrade_nmo_sensitivity/"
    run_name = "{}_regression".format(config["target"])

    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables
    with sqlite3.connect(config["db"]) as conn:
        inclusive_selection = pd.read_sql_query(
            "SELECT event_no FROM truth", conn
        ).values.ravel()

    training_selection = list(
        filter(lambda event_no: event_no % 5 > 0, inclusive_selection)
    )
    testing_selection = list(
        filter(lambda event_no: event_no % 5 == 0, inclusive_selection)
    )

    dataloader_opts = dict(
        db=config["db"],
        pulsemaps=config["pulsemaps"],
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    training_dataloader = make_dataloader(
        shuffle=True,
        selection=training_selection,
        **dataloader_opts,
    )
    testing_dataloader = make_dataloader(
        shuffle=False,
        selection=testing_selection,
        **dataloader_opts,
    )

    # Building model
    def model_factory():
        detector = IceCubeUpgrade(
            graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
        )
        gnn = DynEdge_V2(
            nb_inputs=detector.nb_outputs,
            layer_size_scale=config["gnn/size_scale"],
        )

        if config["target"] == "zenith":
            task = ZenithReconstructionWithKappa(
                hidden_size=gnn.nb_outputs,
                target_labels=config["target"],
                loss_function=VonMisesFisher2DLoss(),
            )
        elif config["target"] in ["energy", "energy_track"]:
            task = EnergyReconstruction(
                hidden_size=gnn.nb_outputs,
                target_labels=config["target"],
                loss_function=LogCoshLoss(),
                transform_prediction_and_target=lambda p: torch.log10(p + 1),
            )
        else:
            task = InelasticityReconstruction(
                hidden_size=gnn.nb_outputs,
                target_labels=config["target"],
                loss_function=MSELoss(),
            )

        model = Model(
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
                    len(training_dataloader) * config["n_epochs"],
                ],
                "factors": [1e-2, 1, 1e-02],
            },
            scheduler_config={
                "interval": "step",
            },
        )
        return model

    model = model_factory()

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        default_root_dir=archive,
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, testing_dataloader)
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    if target == "zenith":
        prediction_columns = [
            config["target"] + "_pred",
            config["target"] + "_kappa",
        ]

    elif target in ["energy", "energy_track", "inelasticity"]:
        prediction_columns = [
            config["target"] + "_pred",
        ]

    else:
        raise ValueError(f"Target {target} not supported")

    results = get_predictions(
        trainer,
        model,
        testing_dataloader,
        prediction_columns,
        additional_attributes=[
            config["target"],
            "event_no",
            "n_pulses",
        ],
    )

    # Resetting model (to avoid saving Trainer, DataLoader, etc.)
    model.save_state_dict("state_dict.pth")
    model = model_factory()
    model.load_state_dict("state_dict.pth")

    save_results(config["db"], run_name, results, archive, model)


# Main function call
if __name__ == "__main__":
    # Choose target
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS)
    args = parser.parse_args()

    main(args.target)
