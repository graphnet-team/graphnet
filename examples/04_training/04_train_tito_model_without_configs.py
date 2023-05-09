"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
)
from graphnet.training.labels import Direction
from graphnet.training.callbacks import ProgressBar
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger

# Constants
features = FEATURES.PROMETHEUS
truth = TRUTH.PROMETHEUS


def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_tito_model")
    run_name = "dynedgeTITO_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        config["path"],
        list(range(0, 100)),  # subset of events for speeding up training
        config["pulsemap"],
        features,
        truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        truth_table=truth_table,
        index_column="event_no",
        labels={
            "direction": Direction(
                azimuth_key="injection_azimuth", zenith_key="injection_zenith"
            )
        },
    )

    # Building model
    detector = Prometheus(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=6),
    )
    gnn = DynEdgeTITO(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["max"],
        layer_size_scale=3,  # 3x the default layer size [256, 256]
    )
    task = DirectionReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=VonMisesFisher3DLoss(),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": config["early_stopping_patience"],
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

    model.fit(
        training_dataloader,
        validation_dataloader,
        callbacks=callbacks,
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Get predictions
    additional_attributes = [
        "injection_zenith",
        "injection_azimuth",
        "event_no",
    ]
    prediction_columns = [
        config["target"][0] + "_x_pred",
        config["target"][0] + "_y_pred",
        config["target"][0] + "_z_pred",
        config["target"][0] + "_kappa_pred",
    ]

    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes,
        prediction_columns=prediction_columns,
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model without the use of config files.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="total",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default=["direction"],
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="mc_truth",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 5),
        ("early-stopping-patience", 2),
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args = parser.parse_args()

    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.wandb,
    )
