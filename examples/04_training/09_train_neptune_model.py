"""Example of training the Neptune point transformer.

Neptune tokenizes an event by farthest point sampling over the 4D
`(x, y, z, t)` point cloud, so it needs neither edges nor a cap on the number
of pulses: an `EdgelessGraph` over `NodesAsPulses` is all it requires.

See https://arxiv.org/abs/2510.01733.
"""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.data_representation import EdgelessGraph, NodesAsPulses
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
)
from graphnet.models.transformer import Neptune
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset

# Constants
features = FEATURES.PROMETHEUS
truth = TRUTH.PROMETHEUS_LEGACY


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
        "dataset_reference": (
            SQLiteDataset if path.endswith(".db") else ParquetDataset
        ),
    }

    # Neptune consumes the raw pulse cloud directly, so no edges and no
    # pulse-count cap are needed.
    data_representation = EdgelessGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        input_feature_names=features,
    )

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_neptune_model")
    run_name = "Neptune_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Use GraphNetDataModule to load in data
    dm = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "truth": truth,
            "truth_table": truth_table,
            "features": features,
            "data_representation": data_representation,
            "pulsemaps": [config["pulsemap"]],
            "path": config["path"],
            "index_column": "event_no",
            "labels": {
                "direction": Direction(
                    azimuth_key="injection_azimuth",
                    zenith_key="injection_zenith",
                )
            },
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
    )

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    # Building model. `Prometheus` standardizes positions as metres / 100 and
    # time as nanoseconds / 1.05e4, so `xyz_scale=0.1` and `time_scale=10.5`
    # bring them to the kilometres and microseconds Neptune expects. This
    # detector records no charge, hence `charge_column=None`.
    #
    # Note that the geometric priors -- `fourier_freq_min` / `_max`,
    # `rope_scales`, and the tokenizer's `metric_time_scale` -- are left at
    # their IceCube-tuned defaults here. Rescale them for a production run on
    # a detector of a very different size.
    backbone = Neptune(
        nb_inputs=data_representation.nb_outputs,
        coordinate_columns=[0, 1, 2],
        time_column=3,
        charge_column=None,
        xyz_scale=0.1,
        time_scale=10.5,
        num_patches=32,
        token_dim=64,
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        tokenizer_kwargs={"mlp_layers": [32, 64]},
        # `compile_encoder=True` is worth 2-4x on GPU, and is what enables
        # the packed attention path. Left off here to keep the example quick.
        compile_encoder=False,
    )
    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=VonMisesFisher3DLoss(),
    )
    model = StandardModel(
        data_representation=data_representation,
        backbone=backbone,
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
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
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
        config["target"] + "_x_pred",
        config["target"] + "_y_pred",
        config["target"] + "_z_pred",
        config["target"] + "_kappa_pred",
    ]

    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes,
        prediction_columns=prediction_columns,
        gpus=config["fit"]["gpus"],
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{path}/results.csv")

    # Save model config and state dict - Version safe save method.
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save_config(f"{path}/model_config.yml")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(description="""
Train the Neptune point transformer without the use of config files.
""")

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
        default="direction",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="mc_truth",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 1),
        ("early-stopping-patience", 2),
        ("batch-size", 16),
        ("num-workers", 2),
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args, unknown = parser.parse_known_args()

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
