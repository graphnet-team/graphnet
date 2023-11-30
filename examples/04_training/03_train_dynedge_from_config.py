"""Simplified example of training DynEdge from pre-defined config files."""

from typing import List, Optional
import os

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.constants import EXAMPLE_OUTPUT_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    suffix: Optional[str] = None,
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

    # Build model from pre-defined config file made from Model.save_config
    model_config = ModelConfig.load(model_config_path)
    model: StandardModel = StandardModel.from_config(model_config, trust=True)

    # Configuration
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    if suffix is not None:
        archive = os.path.join(EXAMPLE_OUTPUT_DIR, f"train_model_{suffix}")
    else:
        archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_model")
    run_name = "dynedge_{}_example".format("_".join(config.target))

    # Construct dataloaders from pre-defined dataset config files.
    # i.e. from Dataset.save_config
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **config.dataloader,
    )

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    if wandb and rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(model_config.as_dict())
        wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Train model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
        ),
        ProgressBar(),
    ]

    model.fit(
        dataloaders["train"],
        dataloaders["validation"],
        callbacks=callbacks,
        logger=wandb_logger if wandb else None,
        **config.fit,
    )

    # Save model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    os.makedirs(path, exist_ok=True)
    logger.info(f"Writing results to {path}")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")

    # Get predictions
    if isinstance(config.target, str):
        additional_attributes = [config.target]
    else:
        additional_attributes = config.target

    logger.info(f"config.target: {config.target}")
    logger.info(f"prediction_columns: {model.prediction_labels}")

    results = model.predict_as_dataframe(
        dataloaders["test"],
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config.fit["gpus"],
    )
    results.to_csv(f"{path}/results.csv")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model.
"""
    )

    parser.with_standard_arguments(
        "dataset-config",
        "model-config",
        "gpus",
        ("max-epochs", 1),
        "early-stopping-patience",
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args, unknown = parser.parse_known_args()

    main(
        args.dataset_config,
        args.model_config,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.suffix,
        args.wandb,
    )
