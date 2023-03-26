"""Simplified example of training Model."""

from typing import List, Optional
import os

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.constants import EXAMPLE_OUTPUT_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger


# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    prediction_names: Optional[List[str]],
    suffix: Optional[str] = None,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project="example-script",
        entity="graphnet-team",
        save_dir=WANDB_DIR,
        log_model=True,
    )

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model = Model.from_config(model_config, trust=True)

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

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **config.dataloader,
    )

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    if rank_zero_only.rank == 0:
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
        logger=wandb_logger,
        **config.fit,
    )

    # Get predictions
    if isinstance(config.target, str):
        prediction_columns = [config.target + "_pred"]
        additional_attributes = [config.target]
    else:
        prediction_columns = [target + "_pred" for target in config.target]
        additional_attributes = config.target

    if prediction_names:
        prediction_columns = prediction_names

    logger.info(f"config.target: {config.target}")
    logger.info(f"prediction_columns: {prediction_columns}")

    results = model.predict_as_dataframe(
        dataloaders["test"],
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    # Save predictions and model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
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
Train GNN model.
"""
    )

    parser.with_standard_arguments(
        "dataset-config",
        "model-config",
        "gpus",
        ("max-epochs", 5),
        "early-stopping-patience",
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--prediction-names",
        nargs="+",
        help="Names of each prediction output feature (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default=None,
    )

    args = parser.parse_args()

    main(
        args.dataset_config,
        args.model_config,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.prediction_names,
        args.suffix,
    )
