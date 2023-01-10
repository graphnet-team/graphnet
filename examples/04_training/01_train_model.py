"""Simplified example of training Model."""

from typing import List, Optional
import os

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.constants import (
    TRAINING_EXAMPLE_DATA_DIR,
    TRAINING_EXAMPLE_SQLITE_DATA,
    TRAINING_EXAMPLE_PARQUET_DATA,
)
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import get_logger


# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

logger = get_logger()


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
) -> None:
    """Run example."""
    # Check data availability
    if "training_example_data" in dataset_config_path:
        if not (
            os.path.exists(TRAINING_EXAMPLE_DATA_DIR)
            and os.path.exists(TRAINING_EXAMPLE_SQLITE_DATA)
            and os.path.exists(TRAINING_EXAMPLE_PARQUET_DATA)
        ):
            logger.error("Training example data was not found in:")
            logger.error(f"  {TRAINING_EXAMPLE_DATA_DIR}")
            logger.error("Please download it using:")
            logger.error("$ source get_training_example_data.sh")
            return

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
        target=model._tasks[0]._target_labels[0],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    archive = "/tmp/graphnet/results/"
    run_name = "dynedge_{}_example".format(config.target)

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **config.dataloader,
    )

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    if rank_zero_only == 0:
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

    results = model.predict_as_dataframe(
        dataloaders["test"],
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    # Save predictions and model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)

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

    parser.add_argument(
        "--dataset-config",
        help="Path to dataset config file (default: %(default)s)",
        default="configs/datasets/training_example_data_sqlite.yml",
    )
    parser.add_argument(
        "--model-config",
        help="Path to model config file (default: %(default)s)",
        default="configs/models/example_model.yml",
    )

    parser.with_standard_arguments(
        "gpus",
        "max-epochs",
        "early-stopping-patience",
        "batch-size",
        "num-workers",
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
    )
