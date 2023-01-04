"""Simplified example of training Model."""

import os

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)


def main(dataset_config_path: str, model_config_path: str) -> None:
    """Run example."""
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

    print(model._tasks[0]._target_labels)
    print(model._detector.nb_inputs)

    # Configuration
    config = TrainingConfig(
        target="energy",
        early_stopping_patience=5,
        fit={
            "gpus": [
                0,
            ],
            "max_epochs": 25,
        },
        dataloader={"batch_size": 128, "num_workers": 10},
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
        action="store",
        help="Path to dataset config file (default: %(default)s)",
        default="configs/datasets/example_data_sqlite.yml",
    )
    parser.add_argument(
        "--model-config",
        action="store",
        help="Path to model config file (default: %(default)s)",
        default="configs/models/example_model.yml",
    )

    args = parser.parse_args()

    main(args.dataset_config, args.model_config)
