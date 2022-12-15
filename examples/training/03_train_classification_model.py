"""Example of training Model."""

import os
from typing import Dict, Any

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)

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


def train(general_config: Dict[str, Any]) -> None:
    """Train model with configuration given by `config`."""
    # Configuration
    config = TrainingConfig(
        target="pid",
        early_stopping_patience=5,
        fit={"gpus": [0], "max_epochs": 5},
        dataloader={"batch_size": 512, "num_workers": 10},
    )

    run_name = "dynedge_{}_classification_example".format(config.target)

    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Construct dataloaders
    dataset_config = DatasetConfig.load(
        "configs/datasets/" + general_config["dataset"] + ".yml"
    )
    # dataloader_test, dataloader_valid, ..
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **config.dataloader,
    )
    wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Build model
    model_config = ModelConfig.load(
        "configs/models/" + general_config["model"] + ".yml"
    )
    model = Model.from_config(model_config, trust=True)
    wandb_logger.experiment.config.update(model_config.as_dict())

    # Training model
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
        prediction_columns = [
            config.target + "_noise_pred",
            config.target + "_muon_pred",
            config.target + "_neutrino_pred",
        ]
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
    path = os.path.join(general_config["archive"], db_name, run_name)

    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


def main() -> None:
    """Run example."""
    # General configuration
    general_config = {
        "dataset": "PID_classification_last_one_lvl3MC.yml",
        "model": "dynedge_PID_Classification_noise_muon_neutrino_example.yml",
        "archive": "/groups/icecube/petersen/GraphNetDatabaseRepository/example_results/train_classification_model",
    }

    train(general_config)


if __name__ == "__main__":
    main()
