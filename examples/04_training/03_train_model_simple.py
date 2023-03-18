"""Simplified example of training Model."""

from typing import List, Optional
import os

from graphnet.constants import EXAMPLE_OUTPUT_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    suffix: Optional[str] = None,
) -> None:
    """Run example."""
    # Build model
    model_config = ModelConfig.load(model_config_path)
    model = Model.from_config(model_config, trust=True)

    # Configuration
    training_config = TrainingConfig(
        target=model.target,
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **training_config.dataloader,
    )

    # Train model
    model.fit(
        dataloaders["train"],
        dataloaders["validation"],
        early_stopping_patience=training_config.early_stopping_patience,
        **training_config.fit,
    )

    results = model.predict_as_dataframe(
        dataloaders["test"],
        additional_attributes=model.target + ["event_no"],
    )

    # Path parsing
    archive = os.path.join(EXAMPLE_OUTPUT_DIR, f"train_model_{suffix}")
    run_name = "dynedge_{}_example".format("_".join(training_config.target))
    db_name = dataset_config.path.split("/")[-1].split(".")[0]

    # Save predictions and model to file
    path = os.path.join(archive, db_name, run_name)
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
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default="example",
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
        args.suffix,
    )
