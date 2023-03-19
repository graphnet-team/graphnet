"""Simplified example of training Model for energy reconstruction."""

from typing import List, Optional
import os

from graphnet.constants import EXAMPLE_OUTPUT_DIR, CONFIG_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
)


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]] = None,
    max_epochs: int = 5,
    batch_size: int = 16,
    num_workers: int = 12,
) -> None:
    """Run example."""
    # Build model
    model_config = ModelConfig.load(model_config_path)
    model = Model.from_config(model_config, trust=True)

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Train model
    model.fit(
        dataloaders["train"],
        dataloaders["validation"],
        gpus=gpus,
        max_epochs=max_epochs,
    )

    results = model.predict_as_dataframe(
        dataloaders["test"],
        additional_attributes=model.target + ["event_no"],
    )

    # Save predictions and model to file
    outdir = os.path.join(EXAMPLE_OUTPUT_DIR, "simple_energy_model_example")
    os.makedirs(outdir, exist_ok=True)
    results.to_csv(f"{outdir}/results.csv")
    model.save_state_dict(f"{outdir}/state_dict.pth")
    model.save(f"{outdir}/model.pth")


if __name__ == "__main__":
    dataset_config_path = (
        f"{CONFIG_DIR}/datasets/training_example_data_sqlite.yml"
    )
    model_config_path = (
        f"{CONFIG_DIR}/models/example_energy_reconstruction_model.yml"
    )
    main(
        dataset_config_path=dataset_config_path,
        model_config_path=model_config_path,
    )
