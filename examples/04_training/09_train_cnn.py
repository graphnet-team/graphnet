"""Example of training a CNN Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import TRUTH
from graphnet.models import StandardModel
from graphnet.models.cnn import LCSC
from graphnet.models.data_representation import PercentileClusters
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
from graphnet.models.detector import ORCA150
from torch_geometric.data import Batch
from graphnet.models.data_representation.images import ExamplePrometheusImage

# Constants
features = ["sensor_id", "sensor_string_id", "t"]
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
        "dataset_reference": (
            SQLiteDataset if path.endswith(".db") else ParquetDataset
        ),
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_cnn_model")
    run_name = "lcsc_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Build pulses → pixels → image tensors via `ImageRepresentation`.
    # 1) `pixel_definition`: aggregates pulses (here per optical module).
    # 2) `grid_definition` (inside `ExamplePrometheusImage`): detector grid
    #    layouts and scatter into shaped tensors.
    # Multiple tensors per event are supported (e.g. IC86 main + DeepCore).

    # Here we use the PercentileClusters pixel definition, which
    # aggregates the light pulses that arrive at the same optical
    # module with percentiles.
    print(features)
    pixel_definition = PercentileClusters(
        cluster_on=["sensor_id", "sensor_string_id"],
        percentiles=[10, 50, 90],
        add_counts=True,
        input_feature_names=features,
    )

    # `ExamplePrometheusImage` wires a Prometheus `GridDefinition`
    # for the example layout.
    # It maps optical modules into the image
    # using the sensor_string_id and sensor_id
    # (number of the optical module).
    # The detector class standardizes the input features,
    # so that the features are in a ML friendly range.
    # For the mapping of the optical modules to the image it is
    # essential to not change the value of the sensor_id and
    # sensor_string_id. Therefore we need to make sure that
    # these features are not standardized, which is done by the
    # `replace_with_identity` argument of the detector.
    image_representation = ExamplePrometheusImage(
        detector=ORCA150(
            replace_with_identity=[
                "sensor_id",
                "sensor_string_id",
            ],
        ),
        pixel_definition=pixel_definition,
        input_feature_names=features,
        string_label="sensor_string_id",
        dom_number_label="sensor_id",
    )

    # Use SQLiteDataset to load in data
    # The input here depends on the dataset being used,
    # in this case the Prometheus dataset.
    dataset = SQLiteDataset(
        path=config["path"],
        pulsemaps=config["pulsemap"],
        truth_table=truth_table,
        features=features,
        truth=truth,
        data_representation=image_representation,
    )

    # Create the training and validation dataloaders.
    training_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        collate_fn=Batch.from_data_list,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        collate_fn=Batch.from_data_list,
    )

    # Building model

    # LCSC spatial size must match this `ImageRepresentation`'s grid.
    image_size = image_representation.single_image_spatial_shape()

    # Define architecture of the backbone, in this example
    # the LCSC architecture from Alexander Harnisch is used.
    backbone = LCSC(
        num_input_features=image_representation.nb_outputs,
        out_put_dim=2,
        input_norm=True,
        num_conv_layers=5,
        conv_filters=[5, 10, 20, 40, 60],
        kernel_size=3,
        image_size=image_size,
        pooling_type=[
            "Avg",
            None,
            "Avg",
            None,
            "Avg",
        ],
        pooling_kernel_size=[
            [1, 1, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
        ],
        pooling_stride=[
            [1, 1, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
        ],
        num_fc_neurons=50,
        norm_list=True,
        norm_type="Batch",
    )
    # Define the task.
    # Here an energy reconstruction, with a LogCoshLoss function.
    # The target and prediction are transformed using the log10 function.
    # When infering the prediction is transformed back to the
    # original scale using 10^x.
    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
        transform_inference=lambda x: torch.pow(10, x),
    )
    # Define the full model, which includes the backbone, task(s),
    # along with typical machine learning options such as
    # learning rate optimizers and schedulers.
    model = StandardModel(
        data_representation=image_representation,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
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
    additional_attributes = model.target_labels
    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config["fit"]["gpus"],
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{path}/cnn_results.csv")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{path}/cnn_state_dict.pth")
    model.save_config(f"{path}/cnn_model_config.yml")


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
        default="total_energy",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="mc_truth",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 1),
        "early-stopping-patience",
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
