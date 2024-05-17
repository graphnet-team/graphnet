"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.flows import INGA
from graphnet.models.graphs import GraphDefinition
from graphnet.models.graphs.nodes import NodesAsPulses

from graphnet.models.task import StandardFlowTask
from graphnet.training.loss_functions import (
    MultivariateGaussianFlowLoss,
)
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

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_model_without_configs")
    run_name = "INGA_example_1mio"

    # Define graph representation
    detector = Prometheus()

    graph_definition = GraphDefinition(
        detector=detector,
        node_definition=NodesAsPulses(),
        input_feature_names=features,
    )
    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        db=config["path"],
        graph_definition=graph_definition,
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        truth_table=truth_table,
        selection=None,
    )

    # Building model
    flow = INGA(
        nb_inputs=graph_definition.nb_outputs,
        n_knots=120,
        num_blocks=4,
        b=100,
        c=100,
    )
    task = StandardFlowTask(
        target_labels=graph_definition.output_feature_names,
        loss_function=MultivariateGaussianFlowLoss(),
        coordinate_columns=flow.coordinate_columns,
        jacobian_columns=flow.jacobian_columns,
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=flow,
        tasks=[task],
    )

    model.fit(
        training_dataloader,
        validation_dataloader,
        **config["fit"],
    )
    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=["event_no"],
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{path}/results.csv")

    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    model.save(f"{path}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save_config(f"{path}/model_config.yml")


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
        "num-workers",
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
