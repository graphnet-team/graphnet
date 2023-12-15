"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
import numpy as np
import pandas as pd

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.flows import INGA
from graphnet.models.graphs import GraphDefinition
from graphnet.models.graphs.nodes import NodesAsPulses

from graphnet.models.task import StandardFlowTask
from graphnet.training.loss_functions import (  # type : ignore
    LogCoshLoss,  # MultivariateGaussianFlowLoss,  # type : ignore
)  # type : ignore
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
        input_feature_names=input_feature_names,
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
        prediction_labels=graph_definition.output_feature_names,
        loss_function=LogCoshLoss,  # MultivariateGaussianFlowLoss(),
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
    database = "/mnt/scratch/rasmus_orsoe/databases/dev_level2_oscNext_pulsenoise_full_v4_remerge_v2/data/dev_level2_oscNext_pulsenoise_full_v4_remerge_v2_part_1.db"
    pulsemap = "SRTTWOfflinePulsesDC"
    target = ""
    truth_table = "truth"
    gpus = [2]
    max_epochs = 400
    early_stopping_patience = 16
    batch_size = 500
    num_workers = 30
    input_feature_names = ["dom_x", "dom_y", "dom_z", "dom_time"]
    string_selection = [83.0, 84.0, 85.0, 86.0]

    string_mask = []
    for string in np.arange(0, 87):
        if string not in string_selection:
            string_mask.append(string)

    main(
        path=database,
        pulsemap=pulsemap,
        target=target,
        truth_table=truth_table,
        gpus=gpus,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        num_workers=num_workers,
    )
