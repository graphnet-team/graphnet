"""Utility functions for `graphnet.training`."""

from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models import Model
from graphnet.utilities.logging import get_logger

logger = get_logger()


# @TODO: Remove in favour of DataLoader{,.from_dataset_config}
def make_dataloader(
    db: str,
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    shuffle: bool,
    selection: List[int] = None,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    node_truth_table: str = None,
    string_selection: List[int] = None,
    loss_weight_table: str = None,
    loss_weight_column: str = None,
) -> DataLoader:
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = SQLiteDataset(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        selection=selection,
        node_truth=node_truth,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
    )

    def collate_fn(graphs: List[Data]) -> Batch:
        """Remove graphs with less than two DOM hits.

        Should not occur in "production.
        """
        graphs = [g for g in graphs if g.n_pulses > 1]
        return Batch.from_data_list(graphs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader


# @TODO: Remove in favour of DataLoader{,.from_dataset_config}
def make_train_validation_dataloader(
    db: str,
    selection: List[int],
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    database_indices: List[int] = None,
    seed: int = 42,
    test_size: float = 0.33,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: str = None,
    node_truth_table: str = None,
    string_selection: List[int] = None,
    loss_weight_column: str = None,
    loss_weight_table: str = None,
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and test `DataLoader` instances."""
    # Reproducibility
    rng = np.random.RandomState(seed=seed)

    # Checks(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    # Perform train/validation split
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame(
            {"event_no": selection, "db": database_indices}
        )
        shuffled_df = df_for_shuffle.sample(
            frac=1, replace=False, random_state=rng
        )
        training_df, validation_df = train_test_split(
            shuffled_df, test_size=test_size, random_state=rng
        )
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
    else:
        training_selection, validation_selection = train_test_split(
            selection, test_size=test_size, random_state=rng
        )

    # Create DataLoaders
    common_kwargs = dict(
        db=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        node_truth=node_truth,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_column=loss_weight_column,
        loss_weight_table=loss_weight_table,
    )

    training_dataloader = make_dataloader(
        shuffle=True,
        selection=training_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    validation_dataloader = make_dataloader(
        shuffle=False,
        selection=validation_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    return (
        training_dataloader,
        validation_dataloader,
    )


# @TODO: Remove in favour of Model.predict{,_as_dataframe}
def get_predictions(
    trainer: Trainer,
    model: Model,
    dataloader: DataLoader,
    prediction_columns: List[str],
    *,
    node_level: bool = False,
    additional_attributes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get `model` predictions on `dataloader`."""
    # Gets predictions from model on the events in the dataloader.
    # NOTE: dataloader must NOT have shuffle = True!

    # Check(s)
    if additional_attributes is None:
        additional_attributes = []
    assert isinstance(additional_attributes, list)

    # Set model to inference mode
    model.inference()

    # Get predictions
    predictions_torch = trainer.predict(model, dataloader)
    predictions_list = [
        p[0].detach().cpu().numpy() for p in predictions_torch
    ]  # Assuming single task
    predictions = np.concatenate(predictions_list, axis=0)
    try:
        assert len(prediction_columns) == predictions.shape[1]
    except IndexError:
        predictions = predictions.reshape((-1, 1))
        assert len(prediction_columns) == predictions.shape[1]

    # Get additional attributes
    attributes: Dict[str, List[np.ndarray]] = OrderedDict(
        [(attr, []) for attr in additional_attributes]
    )
    for batch in dataloader:
        for attr in attributes:
            attribute = batch[attr].detach().cpu().numpy()
            if node_level:
                if attr == "event_no":
                    attribute = np.repeat(
                        attribute, batch["n_pulses"].detach().cpu().numpy()
                    )
            attributes[attr].extend(attribute)

    data = np.concatenate(
        [predictions]
        + [
            np.asarray(values)[:, np.newaxis] for values in attributes.values()
        ],
        axis=1,
    )

    results = pd.DataFrame(
        data, columns=prediction_columns + additional_attributes
    )
    return results


# @TODO: Remove
def save_results(
    db: str, tag: str, results: pd.DataFrame, archive: str, model: Model
) -> None:
    """Save trained model and prediction `results` in `db`."""
    db_name = db.split("/")[-1].split(".")[0]
    path = archive + "/" + db_name + "/" + tag
    os.makedirs(path, exist_ok=True)
    results.to_csv(path + "/results.csv")
    model.save_state_dict(path + "/" + tag + "_state_dict.pth")
    model.save(path + "/" + tag + "_model.pth")
    logger.info("Results saved at: \n %s" % path)
