"""Utility functions for `graphnet.training`."""

from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from graphnet.data.dataset import Dataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
from graphnet.models import Model
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition


def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production".
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


class collator_sequence_buckleting:
    """Perform the sequence bucketing for the graphs in the batch."""

    def __init__(self, batch_splits: List[float] = [0.8]):
        """Set cutting points of the different mini-batches.

        batch_splits: list of floats, each element is the fraction of the total
        number of graphs. This list should not explicitly define the first and
        last elements, which will always be 0 and 1 respectively.
        """
        self.batch_splits = batch_splits

    def __call__(self, graphs: List[Data]) -> Batch:
        """Execute sequence bucketing on the input list of graphs.

        Args:
            graphs: A list of Data objects representing the input graphs.

        Returns:
            A list of Batch objects, each containing a mini-batch of the input
            graphs sorted by their number of pulses.
        """
        graphs = [g for g in graphs if g.n_pulses > 1]
        graphs.sort(key=lambda x: x.n_pulses)
        batch_list = []

        for minp, maxp in zip(
            [0] + self.batch_splits, self.batch_splits + [1]
        ):
            min_idx = int(minp * len(graphs))
            max_idx = int(maxp * len(graphs))
            this_graphs = graphs[min_idx:max_idx]
            if len(this_graphs) > 0:
                this_batch = Batch.from_data_list(this_graphs)
                batch_list.append(this_batch)
        return batch_list


# @TODO: Remove in favour of DataLoader{,.from_dataset_config}
def make_dataloader(
    db: str,
    pulsemaps: Union[str, List[str]],
    graph_definition: GraphDefinition,
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    shuffle: bool,
    selection: Optional[List[int]] = None,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: Optional[List[str]] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: Optional[List[int]] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
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
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
        index_column=index_column,
        graph_definition=graph_definition,
    )

    # adds custom labels to dataset
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

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
    graph_definition: GraphDefinition,
    selection: Optional[List[int]],
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    database_indices: Optional[List[int]] = None,
    seed: int = 42,
    test_size: float = 0.33,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: Optional[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: Optional[List[int]] = None,
    loss_weight_column: Optional[str] = None,
    loss_weight_table: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and test `DataLoader` instances."""
    # Reproducibility
    rng = np.random.default_rng(seed=seed)
    # Checks(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    if selection is None:
        # If no selection is provided, use all events in dataset.
        dataset: Dataset
        if db.endswith(".db"):
            dataset = SQLiteDataset(
                path=db,
                graph_definition=graph_definition,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                truth_table=truth_table,
                index_column=index_column,
            )
        else:
            dataset = ParquetDataset(
                path=db,
                graph_definition=graph_definition,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                truth_table=truth_table,
                index_column=index_column,
            )
        selection = dataset._get_all_indices()

    # Perform train/validation split
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame(
            {"event_no": selection, "db": database_indices}
        )
        shuffled_df = df_for_shuffle.sample(
            frac=1, replace=False, random_state=rng
        )
        training_df, validation_df = train_test_split(
            shuffled_df, test_size=test_size, random_state=seed
        )
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
    else:
        training_selection, validation_selection = train_test_split(
            selection, test_size=test_size, random_state=seed
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
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_column=loss_weight_column,
        loss_weight_table=loss_weight_table,
        index_column=index_column,
        labels=labels,
        graph_definition=graph_definition,
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
    Logger().info("Results saved at: \n %s" % path)


def save_selection(selection: List[int], file_path: str) -> None:
    """Save the list of event numbers to a CSV file.

    Args:
       selection: List of event ids.
       file_path: File path to save the selection.
    """
    assert isinstance(
        selection, list
    ), "Selection should be a list of integers."
    with open(file_path, "w") as f:
        f.write(",".join(map(str, selection)))
        f.write("\n")
