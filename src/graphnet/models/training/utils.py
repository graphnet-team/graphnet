from collections import OrderedDict
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch

from tqdm import tqdm

from graphnet.data.sqlite_dataset import SQLiteDataset
from graphnet.models import Model


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
) -> DataLoader:

    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = SQLiteDataset(
        db,
        pulsemaps,
        features,
        truth,
        selection=selection,
    )

    def collate_fn(graphs):
        # Remove graphs with less than two DOM hits. Should not occur in "production."
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
) -> Tuple[DataLoader]:
    
    # Reproducibility
    rng = np.random.RandomState(seed=seed)

    # Checks(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    # Perform train/validation split
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame({'event_no': selection, 'db': database_indices})
        shuffled_df = df_for_shuffle.sample(frac=1, replace=False, random_state=rng)
        training_df, validation_df = train_test_split(shuffled_df, test_size=test_size, random_state=rng)
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
    else:
        training_selection, validation_selection = train_test_split(selection, test_size=test_size, random_state=rng)

    # Create DataLoaders
    common_kwargs = dict(
        db=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    training_dataloader = make_dataloader(
        shuffle=True,
        selection=training_selection,
        **common_kwargs,
    )

    validation_dataloader = make_dataloader(
        shuffle=False,
        selection=validation_selection,
        **common_kwargs,
    )

    return training_dataloader, validation_dataloader  # , {'valid_selection':validation_selection, 'training_selection':training_selection}

def get_predictions(trainer, model, dataloader, prediction_columns, additional_attributes=None):
    # Check(s)
    if additional_attributes is None:
        additional_attributes = []
    assert isinstance(additional_attributes, list)

    # Set model to inference mode
    model.inference()

    # Get predictions
    predictions_torch = trainer.predict(model, dataloader)
    predictions = [p[0].detach().cpu().numpy() for p in predictions_torch]  # Assuming single task
    predictions = np.concatenate(predictions, axis=0)
    try:
        assert len(prediction_columns) == predictions.shape[1]
    except IndexError:
        predictions = predictions.reshape((-1, 1))
        assert len(prediction_columns) == predictions.shape[1]


    # Get additional attributes
    attributes = OrderedDict([(attr, []) for attr in additional_attributes])

    for batch in dataloader:
        for attr in attributes:
            attribute = batch[attr].detach().cpu().numpy()
            if attr == 'event_no':
                attribute = np.repeat(attribute, batch['n_pulses'].detach().cpu().numpy())
            attributes[attr].extend(attribute)

    data = np.concatenate([predictions] + [
        np.asarray(values)[:, np.newaxis] for values in attributes.values()
    ], axis=1)

    results = pd.DataFrame(data, columns=prediction_columns + additional_attributes)
    return results

def save_results(db, tag, results, archive,model):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    os.makedirs(path, exist_ok = True)
    results.to_csv(path + '/results.csv')
    torch.save(model.cpu().state_dict(), path + '/' + tag + '.pth')
    print('Results saved at: \n %s'%path)

def load_model(db, tag, archive, detector, gnn, task, device):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device=device,
    )
    model.load_state_dict(torch.load(path + '/' + tag + '.pth'))
    model.eval()
    return model
