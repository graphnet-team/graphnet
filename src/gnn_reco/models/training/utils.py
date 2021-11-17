import os
from typing import List, Optional, Tuple

import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch

from gnn_reco.data.sqlite_dataset import SQLiteDataset
from gnn_reco.models import Model

def make_dataloader(
    db: str,
    pulsemap: str,
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    shuffle: bool,
    selection: List[int] = None,
    num_workers: int = 10,
    persistent_workers: bool = True,
) -> DataLoader:
    
    dataset = SQLiteDataset(
        db,
        pulsemap,
        features,
        truth,
        selection=selection,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader

def make_train_validation_dataloader(
    db: str,
    selection: List[int],
    pulsemap: str,
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
        pulsemap=pulsemap,
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
