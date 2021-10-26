import os

from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data.batch import Batch

from gnn_reco.data.sqlite_dataset import SQLiteDataset

def make_train_validation_dataloader(db, selection, pulsemap, batch_size, features, truth, num_workers, persistent_workers=True):
    training_selection, validation_selection = train_test_split(selection, test_size=0.33, random_state=42)

    common_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )
    
    training_dataset = SQLiteDataset(db, pulsemap, features, truth, selection=training_selection)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, **common_kwargs)

    validation_dataset = SQLiteDataset(db, pulsemap, features, truth, selection=validation_selection)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, **common_kwargs)

    return training_dataloader, validation_dataloader

def save_results(db, tag, results, archive,model):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    os.makedirs(path, exist_ok = True)
    results.to_csv(path + '/results.csv')
    torch.save(model.cpu(), path + '/' + tag + '.pkl')
    print('Results saved at: \n %s'%path)
