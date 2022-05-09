import logging
import time

from timer import timer
import torch.multiprocessing
import torch.utils.data
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite_dataset import SQLiteDataset

# Configurations
timer.set_level(logging.INFO)
logging.basicConfig(level=logging.INFO)
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

db = "/groups/icecube/asogaard/temp/sqlite_test_upgrade/data_test/data/data_test.db"
pulsemap = "I3RecoPulseSeriesMapRFCleaned_mDOM"
batch_size = 1024
num_workers = 30
wait_time = 0.00  # sec.

# Common variables
dataset = SQLiteDataset(
    db,
    pulsemap,
    features,
    truth,
)

print(dataset[1])
print(dataset[1].x)
dataset.close_connection()  # This is necessary iff `dataset` has been indexed between instantiation and passing to `DataLoader`

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=Batch.from_data_list,
    persistent_workers=True,
    prefetch_factor=2,
)

with timer("torch dataloader"):
    for batch in tqdm(dataloader, unit=" batches", colour="green"):
        time.sleep(wait_time)

print(batch)
print(batch.size(), batch.num_graphs)
