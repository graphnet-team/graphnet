"""Base `Dataloader` class(es) used in `graphnet`."""

from typing import Any, Callable, List

import torch.utils.data
from torch_geometric.data import Batch, Data

from graphnet.data.dataset import Dataset


def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


class DataLoader(torch.utils.data.DataLoader):
    """Class for loading data from a `Dataset`."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 10,
        persistent_workers: bool = True,
        collate_fn: Callable = collate_fn,
        **kwargs: Any,
    ) -> None:
        """Construct `DataLoader`."""
        # Base class constructor
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
            **kwargs,
        )
