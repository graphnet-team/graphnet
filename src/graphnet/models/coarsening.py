"""Classes for coarsening operations (i.e., clustering, or local pooling."""

from abc import ABC, abstractmethod

import torch
from torch import LongTensor, Tensor
from torch_geometric.data import Data

from graphnet.components.pool import (
    group_by,
    avg_pool,
    max_pool,
    min_pool,
    sum_pool,
    avg_pool_x,
    max_pool_x,
    min_pool_x,
    sum_pool_x,
    std_pool_x,
)


class Coarsening(ABC):
    """Base class for coarsening operations."""

    # Class variables
    reduce_options = {
        "avg": (avg_pool, avg_pool_x),
        "min": (min_pool, min_pool_x),
        "max": (max_pool, max_pool_x),
        "sum": (sum_pool, sum_pool_x),
    }

    def __init__(
        self,
        reduce: str = "avg",
        transfer_attributes: bool = True,
    ):
        assert reduce in self.reduce_options

        (
            self._reduce_method,
            self._attribute_reduce_method,
        ) = self.reduce_options[reduce]
        self._do_transfer_attributes = transfer_attributes

    @abstractmethod
    def _perform_clustering(self, data: Data) -> LongTensor:
        """Perform clustering of nodes in `data` by assigning unique cluster indices to each."""

    def _additional_features(self, cluster: LongTensor, data: Data) -> Tensor:
        """Additional poolings of feature tensor `x` on `data`.

        By default the nominal `pooling_method` is used for features as well.
        This method can be overwritten for bespoke coarsening operations.
        """
        return None

    def _transfer_attributes(
        self, cluster: LongTensor, original_data: Data, pooled_data: Data
    ) -> Data:
        """Transfer attributes on `original_data` to `pooled_data`."""
        if not self._do_transfer_attributes:
            return pooled_data

        attributes = list(original_data._store.keys())
        for attr in attributes:
            if attr not in pooled_data._store:
                values = getattr(original_data, attr)

                if (
                    isinstance(values, Tensor)
                    and values.size() == original_data.batch.size()
                ):  # Node-level tensor
                    values = self._attribute_reduce_method(
                        cluster, values, original_data.batch
                    )[0]

                setattr(pooled_data, attr, values)

        return pooled_data

    def __call__(self, data: Data) -> Data:
        """Coarsening operation."""
        # Get tensor of cluster indices for each node.
        cluster = self._perform_clustering(data)

        # Check whether a graph has already been built. Otherwise, set a dummy
        # connectivity, as this is required by pooling functions.
        edge_index = data.edge_index
        if edge_index is None:
            data.edge_index = torch.tensor([[]], dtype=torch.int64)

        # Pool `data` object, including `x`, `batch`. and `edge_index`.
        pooled_data = self._reduce_method(cluster, data)

        # Optionally overwrite feature tensor
        x = self._additional_features(cluster, data)
        if x is not None:
            pooled_data.x = torch.cat(
                (
                    pooled_data.x,
                    x,
                ),
                dim=1,
            )

        # Reset `edge_index` if necessary.
        if edge_index is None:
            data.edge_index = edge_index
            pooled_data.edge_index = edge_index

        # Transfer attributes on `data`, pooling as required.
        pooled_data = self._transfer_attributes(cluster, data, pooled_data)

        return pooled_data


class DOMCoarsening(Coarsening):
    def _perform_clustering(self, data: Data) -> LongTensor:
        """Perform clustering of nodes in `data` by assigning unique cluster indices to each."""
        # dom_index = group_pulses_to_dom(data)
        dom_index = group_by(
            data, ["dom_x", "dom_y", "dom_z", "rde", "pmt_area"]
        )
        return dom_index


class CustomDOMCoarsening(DOMCoarsening):
    def _additional_features(self, cluster: LongTensor, data: Data) -> Tensor:
        """Additional poolings of feature tensor `x` on `data`."""
        batch = data.batch

        features = data.features
        if batch is not None:
            features = [feats[0] for feats in features]

        ix_time = features.index("dom_time")
        ix_charge = features.index("charge")

        time = data.x[:, ix_time]
        charge = data.x[:, ix_charge]

        x = torch.stack(
            (
                min_pool_x(cluster, time, batch)[0],
                max_pool_x(cluster, time, batch)[0],
                std_pool_x(cluster, time, batch)[0],
                min_pool_x(cluster, charge, batch)[0],
                max_pool_x(cluster, charge, batch)[0],
                std_pool_x(cluster, charge, batch)[0],
                sum_pool_x(cluster, torch.ones_like(charge), batch)[
                    0
                ],  # Num. nodes (pulses) per cluster (DOM)
            ),
            dim=1,
        )

        return x


class LoopBasedCoarsening:
    def __call__(self, data: Data) -> Data:
        """Coarsening to DOM-level."""
        unique_doms, inverse_idx, n_pulses_pr_dom = torch.unique(
            data.x[:, [0, 1, 2, 5, 6]],
            return_counts=True,
            return_inverse=True,
            dim=0,
        )
        unique_inverse_indices = torch.unique(inverse_idx)
        count = 0
        pulse_statistics = torch.zeros(size=(len(unique_doms), 8))

        # 'dom_x','dom_y','dom_z','dom_time','charge','rde','pmt_area'
        for unique_inverse_idx in unique_inverse_indices:
            time = data.x[inverse_idx == unique_inverse_idx, 3]
            charge = data.x[inverse_idx == unique_inverse_idx, 4]
            pulse_statistics[count, 0] = torch.min(time)
            pulse_statistics[count, 1] = torch.mean(time)
            pulse_statistics[count, 2] = torch.max(time)
            pulse_statistics[count, 3] = torch.std(time, unbiased=False)
            pulse_statistics[count, 4] = torch.min(charge)
            pulse_statistics[count, 5] = torch.mean(charge)
            pulse_statistics[count, 6] = torch.max(charge)
            pulse_statistics[count, 7] = torch.std(charge, unbiased=False)
            count += 1

        data = data.clone()  # @TODO: To avoid modifying in-place?
        data.x = torch.cat(
            (unique_doms, n_pulses_pr_dom.unsqueeze(1), pulse_statistics),
            dim=1,
        )
        return data
