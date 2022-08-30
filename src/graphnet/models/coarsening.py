"""Classes for coarsening operations (i.e., clustering, or local pooling."""

from abc import ABC, abstractmethod
from multiprocessing import pool
from typing import List, Optional, Union
from copy import deepcopy
import torch
from torch import LongTensor, Tensor
from torch_geometric.data import Data, Batch
from sklearn.cluster import DBSCAN

# from torch_geometric.utils import unbatch_edge_index
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
    time_window_clustering,
)
from graphnet.utilities.logging import LoggerMixin

# Utility method(s)
from torch_geometric.utils import degree

# NOTE: From [https://github.com/pyg-team/pytorch_geometric/pull/4903]
# TODO: Remove once bumping to torch_geometric>=2.1.0
#       See [https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md]


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.
    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


class Coarsening(ABC, LoggerMixin):
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
    def _perform_clustering(self, data: Union[Data, Batch]) -> LongTensor:
        """Perform clustering of nodes in `data` by assigning unique cluster indices to each."""

    def _additional_features(self, cluster: LongTensor, data: Batch) -> Tensor:
        """Additional poolings of feature tensor `x` on `data`.

        By default the nominal `pooling_method` is used for features as well.
        This method can be overwritten for bespoke coarsening operations.
        """

    def _transfer_attributes(
        self, cluster: LongTensor, original_data: Batch, pooled_data: Batch
    ) -> Batch:
        """Transfer attributes on `original_data` to `pooled_data`."""
        # Check(s)
        if not self._do_transfer_attributes:
            return pooled_data

        attributes = list(original_data._store.keys())
        batch: Optional[LongTensor] = original_data.batch
        for ix, attr in enumerate(attributes):
            if attr not in pooled_data._store:
                values = getattr(original_data, attr)

                attr_is_node_level_tensor = False
                if isinstance(values, Tensor):
                    if batch is None:
                        attr_is_node_level_tensor = (
                            values.dim() > 1 or values.size(dim=0) > 1
                        )
                    else:
                        attr_is_node_level_tensor = (
                            values.size() == original_data.batch.size()
                        )

                if attr_is_node_level_tensor:
                    values: Tensor = self._attribute_reduce_method(
                        cluster,
                        values,
                        batch=torch.zeros_like(values, dtype=torch.int32),
                    )[0]

                setattr(pooled_data, attr, values)

        return pooled_data

    def __call__(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        """Coarsening operation."""

        # Get tensor of cluster indices for each node.
        cluster: LongTensor = self._perform_clustering(data)

        # Check whether a graph has already been built. Otherwise, set a dummy
        # connectivity, as this is required by pooling functions.
        edge_index = data.edge_index
        if edge_index is None:
            data.edge_index = torch.tensor([[]], dtype=torch.int64)

        # Pool `data` object, including `x`, `batch`. and `edge_index`.
        pooled_data: Batch = self._reduce_method(cluster, data)

        # Optionally overwrite feature tensor
        add_feats = self._additional_features(cluster, data)
        if add_feats is not None:
            x, add_labels = add_feats
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
        if add_feats is not None:
            pooled_data.features = [feats + add_labels for feats in pooled_data.features]

        # Reconstruct Batch Attributes
        if isinstance(data, Batch):  # if a Batch object
            pooled_data = self._reconstruct_batch(data, pooled_data)

        return pooled_data

    def _reconstruct_batch(self, original, pooled):
        pooled = self._add_slice_dict(original, pooled)
        pooled = self._add_inc_dict(original, pooled)
        return pooled

    def _add_slice_dict(self, original, pooled):
        # Copy original slice_dict and count nodes in each graph in pooled batch
        slice_dict = deepcopy(original._slice_dict)
        _, counts = torch.unique_consecutive(pooled.batch, return_counts=True)
        # Reconstruct the entry in slice_dict for pulsemaps - only these are affected by pooling
        pulsemap_slice = [0]
        for i in range(len(counts)):
            pulsemap_slice.append(pulsemap_slice[i] + counts[i].item())

        # Identifies pulsemap entries in slice_dict and set them to pulsemap_slice
        for field in slice_dict.keys():
            if (original._num_graphs) == slice_dict[field][-1]:
                pass  # not pulsemap, so skip
            else:
                slice_dict[field] = pulsemap_slice
        pooled._slice_dict = slice_dict
        return pooled

    def _add_inc_dict(self, original, pooled):
        # not changed by coarsening
        pooled._inc_dict = deepcopy(original._inc_dict)
        return pooled


class DOMCoarsening(Coarsening):
    def _perform_clustering(self, data: Union[Data, Batch]) -> LongTensor:
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
            features = features[0]

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

        additional_feature_labels = \
            [
                "min_pool_time",
                "max_pool_time",
                "std_pool_time",
                "min_pool_charge",
                "max_pool_charge",
                "std_pool_charge",
                "pulses_per_dom"
            ]

        return x, additional_feature_labels


class TimeWindowCoarsening(Coarsening):
    def __init__(self, time_window: float, reduce: str = "avg", transfer_attributes: bool = True,):
        super().__init__(reduce, transfer_attributes)
        self._time_window = time_window
        self._cluster_method = DBSCAN(self._time_window, min_samples=1)

    def _perform_clustering(self, data: Union[Data, Batch]) -> LongTensor:
        time_clusters = time_window_clustering(data, self._cluster_method, self._time_window)
        return time_clusters


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
