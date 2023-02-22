"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from graphnet.data.generators import Generator

from graphnet.utilities.imports import has_icecube_package

from torch import Tensor

from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
    )  # pyright: reportMissingImports=false


class CoarsePulseGenerator(Generator):
    """Generator for producing coarse pulsemaps from I3PulseSeriesMaps."""

    def __init__(
        self,
        pulsemap: str,
        name: str,
        method: str,
        coarsen_on: List[str] = [
            "dom_x",
            "dom_y",
            "dom_z",
        ],
        time_label: str = "dom_time",
        keep_columns: Optional[List[str]] = None,
        reduce: int = 100,
        min_n: int = 25,
    ):
        """Construct CoarsePulsemapGenerator.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            name: Name of the `Generator` instance.
            method: Method to use for coarse pulsemap generation.
            coarsen_on: List of pulsemap columns to use for pseudo-distance used by coarsening algorithm.
            time_label: Name of the time column in the pulsemap.
            keep_columns: List of pulsemap columns to keep in final coarse pulsemap.
            reduce: Target reduction factor for coarse pulsemap.
            min_n: Minimum number of pulses to keep in coarse pulsemap.
        """
        # Member variable(s)
        self._pulsemap = pulsemap
        # reduction method
        self._method = method
        # pseudo-distance to use for coarsening force time to be included at end.
        self._coarsen_on = coarsen_on + [time_label]
        # target reduction factor ie. 1/reduce
        self._reduce = reduce
        # minimum number of pulses to keep
        self._min_n = min_n
        # columns to keep in final coarse pulsemap
        self._keep_columns = keep_columns

        # Base class constructor
        super().__init__(name)

    def __call__(self, data: Dict[str, Any]) -> dict:
        """Extract reconstructed features from `frame`.

        Args:
            data: Ordered dictionary generated from physics I3 frame.

        Returns:
            data:
        """
        # Get pulse series
        self._pulse_data = data[self._pulsemap]
        # get feature keys
        self._pulse_names = list(self._pulse_data.keys())

        # Get keep columns
        if self._keep_columns is None:
            self._keep_columns = self._pulse_names
        # Get charge index
        self._charge_index = [
            self._pulse_names.index(i)
            for i in self._pulse_names
            if "charge" in i
        ]
        # Get coarse pulse series
        coarse_pulse_data = self.get_coarse_pulse_data()
        # return coarsened pulse series
        return {self._name: coarse_pulse_data}

    def get_coarse_pulse_data(self) -> dict:
        """Get coarse pulse series.

        Returns:
            coarse_pulse_data: Coarsened pulse series.
        """
        # get index values for grouping
        index = coarsening_index(
            pulse_data=self._pulse_data,
            coarsen_on=self._coarsen_on,
            reduce=self._reduce,
            min_n=self._min_n,
            method=self._method,
        )
        # group pulses by index
        coarse_pulse_data = group_by_index(
            pulse_data=self._pulse_data,
            index=index,
            pulse_names=self._pulse_names,
            charge_index=self._charge_index,
            keep_columns=self._keep_columns,
        )

        return coarse_pulse_data


def coarsening_index(
    pulse_data: dict,
    coarsen_on: List[str],
    reduce: int,
    min_n: int,
    method: str,
) -> np.array:
    """Get coarsening index.

    Args:
        pulse_data: Pulse series to coarsen.
        coarsen_on: List of pulsemap columns to use for pseudo-distance used by coarsening algorithm, time assumed included as last entry.
        reduce: Target reduction factor for coarse pulsemap.
        min_n: Minimum number of pulses to keep in coarse pulsemap.
        method: Method to use for coarse pulsemap generation.

    Returns:
        index: Index list for grouping.
    """
    data = []
    for i in coarsen_on:
        data.append(np.array(pulse_data[i]))
    data = np.array(data)
    # Get coarse pulse series
    # change time into  distance using speed of light in ice.
    data[-1] = data[-1] * 2.3 * 10 ** (-1)
    # Take the spatial + time (transformed) values and use those for the coarsening algorithm
    tensor = Tensor(data).T
    min_n = min([len(tensor), min_n])
    reduce = int(len(tensor) / reduce)
    # reduce by factor 100 ensuring not to   reduce below min red (unless less dom activations in event)
    n_clusters = max([reduce, min_n])
    if len(tensor) > min_n:
        if method == "Kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=10,
                init="random",
                n_init=1,
            )
        else:
            raise ValueError("Method not implemented")

        index = clusterer.fit_predict(tensor)
    else:  # if less dom activations than clusters, just return the doms
        index = np.arange(len(tensor))

    return index


def group_by_index(
    pulse_data: dict,
    index: List[int],
    pulse_names: List[str],
    charge_index: Optional[List[int]] = None,
    keep_columns: Optional[List[str]] = None,
) -> dict:
    """Group pulses by given grouping index.

    Args:
        pulse_data: Pulse series to group.
        index: Index list for grouping.
        pulse_names: List of pulsemap columns.
        charge_index: Index of charge column.
        keep_columns: List of pulsemap columns to keep in final coarse pulsemap.

    Returns:
        result: Pulsemap grouped by input index.
    """
    pulse_df = pd.DataFrame(pulse_data, index=None).T
    data_with_group = np.vstack([index, pulse_df])
    data_with_group = data_with_group.T[data_with_group[0, :].argsort()]
    data_grouped = np.array(
        np.split(
            data_with_group[:, 1:],
            np.unique(data_with_group[:, 0], return_index=True)[1][1:],
        ),
        dtype=object,
    )

    # get mean of grouped data and multiply charge by number of pulses in group.
    for data, ind in zip(data_grouped, range(len(data_grouped))):
        counter = np.shape(data)[0]
        data_grouped[ind] = np.mean(data, axis=0)
        if charge_index is not None:
            data_grouped[ind][charge_index] = (
                data_grouped[ind][charge_index] * counter
            )
    if len(np.shape(np.array(list(data_grouped)).T)) == 3:
        data_grouped = data_grouped[:, 0, :]

    result = np.array(list(data_grouped)).T
    # turn the np array of np arrays into a list of lists

    # get index values of columns to keep, and keep only those columns.
    if keep_columns is not None:
        keep_index = [pulse_names.index(i) for i in keep_columns]
        result = [list(i) for i in result[keep_index]]
        result = dict(
            zip(
                list(np.array(pulse_names)[keep_index]),
                result,
            )
        )
    else:  # if no keep columns specified, keep all columns.
        result = [list(i) for i in result]
        result = dict(
            zip(
                list(pulse_names),
                result,
            )
        )

    return result
