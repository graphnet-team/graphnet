"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, OrderedDict, List, Optional
from graphnet.data.generators import Generator

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import warn_once

from copy import deepcopy

from torch import Tensor, unique

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
        reduc: int = 100,
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
            reduc: Target reduction factor for coarse pulsemap.
            min_n: Minimum number of pulses to keep in coarse pulsemap.
        """
        # Member variable(s)
        self._pulsemap = pulsemap
        # reduction method
        self._method = method
        # pseudo-distance to use for coarsening force time to be included at end.
        self._coarsen_on = coarsen_on + [time_label]
        # target reduction factor ie. 1/reduc
        self._reduc = reduc
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

        Args:
            pulse_data: Pulse series to coarsen.

        Returns:
            coarse_pulse_data: Coarsened pulse series.
        """
        # get index values of columns to keep.
        if self._keep_columns is not None:
            keep_index = [
                self._pulse_names.index(i) for i in self._keep_columns
            ]

        data_CP = []
        for i in self._coarsen_on:
            data_CP.append(np.array(self._pulse_data[i]))
        data_CP = np.array(data_CP)
        # Get coarse pulse series
        # change time into  distance using speed of light in ice.
        data_CP[-1] = data_CP[-1] * 2.3 * 10 ** (-1)
        # Take the spatial + time (transformed) values and use those for the coarsening algorithm
        tensor = Tensor(data_CP).T
        min_n = min([len(tensor), self._min_n])
        reduc = int(len(tensor) / self._reduc)
        # reduce by factor 100 ensuring not to   reduce below min red (unless less dom activations in event)
        n_clusters = max([reduc, min_n])
        if len(tensor) > self._min_n:
            if self._method == "Kmeans":
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

        pulse_df = pd.DataFrame(self._pulse_data, index=None).T
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
            data_grouped[ind][self._charge_index] = (
                data_grouped[ind][self._charge_index] * counter
            )
        if len(np.shape(np.array(list(data_grouped)).T)) == 3:
            data_grouped = data_grouped[:, 0, :]

        result = np.array(list(data_grouped)).T
        # turn the np array of np arrays into a list of lists
        result = [list(i) for i in result[keep_index]]
        # write to dict
        result = dict(
            zip(
                list(np.array(self._pulse_names)[keep_index]),
                result,
            )
        )

        return result
