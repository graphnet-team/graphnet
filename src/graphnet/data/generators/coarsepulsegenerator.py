"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, OrderedDict, List, Optional
from graphnet.data.generators import Generator

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import warn_once

from torch import Tensor, unique

from sklearn.cluster import KMeans

import numpy as np

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
    )  # pyright: reportMissingImports=false


class CoarsePulseGenerator(Generator):
    def __init__(
        self,
        pulsemap: str,
        name: str,
        method: str,
        coarsen_on: Optional[List[str]] = None,
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
            keep_columns: List of pulsemap columns to keep in final coarse pulsemap.
        """
        # Member variable(s)
        self._pulsemap = pulsemap
        # reduction method
        self._method = method
        # pseudo-distance to use for coarsening
        self._coarsen_on = coarsen_on
        # target reduction factor ie. 1/reduc
        self.reduc = reduc
        # minimum number of pulses to keep
        self.min_n = min_n
        # columns to keep in final coarse pulsemap
        self._keep_columns = keep_columns

        # set coarsen_on if not specified
        if coarsen_on == None:
            self._coarsen_on = [
                "dom_x",
                "dom_y",
                "dom_z",
            ]

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
        self._pulse_names = self._pulse_data.keys()
        # Get keep columns
        if self._keep_columns is None:
            self._keep_columns = self._pulse_names
        # Get time index
        self._time_index = [i for i in self._pulse_data if "time" in i]
        # Get charge index
        self._charge_index = [i for i in self._pulse_data if "charge" in i]
        # Get coarse pulse series
        coarse_pulse_data = self.get_coarse_pulse_data(self._method)
        # return coarsened pulse series
        return OrderedDict(self._pulsemap + "_coarse", coarse_pulse_data)

    def get_coarse_pulse_data(self) -> any:
        """Get coarse pulse series.

        Args:
            pulse_data: Pulse series to coarsen.
        Returns:
            coarse_pulse_data: Coarsened pulse series.
        """
        # Get coarse pulse series
        data_CP = self._pulse_data.deepcopy()
        # change time into  distance using speed of light in ice.
        data_CP[self._time_index] = (
            data_CP[self._time_index] * 2.3 * 10 ** (-1)
        )
        # Take the spatial + time (transformed) values and use those for the coarsening algorithm
        tensor = Tensor(data_CP[self._coarsen_on + self._time_index]).T
        reduc = min([len(tensor), reduc])
        min_n = int(len(tensor) / min_n)
        # reduce by factor 100 ensuring not to   reduce below min red (unless less dom activations in event)
        n_clusters = max([reduc, min_n])
        if self._method == "Kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters, random_state=10, init="random", n_init=1
            )
        else:
            raise ValueError("Method not implemented")

        index = clusterer.fit_predict(tensor)

        data_with_group = np.vstack([index, self._pulse_data])
        data_with_group = data_with_group.T[data_with_group[0, :].argsort()]
        data_grouped = np.array(
            np.split(
                data_with_group[:, 1:],
                np.unique(data_with_group[:, 0], return_index=True)[1][1:],
            ),
            dtype=object,
        )
        # mget mean of grouped data and multiply charge by number of pulses in group.
        for self._pulse_data, i in zip(data_grouped, range(len(data_grouped))):
            counter = np.shape(self._pulse_data)[0]
            data_grouped[i] = np.mean(self._pulse_data, axis=0)
            data_grouped[i][self._charge_index] = (
                data_grouped[i][self._charge_index] * counter
            )
        if len(np.shape(np.array(list(data_grouped)).T)) == 3:
            data_grouped = data_grouped[:, 0, :]

        result = np.array(list(data_grouped)).T
        result = OrderedDict(result.T, columns=self._pulse_names)[
            self._keep_columns
        ]

        return result

    def get_time_index(self):
        """Get time index.

        Args:
            None
        Returns:
            time_index: Index of time feature.
        """
        time_index = "dom_time"
        return time_index
