"""Utility functions for construction of graphs."""

from typing import List, Tuple, Optional, Union
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from graphnet.constants import DATA_DIR


def lex_sort(x: np.array, cluster_columns: List[int]) -> np.ndarray:
    """Sort numpy arrays according to columns on ´cluster_columns´.

    Note that `x` is sorted along the dimensions in `cluster_columns`
    backwards. I.e. `cluster_columns = [0,1,2]`
    means `x` is sorted along `[2,1,0]`.

    Args:
        x: array to be sorted.
        cluster_columns: Columns of `x` to be sorted along.

    Returns:
        A sorted version of `x`.
    """
    tmp_list = []
    for cluster_column in cluster_columns:
        tmp_list.append(x[:, cluster_column])
    return x[np.lexsort(tuple(tmp_list)), :]


def gather_cluster_sequence(
    x: np.ndarray, feature_idx: int, cluster_columns: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn `x` into rows of clusters with sequences along columns.

    Sequences along columns are added which correspond to
    gathered sequences of the feature in `x` specified by column index
    `feature_idx` associated with each column. Sequences are padded with NaN to
    be of same length. Dimension of clustered array is `[n_clusters, l +
    len(cluster_columns)]`,where l is the largest sequence length.

    **Example**:
    Suppose `x` represents a neutrino event and we have chosen to cluster on
    the PMT positions and that `feature_idx` correspond to pulse time.

    The resulting array will have dimensions `[n_pmts, m + 3]` where `m` is the
    maximum number of same-pmt pulses found in `x`, and `+3`for the three
    spatial directions  defining each cluster.

    Args:
        x:  Array for clustering
        feature_idx: Index of the feature in `x` to
        be gathered for each cluster.
        cluster_columns: Index in `x` from which to build clusters.

    Returns:
        array: Array with dimensions  `[n_clusters, l + len(cluster_columns)]`
        column_offset: Indices of the columns in `array` that defines clusters.
    """
    # sort pulses according to cluster columns
    x = lex_sort(x=x, cluster_columns=cluster_columns)

    # Calculate clusters and counts
    unique_sensors, counts = np.unique(
        x[:, cluster_columns], return_counts=True, axis=0
    )
    # sort DOMs and pulse-counts
    sensor_counts = counts.reshape(-1, 1)
    contingency_table = np.concatenate([unique_sensors, sensor_counts], axis=1)
    sensors_in_contingency_table = np.arange(0, unique_sensors.shape[1], 1)
    contingency_table = lex_sort(
        x=contingency_table, cluster_columns=sensors_in_contingency_table
    )
    unique_sensors = contingency_table[:, 0 : unique_sensors.shape[1]]
    count_part = contingency_table[:, unique_sensors.shape[1] :]
    flattened_counts = count_part.flatten()
    counts = flattened_counts.astype(int)

    # Pad unique sensor columns with NaN's up until the maximum number of
    # Same pmt-pulses. Each of padded columns represents a pulse.
    pad = np.empty((unique_sensors.shape[0], max(counts)))
    pad[:] = np.nan
    array = np.concatenate([unique_sensors, pad], axis=1)
    column_offset = unique_sensors.shape[1]

    # Construct indices for loop
    cumsum = np.zeros(len(np.cumsum(counts)) + 1)
    cumsum[0] = 0
    cumsum[1:] = np.cumsum(counts)
    cumsum = cumsum.astype(int)

    # Insert pulse attribute in place of NaN.
    for k in range(len(counts)):
        array[k, column_offset : (column_offset + counts[k])] = x[
            cumsum[k] : cumsum[k + 1], feature_idx
        ]
    return array, column_offset, counts


def identify_indices(
    feature_names: List[str], cluster_on: List[str]
) -> Tuple[List[int], List[int], List[str]]:
    """Identify indices for clustering and summarization."""
    features_for_summarization = []
    for feature in feature_names:
        if feature not in cluster_on:
            features_for_summarization.append(feature)
    cluster_indices = [feature_names.index(column) for column in cluster_on]
    summarization_indices = [
        feature_names.index(column) for column in features_for_summarization
    ]
    return cluster_indices, summarization_indices, features_for_summarization


class cluster_and_pad:
    """Cluster and pad the data for further summarization.

    Clusters the inptut data according to the specified columns
    and computes aggregate statistics on the clusters.
    The clustering will happen only ones creating a cluster matrix
    which will hold all the aggregated statistics and a padded matrix which
    will hold the padded data for quick calculation of aggregate statistics.

    Example:
    clustered_x = cluster_and_pad(x = single_event_as_array,
                                 cluster_columns = [0,1,2])
    # Creates a cluster matrix and a padded matrix,
    # the cluster matrix will contain the unique values of the cluster columns,
    # no additional aggregate statistics are added yet.

    clustered_x_with_percentiles = cluster_class.add_percentile_summary(
    summarization_indices = [3,4,5], percentiles = [10,50,90])
    # Adds the 10th, 50th and 90th percentile of columns 3,4
    # and 5 in the input data to the cluster matrix.

    clustered_x_with_percentiles_and_std = cluster_class.add_std(column = 4)
    # Adds the standard deviation of column 4 in the input data
    # to the cluster matrix.
    """

    def __init__(self, x: np.ndarray, cluster_columns: List[int]) -> None:
        """Initialize the class with the data and cluster columns.

        Args:
            x: Array to be clustered
            cluster_columns: List of column indices on which the clusters
                            are constructed.
            Returns: None
            Adds:
                clustered_x: Added to the class
                _counts: Added to the class
                _padded_x: Added to the class
        """
        x = lex_sort(x=x, cluster_columns=cluster_columns)

        unique_sensors, self._counts = np.unique(
            x[:, cluster_columns], axis=0, return_counts=True
        )

        contingency_table = np.concatenate(
            [unique_sensors, self._counts.reshape(-1, 1)], axis=1
        )

        contingency_table = lex_sort(
            x=contingency_table, cluster_columns=cluster_columns
        )

        self.clustered_x = contingency_table[:, 0 : unique_sensors.shape[1]]
        self._counts = (
            contingency_table[:, self.clustered_x.shape[1] :]
            .flatten()
            .astype(int)
        )

        self._padded_x = np.empty(
            (len(self._counts), max(self._counts), x.shape[1])
        )
        self._padded_x.fill(np.nan)

        for i in range(len(self._counts)):
            self._padded_x[i, : self._counts[i]] = x[: self._counts[i]]
            x = x[self._counts[i] :]
        return self.clustered_x

    def _add_column(
        self, column: np.ndarray, location: Optional[int] = None
    ) -> None:
        """Add a column to the clustered tensor.

        Args:
            column: Column to be added to the tensor
            location: Location to insert the column in the clustered tensor
        Returns:
            clustered_x: The clustered tensor with the column added
        """
        if location is None:
            self.clustered_x = np.column_stack([self.clustered_x, column])
        else:
            self.clustered_x = np.insert(
                self.clustered_x, location, column, axis=1
            )

    def _calculate_charge_sum(self, charge_index: int) -> np.ndarray:
        """Calculate the sum of the charge."""
        assert not hasattr(
            self, "_charge_sum"
        ), "Charge sum has already been calculated, \
            re-calculation is not allowed"
        self._charge_sum = self._padded_x[:, :, charge_index].sum(axis=1)
        return self._charge_sum

    def _calculate_charge_weights(self, charge_index: int) -> np.ndarray:
        """Calculate the weights of the charge."""
        assert not hasattr(
            self, "_charge_weights"
        ), "Charge weights have already been calculated, \
            re-calculation is not allowed"
        assert hasattr(
            self, "_charge_sum"
        ), "Charge sum has not been calculated, \
            please run calculate_charge_sum"
        self._charge_weights = (
            self._padded_x[:, :, charge_index]
            / self._charge_sum[:, np.newaxis]
        )
        return self._charge_weights

    def add_charge_threshold_summary(
        self,
        summarization_indices: List[int],
        percentiles: List[int],
        charge_index: int,
        location: Optional[int] = None,
    ) -> np.ndarray:
        """Summarize features through percentiles on charge of sensor.

        Args:
            summarization_indices: List of column indices that defines features
                                   that will be summarized with percentiles.
            percentiles: percentiles used to summarize `x`. E.g. [10,50,90].
            charge_index: index of the charge column in the padded tensor
            location: Location to insert the summarization indices in the
                      clustered tensor defaults to adding at the end
        Returns:
            clustered_x: The clustered tensor with the summarization indices
                         added
        Adds:
            _charge_sum: Added to the class
            _charge_weights: Added to the class
        Altered:
            _padded_x: Charge is altered to be the cumulative sum
                       of the charge divided by the total charge
            clustered_x: The summarization indices are added at the end
                         of the tensor or inserted at the specified location.
        """
        # convert the charge to the cumulative sum of the charge divided
        # by the total charge
        self._calculate_charge_sum(charge_index)
        self._calculate_charge_weights(charge_index)

        self._padded_x[:, :, charge_index] = (
            self._padded_x[:, :, charge_index]
            / self._charge_sum[:, np.newaxis]
        )

        # Summarize the charge at different percentiles
        selections = np.argmax(
            self._padded_x[:, :, charge_index][:, :, np.newaxis]
            >= (np.array(percentiles) / 100),
            axis=1,
        )

        selections += (np.arange(len(self._counts)) * self._padded_x.shape[1])[
            :, np.newaxis
        ]

        selections = self._padded_x[:, :, summarization_indices].reshape(
            -1, len(summarization_indices)
        )[selections]
        selections = selections.transpose(0, 2, 1).reshape(
            len(self.clustered_x), -1
        )
        self._add_column(selections, location)
        return self.clustered_x

    def add_percentile_summary(
        self,
        summarization_indices: List[int],
        percentiles: List[int],
        method: str = "linear",
        location: Optional[int] = None,
    ) -> np.ndarray:
        """Summarize the features of the sensors using percentiles.

        Args:
            summarization_indices: List of column indices that defines features
                                    that will be summarized with percentiles.
            percentiles: percentiles used to summarize `x`. E.g. [10,50,90].
            method: Method to summarize the features. E.g. "linear"
            location: Location to insert the summarization indices in the
                       clustered tensor defaults to adding at the end
        Returns:
            None
        Adds:
            None
        Altered:
            clustered_x: The summarization indices are added at the end of
                         the tensor or inserted at the specified location
        """
        percentiles_x = np.nanpercentile(
            self._padded_x[:, :, summarization_indices],
            percentiles,
            axis=1,
            method=method,
        )

        percentiles_x = percentiles_x.transpose(1, 2, 0).reshape(
            len(self.clustered_x), -1
        )
        self._add_column(percentiles_x, location)
        return self.clustered_x

    def add_counts(self, location: Optional[int] = None) -> np.ndarray:
        """Add the counts of the sensor to the summarization features."""
        self._add_column(np.log10(self._counts), location)
        return self.clustered_x

    def add_sum_charge(self, location: Optional[int] = None) -> np.ndarray:
        """Add the sum of the charge to the summarization features."""
        assert hasattr(
            self, "_charge_sum"
        ), "Charge sum has not been calculated, \
            please run calculate_charge_sum"
        self._add_column(self._charge_sum, location)
        return self.clustered_x

    def add_std(
        self,
        column: int,
        location: Optional[int] = None,
        weights: Union[np.ndarray, int] = 1,
    ) -> np.ndarray:
        """Add the standard deviation of the column.

        Args:
            column: Index of the column in the padded tensor to
                    calculate the standard deviation
            location: Location to insert the standard deviation in the
                      clustered tensor defaults to adding at the end
            weights: Optional weights to be applied to the standard deviation
        """
        self._add_column(
            np.nanstd(self._padded_x[:, :, column] * weights, axis=1), location
        )
        return self.clustered_x

    def add_mean(
        self,
        column: int,
        location: Optional[int] = None,
        weights: Union[np.ndarray, int] = 1,
    ) -> np.ndarray:
        """Add the mean of the column."""
        self._add_column(
            np.nanmean(self._padded_x[:, :, column] * weights, axis=1),
            location,
        )
        return self.clustered_x


def ice_transparency(
    z_offset: Optional[float] = None, z_scaling: Optional[float] = None
) -> Tuple[interp1d, interp1d]:
    """Return interpolation functions for optical properties of IceCube.

        NOTE: The resulting interpolation functions assumes that the
        Z-coordinate of pulse are scaled as `z = z/500`.
        Any deviation from this scaling method results in inaccurate results.

    Args:
        z_offset: Offset to be added to the depth of the DOM.
        z_scaling: Scaling factor to be applied to the depth of the DOM.

    Returns:
        f_scattering: Function that takes a normalized depth and returns the
        corresponding normalized scattering length.
        f_absorption: Function that takes a normalized depth and returns the
        corresponding normalized absorption length.
    """
    # Data from page 31 of https://arxiv.org/pdf/1301.5361.pdf
    df = pd.read_parquet(
        os.path.join(DATA_DIR, "ice_properties/ice_transparency.parquet"),
    )

    z_offset = z_offset or -1950.0
    z_scaling = z_scaling or 500.0

    df["z_norm"] = (df["depth"] + z_offset) / z_scaling
    df[["scattering_len_norm", "absorption_len_norm"]] = (
        RobustScaler().fit_transform(df[["scattering_len", "absorption_len"]])
    )

    f_scattering = interp1d(df["z_norm"], df["scattering_len_norm"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len_norm"])
    return f_scattering, f_absorption
