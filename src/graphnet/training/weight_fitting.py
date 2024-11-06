"""Classes for fitting per-event weights for training."""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Callable

import numpy as np
import pandas as pd
import sqlite3

from graphnet.data.utilities.sqlite_utilities import (
    create_table_and_save_to_sql,
)
from graphnet.utilities.logging import Logger


class WeightFitter(ABC, Logger):
    """Produces per-event weights.

    Weights are returned by the public method `fit_weights()`, and the weights
    can be saved as a table in the database.
    """

    def __init__(
        self,
        database_path: str,
        truth_table: str = "truth",
        index_column: str = "event_no",
    ):
        """Construct `UniformWeightFitter`."""
        self._database_path = database_path
        self._truth_table = truth_table
        self._index_column = index_column

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def _get_truth(
        self, variable: str, selection: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Return truth `variable`, optionally only for `selection` events."""
        if selection is None:
            query = f"select {self._index_column}, {variable} from {self._truth_table}"  # noqa: E501
        else:
            query = f"select {self._index_column}, {variable} from {self._truth_table} where {self._index_column} in {str(tuple(selection))}"  # noqa: E501
        with sqlite3.connect(self._database_path) as con:
            data = pd.read_sql(query, con)
        return data

    def fit(
        self,
        bins: np.ndarray,
        variable: str,
        weight_name: Optional[str] = None,
        add_to_database: bool = False,
        selection: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        db_count_norm: Optional[int] = None,
        automatic_log_bins: bool = False,
        max_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit weights.

        Calls private `_fit_weights` method. Output is returned as a
        pandas.DataFrame and optionally saved to sql.

        Args:
            bins: Desired bins used for fitting.
            variable: the name of the variable. Must match corresponding column
                name in the truth table.
            weight_name: Name of the weights.
            add_to_database: If True, the weights are saved to sql in a table
                named weight_name.
            selection: a list of event_no's. If given, only events in the
                selection is used for fitting.
            transform: A callable method that transform the variable into a
                desired space. E.g. np.log10 for energy. If given, fitting will
                happen in this space.
            db_count_norm: If given, the total sum of the weights for the given
                db will be this number.
            automatic_log_bins: If True, the bins are generated as a log10
                space between the min and max of the variable.
            max_weight: If given, the weights are capped such that a single
                event weight cannot exceed this number times the sum of
                all weights.
            **kwargs: Additional arguments passed to `_fit_weights`.


        Returns:
            DataFrame that contains weights, event_nos.
        """
        # Member variables
        self._variable = variable
        self._add_to_database = add_to_database
        self._selection = selection
        self._bins = bins
        self._transform = transform
        if max_weight is not None:
            assert max_weight > 0 and max_weight < 1
            self._max_weight = max_weight

        if weight_name is None:
            self._weight_name = self._generate_weight_name()
        else:
            self._weight_name = weight_name

        truth = self._get_truth(self._variable, self._selection)
        if self._transform is not None:
            truth[self._variable] = self._transform(truth[self._variable])
        if automatic_log_bins:
            assert isinstance(bins, int)
            self._bins = np.logspace(
                np.log10(truth[self._variable].min()),
                np.log10(truth[self._variable].max() + 1),
                bins,
            )

        weights = self._fit_weights(truth, **kwargs)
        if self._max_weight is not None:
            weights[self._weight_name] = np.where(
                weights[self._weight_name]
                > weights[self._weight_name].sum() * self._max_weight,
                weights[self._weight_name].sum() * self._max_weight,
                weights[self._weight_name],
            )

        if db_count_norm is not None:
            weights[self._weight_name] = (
                weights[self._weight_name]
                * db_count_norm
                / weights[self._weight_name].sum()
            )
        if add_to_database:
            create_table_and_save_to_sql(
                weights, self._weight_name, self._database_path
            )
        return weights.sort_values(self._index_column).reset_index(drop=True)

    @abstractmethod
    def _fit_weights(self, truth: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _generate_weight_name(self) -> str:
        pass


class Uniform(WeightFitter):
    """Produces per-event weights making variable distribution uniform."""

    def _fit_weights(self, truth: pd.DataFrame) -> pd.DataFrame:
        """Fit per-event weights.

        Args:
            truth: DataFrame containing the truth information.

        Returns:
            The fitted weights.
        """
        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[self._variable], bins=self._bins)

        # Get reweighting for each bin to achieve uniformity.
        # (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`, get the weight in
        # the corresponding bin
        ix = np.digitize(truth[self._variable], bins=self._bins) - 1
        sample_weights = bin_weights[ix]
        sample_weights = sample_weights / sample_weights.mean()

        truth[self._weight_name] = sample_weights
        return truth.sort_values("event_no").reset_index(drop=True)

    def _generate_weight_name(self) -> str:
        return self._variable + "_uniform_weight"


class BjoernLow(WeightFitter):
    """Produces per-event weights.

    Events below x_low are weighted to be uniform, whereas events above x_low
    are weighted to follow a 1/(1+a*(x_low -x)) curve.
    """

    def _fit_weights(  # type: ignore[override]
        self,
        truth: pd.DataFrame,
        x_low: float,
        alpha: float = 0.05,
        percentile: bool = False,
    ) -> pd.DataFrame:
        """Fit per-event weights.

        Args:
            truth: DataFrame containing the truth information.
            x_low: The cut-off for the truth variable. Values at or below x_low
                will be weighted to be uniform. Values above will follow a
                1/(1+a*(x_low -x)) curve.
            alpha: A scalar factor that controls how fast the weights above
                x_low approaches zero. Larger means faster.
            percentile: If True, x_low is interpreted as a percentile of the
                truth variable.

        Returns:
            The fitted weights.
        """
        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[self._variable], bins=self._bins)

        # Get reweighting for each bin to achieve uniformity.
        # (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`,
        # get the weight in the corresponding bin
        ix = np.digitize(truth[self._variable], bins=self._bins) - 1
        sample_weights = bin_weights[ix]
        sample_weights = sample_weights / sample_weights.mean()
        truth[self._weight_name] = sample_weights  # *0.1
        bin_counts, _ = np.histogram(
            truth[self._variable],
            bins=self._bins,
            weights=truth[self._weight_name],
        )
        c = bin_counts.max()

        if percentile:
            assert 0 < x_low < 1
            x_low = np.quantile(truth[self._variable], x_low)

        slice = truth[self._variable][truth[self._variable] > x_low]
        truth[self._weight_name][truth[self._variable] > x_low] = 1 / (
            1 + alpha * (slice - x_low)
        )

        bin_counts, _ = np.histogram(
            truth[self._variable],
            bins=self._bins,
            weights=truth[self._weight_name],
        )
        d = bin_counts.max()
        truth[self._weight_name][truth[self._variable] > x_low] = (
            truth[self._weight_name][truth[self._variable] > x_low] * c / d
        )
        return truth.sort_values(self._index_column).reset_index(drop=True)

    def _generate_weight_name(self) -> str:
        return self._variable + "_bjoern_low_weight"
