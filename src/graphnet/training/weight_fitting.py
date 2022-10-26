import numpy as np
import pandas as pd
import sqlite3
from typing import Optional, List, Callable
from graphnet.data.sqlite.sqlite_utilities import (
    run_sql_code,
    save_to_sql,
    create_table,
)
from abc import ABC, abstractmethod
from graphnet.utilities.logging import LoggerMixin


class WeightFitter(ABC, LoggerMixin):
    def __init__(
        self,
        database_path,
        truth_table="truth",
        index_column="event_no",
    ):
        self._database_path = database_path
        self._truth_table = truth_table
        self._index_column = index_column

    def _get_truth(self, variable: str, selection: Optional[List[int]] = None):
        """Return truth `variable`, optionally only for `selection` event nos."""
        if selection is None:
            query = f"select {self._index_column}, {variable} from {self._truth_table}"
        else:
            query = f"select {self._index_column}, {variable} from {self._truth_table} where {self._index_column} in {str(tuple(selection))}"
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
        **kwargs,
    ) -> pd.DataFrame:
        """Fits weights. Calls private _fit_weights method. Output is returned as a pandas.DataFrame and optionally saved to sql.

        Args:
            bins (np.ndarray): Desired bins used for fitting.
            variable (str): the name of the variable. Must match corresponding column name in the truth table.
            weight_name (Optional[str], optional): Name of the weights. Defaults to None.
            add_to_database (bool, optional): if True, the weights are saved to sql in a table named weight_name. Defaults to False.
            selection (Optional[List[int]], optional): a list of event_no's. If given, only events in the selection is used for fitting. Defaults to None.
            transform (Optional[Callable], optional): A callable method that transform the variable into a desired space. E.g. np.log10 for energy. If given, fitting will happen in this space.
            **kwargs: additional arguments

        Returns:
            pd.DataFrame: a pandas.DataFrame that contains weights, event_nos
        """

        # member_variables
        self._variable = variable
        self._add_to_database = add_to_database
        self._selection = selection
        self._bins = bins
        self._transform = transform

        if weight_name is None:
            self._weight_name = self._generate_weight_name()
        else:
            self._weight_name = weight_name

        truth = self._get_truth(self._variable, self._selection)
        if self._transform is not None:
            truth[self._variable] = self._transform(truth[self._variable])
        weights = self._fit_weights(truth, **kwargs)

        if add_to_database:
            create_table(weights, self._weight_name, self._database_path)
            save_to_sql(weights, self._weight_name, self._database_path)
        return weights.sort_values(self._index_column).reset_index(drop=True)

    @abstractmethod
    def _fit_weights(self, truth, **kwargs) -> pd.DataFrame:
        return

    @abstractmethod
    def _generate_weight_name(self):
        return


class Uniform(WeightFitter):
    """Produces per-event weights making fitted variable distribution uniform."""

    def _fit_weights(self, truth: pd.DataFrame) -> pd.DataFrame:
        """Produces per-event weights making fitted variable distribution uniform.

        Args:
            truth (pd.DataFrame): a pandas.DataFrame containing the truth information.

        Returns:
            pd.DataFrame: The fitted weights.
        """
        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[self._variable], bins=self._bins)

        # Get reweighting for each bin to achieve uniformity. (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`, get the weight in the corresponding bin
        ix = np.digitize(truth[self._variable], bins=self._bins) - 1
        sample_weights = bin_weights[ix]
        sample_weights = sample_weights / sample_weights.mean()

        truth[self._weight_name] = sample_weights
        return truth.sort_values("event_no").reset_index(drop=True)

    def _generate_weight_name(self):
        return self._variable + "_uniform_weight"


class BjoernLow(WeightFitter):
    """Produces pr. event weights. Events below x_low are weighted to be uniform, whereas events above x_low are weighted to follow a 1/(1+a*(x_low -x)) curve."""

    def _fit_weights(
        self, truth: pd.DataFrame, x_low: float, alpha: float = 0.05
    ) -> pd.DataFrame:
        """Produces pr. event weights. Events below x_low are weighted to be uniform, whereas events above x_low are weighted to follow a 1/(1+a*(x_low -x)) curve.

        Args:
            truth (pd.DataFrame): A pandas.DataFrame containing the truth information.
            x_low (float): The cut-off for the truth variable. Values at or below x_low will be weighted to be uniform. Values above will follow a 1/(1+a*(x_low -x)) curve
            alpha (float, optional): A scalar factor that controls how fast the weights above x_low approaches zero. Larger means faster. Defaults to 0.05.

        Returns:
            pd.DataFrame: A pandas.DataFrame containing the fitted weights.
        """
        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[self._variable], bins=self._bins)

        # Get reweighting for each bin to achieve uniformity. (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`, get the weight in the corresponding bin
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

    def _generate_weight_name(self):
        return self._variable + "_bjoern_low_weight"
