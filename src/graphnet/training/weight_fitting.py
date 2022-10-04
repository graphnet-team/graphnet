import numpy as np
import pandas as pd
import sqlite3
from typing import Optional, List
from graphnet.data.utils import run_sql_code, save_to_sql, create_table


class UniformWeightFitter:
    """Produces per-event weights making fitted variable distribution uniform.
    Weights are returned by the public method `fit_weights()`, and the weights can be saved as a table in the database.
    """

    def __init__(
        self,
        database_path,
        truth_table="truth",
        index_column="event_no",
    ):
        self._database_path = database_path
        self._truth_table = truth_table
        self._index_column = index_column

    def fit_weights(
        self,
        bins: np.ndarray,
        variable: str,
        weight_name: Optional[str] = None,
        add_to_database: bool = False,
        selection: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Calculate per-event weights to make `variable` distribution uniform.

        If selection is None, the weights are calculated using all events in the
        database.

        Args:
            bins (numpy.array): An array containing the bins. E.g.
                numpy.arange(0,180,1)
            variable (str): the name of the truth variable in truth_table
            weight_name (str, optional): the assigned column name of the weights
                if add_to_database = True. Defaults to None.
            add_to_database (bool, optional): If True, the weights are written
                to the database as a new table named weight_name. Defaults to
                False.
            selection (List, optional): a List containing the event_no's to
                which the weights are calculated. Defaults to None.

        Returns:
            pandas.DataFrame
        """
        truth = self._get_truth(variable, selection)
        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[variable], bins=bins)

        # Get reweighting for each bin to achieve uniformity. (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`, get the weight in the corresponding bin
        ix = np.digitize(truth[variable], bins=bins) - 1
        sample_weights = bin_weights[ix]
        sample_weights = sample_weights / sample_weights.mean()
        if weight_name is None:
            weight_name = variable + "_uniform_weight"
        truth[weight_name] = sample_weights
        truth = truth.drop(columns=[variable])
        if add_to_database:
            create_table(self._database_path, weight_name, truth)
            save_to_sql(truth, weight_name, self._database_path)
        return truth.sort_values("event_no").reset_index(drop=True)

    def _get_truth(self, variable: str, selection: Optional[List[int]] = None):
        """Return truth `variable`, optionally only for `selection` event nos."""
        if selection is None:
            query = f"select {self._index_column}, {variable} from {self._truth_table}"
        else:
            query = f"select {self._index_column}, {variable} from {self._truth_table} where {self._index_column} in {str(tuple(selection))}"
        with sqlite3.connect(self._database_path) as con:
            data = pd.read_sql(query, con)
        return data
