import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graphnet.training.weight_fitting import UniformWeightFitter


def get_truth_zenith(db: str) -> pd.DataFrame:
    """Gets the true zenith for neutrinos in db.

    Args:
        db (str): path to database

    Returns:
        pandas.DataFrame
    """
    with sqlite3.connect(db) as con:
        query = "select event_no, zenith from truth where abs(pid) != 1"
        data = pd.read_sql(query, con)
    return data


database = "/my_databases/my_database/data/my_database.db"

# Choose truth variable
variable = "zenith"

# Choose binning
bins = np.arange(0, np.deg2rad(180.5), np.deg2rad(0.5))

# Fit the uniform weights
fitter = UniformWeightFitter(database)
weights = fitter.fit_weights(
    bins=bins, variable=variable, add_to_database=True
)

# Plot the results
truth = get_truth_zenith(database)
fig = plt.figure()
plt.hist(truth["zenith"], bins=bins, weights=weights["zenith_uniform_weight"])
fig.savefig("test_hist.png")
