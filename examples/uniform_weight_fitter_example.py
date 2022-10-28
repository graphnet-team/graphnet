import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graphnet.training.weight_fitting import Uniform

database = "/my_databases/my_database/data/my_database.db"

# Choose truth variable
variable = "zenith"

# Choose binning
bins = np.arange(0, np.deg2rad(180.5), np.deg2rad(0.5))

# Fit the uniform weights
fitter = Uniform(database)
weights = fitter.fit_weights(
    bins=bins, variable=variable, add_to_database=True
)

# Plot the results
fig = plt.figure()
plt.hist(
    weights["zenith"], bins=bins, weights=weights["zenith_uniform_weight"]
)
fig.savefig("test_hist.png")
