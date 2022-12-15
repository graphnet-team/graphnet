"""Example of using the WeightFitter class, specifically Uniform."""

import numpy as np
import matplotlib.pyplot as plt

from graphnet.training.weight_fitting import Uniform


def main() -> None:
    """Run example."""
    # Configuration
    database = "/my_databases/my_database/data/my_database.db"
    variable = "zenith"
    bins = np.arange(0, np.deg2rad(180.5), np.deg2rad(0.5))

    # Fit the uniform weights
    fitter = Uniform(database)
    weights = fitter.fit(bins=bins, variable=variable, add_to_database=True)

    # Plot the results
    fig = plt.figure()
    plt.hist(
        weights["zenith"], bins=bins, weights=weights["zenith_uniform_weight"]
    )
    fig.savefig("test_hist.png")


if __name__ == "__main__":
    main()
