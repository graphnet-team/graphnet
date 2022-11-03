"""Example of using the WeightFitter class, specifically BjoernLow."""

import numpy as np
import matplotlib.pyplot as plt

from graphnet.training.weight_fitting import BjoernLow


def main() -> None:
    """Run example."""
    # Configuration
    database = "/my_databases/my_database/data/my_database.db"
    variable = "energy"
    bins = bins = np.arange(0, 5, 0.01)  # in log10

    # Fit the uniform weights
    fitter = BjoernLow(database)
    weights = fitter.fit(
        bins=bins,
        variable=variable,
        add_to_database=True,
        transform=np.log10,
        x_low=1.5,
    )

    # Plot the results
    fig = plt.figure()
    plt.hist(
        weights["energy"],
        bins=bins,
        weights=weights["energy_bjørn_low_weight"],
    )
    fig.savefig("test_hist.png")


if __name__ == "__main__":
    main()
