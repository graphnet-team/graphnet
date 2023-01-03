"""Example of using the WeightFitter class, specifically BjoernLow."""

import numpy as np
import matplotlib.pyplot as plt

from graphnet.constants import TEST_SQLITE_DATA
from graphnet.training.weight_fitting import BjoernLow
from graphnet.utilities.argparse import ArgumentParser


def main() -> None:
    """Run example."""
    # Configuration
    database = TEST_SQLITE_DATA
    variable = "energy"
    bins = bins = np.arange(0, 5, 0.01)  # in log10

    # Fit the uniform weights
    fitter = BjoernLow(database)
    weights = fitter.fit(
        bins=bins,
        variable=variable,
        add_to_database=False,
        transform=np.log10,
        x_low=1.5,
    )
    print(weights.head())

    # Plot the results
    fig = plt.figure()
    plt.hist(
        weights["energy"],
        bins=bins,
        weights=weights["energy_bjoern_low_weight"],
    )
    fig.savefig("test_hist_weighting_bjoern.png")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Fit per-event weights according to the `BjoernLow` weight fitter.
"""
    )

    args = parser.parse_args()

    main()
