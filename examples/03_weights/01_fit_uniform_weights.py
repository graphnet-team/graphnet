"""Example of using the WeightFitter class, specifically Uniform."""

import numpy as np
import matplotlib.pyplot as plt

from graphnet.constants import TEST_SQLITE_DATA
from graphnet.training.weight_fitting import Uniform
from graphnet.utilities.argparse import ArgumentParser


def main() -> None:
    """Run example."""
    # Configuration
    database = TEST_SQLITE_DATA
    variable = "zenith"
    bins = np.arange(0, np.deg2rad(180.5), np.deg2rad(0.5))

    # Fit the uniform weights
    fitter = Uniform(database)
    weights = fitter.fit(bins=bins, variable=variable, add_to_database=False)
    print(weights.head())

    # Plot the results
    fig = plt.figure()
    plt.hist(
        weights["zenith"], bins=bins, weights=weights["zenith_uniform_weight"]
    )
    fig.savefig("test_hist_weighting_uniform.png")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Fit per-event weights to make the truth-level zenith distribution uniform.
"""
    )

    args = parser.parse_args()

    main()
