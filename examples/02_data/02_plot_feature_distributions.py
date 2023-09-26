"""Example of visualization of input data from a configured Dataset."""

import os.path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from graphnet.constants import CONFIG_DIR
from graphnet.data.dataset import Dataset
from graphnet.utilities.logging import Logger
from graphnet.utilities.argparse import ArgumentParser


def main() -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Construct dataloader
    dataset = Dataset.from_config(
        os.path.join(CONFIG_DIR, "datasets/test_data_sqlite.yml")
    )

    assert isinstance(dataset, Dataset)
    features = dataset._features[1:]

    # Get feature matrix
    x_preprocessed_list = []
    for batch in tqdm(dataset, colour="green"):
        x_preprocessed_list.append(batch.x.numpy())

    x_preprocessed = np.concatenate(x_preprocessed_list, axis=0)
    logger.info(f"Number of NaNs: {np.sum(np.isnan(x_preprocessed))}")
    logger.info(f"Number of infs: {np.sum(np.isinf(x_preprocessed))}")

    # Plot feature distributions
    nb_features_preprocessed = x_preprocessed.shape[1]
    dim = int(np.ceil(np.sqrt(nb_features_preprocessed)))
    axis_size = 4
    bins = 50

    # -- Preprocessed
    fig, axes = plt.subplots(
        dim, dim, figsize=(dim * axis_size, dim * axis_size)
    )
    for ix, ax in enumerate(axes.ravel()[:nb_features_preprocessed]):
        ax.hist(x_preprocessed[:, ix], bins=bins, color="orange")
        ax.set_xlabel(
            f"x{ix}: {features[ix] if ix < len(features) else 'N/A'}"
        )
        ax.set_yscale("log")

    fig.tight_layout
    figure_name_preprocessed = "feature_distribution_preprocessed.png"
    fig.savefig(figure_name_preprocessed)
    logger.info(f"Figure written to {figure_name_preprocessed}")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Plot feature distributions in dataset.
"""
    )

    args, unknown = parser.parse_known_args()

    main()
