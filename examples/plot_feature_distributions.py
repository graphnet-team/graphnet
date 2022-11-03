"""Example of plotting feature distributions from SQLite database."""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_equal_proportion_neutrino_indices,
)
from graphnet.models.detector.icecube import IceCubeUpgrade, IceCubeUpgrade_V2
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.logging import get_logger


logger = get_logger()

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE


def main() -> None:
    """Run example."""
    # Remove `interaction_time` if it exists
    try:
        del truth[truth.index("interaction_time")]
    except ValueError:
        # not found in list
        pass

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    db = "/groups/icecube/asogaard/data/sqlite/dev_upgrade_step4_preselection_decemberv2/data/dev_upgrade_step4_preselection_decemberv2.db"
    pulsemaps = [
        "IceCubePulsesTWSRT",
        "I3RecoPulseSeriesMapRFCleaned_mDOM",
        "IceCubePulsesTWSRT",
    ]
    batch_size = 256
    num_workers = 1

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(db)
    train_selection = train_selection[0:10000]

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        db,
        train_selection,
        pulsemaps,
        features,
        truth,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Building model
    detector = IceCubeUpgrade_V2(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )

    # Get feature matrix
    x_original_list = []
    x_preprocessed_list = []
    for batch in tqdm(training_dataloader):
        x_original_list.append(batch.x.numpy())
        x_preprocessed_list.append(detector(batch).x.numpy())

    x_original = np.concatenate(x_original_list, axis=0)
    x_preprocessed = np.concatenate(x_preprocessed_list, axis=0)

    logger.info("Number of NaNs:", np.sum(np.isnan(x_original)))
    logger.info("Number of infs:", np.sum(np.isinf(x_original)))

    # Plot feature distributions
    nb_features_original = x_original.shape[1]
    nb_features_preprocessed = x_preprocessed.shape[1]
    dim = int(np.ceil(np.sqrt(nb_features_preprocessed)))
    axis_size = 4
    bins = 100

    # -- Original
    fig, axes = plt.subplots(
        dim, dim, figsize=(dim * axis_size, dim * axis_size)
    )
    for ix, ax in enumerate(axes.ravel()[:nb_features_original]):
        ax.hist(x_original[:, ix], bins=bins)
        ax.set_xlabel(
            f"x{ix}: {features[ix] if ix < len(features) else 'N/A'}"
        )
        ax.set_yscale("log")

    fig.tight_layout
    fig.savefig("feature_distribution_original.png")

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
    fig.savefig("feature_distribution_preprocessed.png")


if __name__ == "__main__":
    main()
