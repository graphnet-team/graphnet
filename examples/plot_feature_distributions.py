import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.training.utils import make_train_validation_dataloader


# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

# Main function definition
def main():

    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuraiton
    db = '/groups/icecube/asogaard/data/sqlite/dev_upgrade_step4_preselection_decemberv2/data/dev_upgrade_step4_preselection_decemberv2.db'
    pulsemaps = ['I3RecoPulseSeriesMapRFCleaned_mDOM', 'IceCubePulsesTWSRT']
    batch_size = 256
    num_workers = 10

    # Common variables
    train_selection, _ = get_equal_proportion_neutrino_indices(db)
    train_selection = train_selection[0:10000]

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db,
        train_selection,
        pulsemaps,
        features,
        truth,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Building model
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )

    # Get feature matrix
    x_original = []
    x_preprocessed = []
    for batch in tqdm(training_dataloader):
        x_original.append(batch.x.numpy())
        x_preprocessed.append(detector(batch).x.numpy())

    x_original = np.concatenate(x_original, axis=0)
    x_preprocessed = np.concatenate(x_preprocessed, axis=0)

    print("Number of NaNs:", np.sum(np.isnan(x_original)))
    print("Number of infs:", np.sum(np.isinf(x_original)))

    # Plot feature distributions
    nb_features = len(features)
    dim = int(np.ceil(np.sqrt(nb_features)))
    axis_size = 4
    bins = 100

    # -- Original
    fig, axes = plt.subplots(dim, dim, figsize=(dim * axis_size, dim * axis_size))
    for ix, ax in enumerate(axes.ravel()[:nb_features]):
        ax.hist(x_original[:,ix], bins=bins)
        ax.set_xlabel(f"x{ix}: {features[ix]}")
        ax.set_yscale('log')

    fig.tight_layout
    fig.savefig('feature_distribution_original.png')

    # -- Preprocessed
    fig, axes = plt.subplots(dim, dim, figsize=(dim * axis_size, dim * axis_size))
    for ix, ax in enumerate(axes.ravel()[:nb_features]):
        ax.hist(x_preprocessed[:,ix], bins=bins, color='orange')
        ax.set_xlabel(f"x{ix}: {features[ix]}")
        ax.set_yscale('log')

    fig.tight_layout
    fig.savefig('feature_distribution_preprocessed.png')

# Main function call
if __name__ == "__main__":
    main()
