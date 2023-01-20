"""Unit tests for dataloader utilities.

@NOTE: These utility methods should be deprecated in favour of the indicated
member methods in `DataLoader`.
"""

import os.path

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.constants import TEST_DATA_DIR
from graphnet.training.utils import make_train_validation_dataloader

# Configuration
NB_EVENTS_TOTAL = 5
PATH = os.path.join(
    TEST_DATA_DIR,
    "sqlite",
    "oscNext_genie_level7_v02",
    "oscNext_genie_level7_v02_first_5_frames.db",
)


# Unit test(s)
def test_none_selection() -> None:
    """Test agreement of the two ways to calculate this loss."""
    (train_dataloader, test_dataloader,) = make_train_validation_dataloader(
        PATH,
        selection=None,
        pulsemaps=["SRTInIcePulses"],
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        batch_size=1,
    )

    assert len(train_dataloader) + len(test_dataloader) == NB_EVENTS_TOTAL


def test_array_selection() -> None:
    """Test agreement of the two ways to calculate this loss."""
    selection = [0, 1, 3, 4]

    (train_dataloader, test_dataloader,) = make_train_validation_dataloader(
        PATH,
        selection=selection,
        pulsemaps=["SRTInIcePulses"],
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        batch_size=1,
    )

    assert len(train_dataloader) + len(test_dataloader) == len(selection)


if __name__ == "__main__":
    test_array_selection()
