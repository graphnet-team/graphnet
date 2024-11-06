"""RNG-related utility functions relevant to the graphnet.data package."""

from typing import List, Tuple
import pandas as pd


def pairwise_shuffle(
    i3_list: List[str], gcd_list: List[str]
) -> Tuple[List[str], List[str]]:
    """Shuffle the I3 file list and the correponding gcd file list.

    This is handy because it ensures a more even extraction load for each
    worker.

    Args:
        files_list: List of I3 file paths.
        gcd_list: List of corresponding gcd file paths.

    Returns:
        i3_shuffled: List of shuffled I3 file paths.
        gcd_shuffled: List of corresponding gcd file paths.
    """
    df = pd.DataFrame({"i3": i3_list, "gcd": gcd_list})
    df_shuffled = df.sample(frac=1, replace=False)
    i3_shuffled = df_shuffled["i3"].tolist()
    gcd_shuffled = df_shuffled["gcd"].tolist()
    return i3_shuffled, gcd_shuffled
