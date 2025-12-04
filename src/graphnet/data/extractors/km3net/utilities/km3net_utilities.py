"""Code with some functionalities for the extraction."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def create_unique_id_run_by_run(
    file_type: str,
    run_id: List[int],
    evt_id: List[int],
    hnl_model: str,
) -> List[int]:
    """Create a unique ID for each event based on its parameters.

    Args:
        file_type (str): 'neutrino', 'muon', 'noise', 'data', or 'hnl'.
        run_id (List[int]): List of run IDs for the events.
        evt_id (List[int]): List of event IDs within each run.

    Returns:
        List[str]: A list of unique IDs for each event, formatted as strings.
    """
    file_type_dict = {
        "neutrino": 1,
        "muon": 2,
        "noise": 3,
        "data": 4,
        "hnl": 5,
    }

    hnl_type_dict = {
        "none": 0,  # possibility of adding hnl models to break run-filetype degeneracy
    }
    unique_id = []
    for i in range(len(run_id)):
        unique_id.append(
            int(
                str(run_id[i])
                + str(evt_id[i])
                + str(file_type_dict[file_type])
                + str(hnl_type_dict[hnl_model])
            )
        )

    return unique_id


def filter_None_NaN(
    values: Union[List[float], np.ndarray],
    padding_value: float,
) -> np.ndarray:
    """Remove None and NaN, transforming them to padding value."""
    values = [padding_value if v is None else v for v in values]
    values = np.array(values, dtype=float)
    values[np.isnan(values)] = padding_value
    return values


def xyz_dir_to_zen_az(
    dir_x: List[float],
    dir_y: List[float],
    dir_z: List[float],
    padding_value: float,
) -> Tuple[List[float], List[float]]:
    """Convert direction vector to zenith and azimuth angles."""
    # Compute zenith angle (elevation angle)
    with np.errstate(invalid="ignore"):
        zenith = np.arccos(dir_z)  # zenith angle in radians

    # Compute azimuth angle
    azimuth = np.arctan2(dir_y, dir_x)  # azimuth angle in radians
    az_centered = azimuth + np.pi * np.ones(
        len(azimuth)
    )  # Center the azimuth angle around zero
    # check for NaN in the zenith and replace with padding_value
    zenith[np.isnan(zenith)] = padding_value
    # change the azimuth values to padding value if the zenith is padding value
    az_centered[zenith == padding_value] = padding_value

    return zenith, az_centered


def creating_time_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the event time so that the first hit has zero in time."""
    df = df.sort_values(by=["event_no", "t"])
    df["min_t"] = df.groupby("event_no")["t"].transform("min")
    df["t"] = df["t"] - df["min_t"]
    df = df.drop(["min_t"], axis=1)

    return df


def assert_no_uint_values(df: pd.DataFrame) -> pd.DataFrame:
    """Assert no format no supported by sqlite is in the data."""
    for column in df.columns:
        if df[column].dtype == "uint32":
            df[column] = df[column].astype("int32")
        elif df[column].dtype == "uint64":
            df[column] = df[column].astype("int64")
        else:
            pass
    return df
