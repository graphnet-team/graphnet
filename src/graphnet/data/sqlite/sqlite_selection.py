"""Selection-specific utility functions for use in `graphnet.data.sqlite`."""

import sqlite3
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from graphnet.utilities.logging import get_logger

logger = get_logger()


def get_desired_event_numbers(
    database: str,
    desired_size: int,
    fraction_noise: float = 0,
    fraction_muon: float = 0,
    fraction_nu_e: float = 0,
    fraction_nu_mu: float = 0,
    fraction_nu_tau: float = 0,
    seed: int = 0,
) -> List[int]:
    """Get event numbers for specified fractions of physics processes.

    Args:
        database: Path to database from which to get event numbers.
        desired_size: Number of event numbers to get.
        fraction_noise: Fraction of noise events.
        fraction_muon: Fraction of noise events.
        fraction_nu_e: Fraction of nu_e events.
        fraction_nu_mu: Fraction of nu_mu events.
        fraction_nu_tau: Fraction of nu_tau events.
        seed: Random number generator reed.

    Returns:
        List of event numbers.
    """
    assert (
        fraction_noise
        + fraction_muon
        + fraction_nu_e
        + fraction_nu_mu
        + fraction_nu_tau
        == 1.0
    ), "Sum of fractions not equal to one."
    rng = np.random.RandomState(seed=seed)

    fracs = [
        fraction_noise,
        fraction_muon,
        fraction_nu_e,
        fraction_nu_mu,
        fraction_nu_tau,
    ]
    numbers_desired = [int(x * desired_size) for x in fracs]
    pids = [1, 13, 12, 14, 16]

    with sqlite3.connect(database) as con:
        total_query = "SELECT event_no FROM truth WHERE abs(pid) IN {}".format(
            tuple(pids)
        )
        tot_event_nos = pd.read_sql(total_query, con)
        if len(tot_event_nos) < desired_size:
            desired_size = len(tot_event_nos)
            numbers_desired = [int(x * desired_size) for x in fracs]
            logger.info(
                "Only {} events in database, using this number instead.".format(
                    len(tot_event_nos)
                )
            )

        list_of_dataframes: List[pd.DataFrame] = []
        restart_trigger = True
        while restart_trigger:
            restart_trigger = False
            for number, particle_type in zip(numbers_desired, pids):
                query_is = (
                    "SELECT event_no FROM truth WHERE abs(pid) == {}".format(
                        particle_type
                    )
                )
                tmp_dataframe = pd.read_sql(query_is, con)
                try:
                    dataframe = tmp_dataframe.sample(
                        number, replace=False, random_state=rng
                    ).reset_index(
                        drop=True
                    )  # could add weights (re-weigh) here with replace=True
                except ValueError:
                    if len(tmp_dataframe) == 0:
                        logger.info(
                            "There are no particles of type {} in this database please make new request.".format(
                                particle_type
                            )
                        )
                        raise
                    logger.info(
                        "There have been {} requested of particle {}, we can only supply {}. \nRenormalising...".format(
                            number, particle_type, len(tmp_dataframe)
                        )
                    )
                    numbers_desired = [
                        int(new_x * (len(tmp_dataframe) / number))
                        for new_x in numbers_desired
                    ]
                    restart_trigger = True
                    list_of_dataframes = []
                    break

                list_of_dataframes.append(dataframe)
        retrieved_event_nos_pd = pd.concat(list_of_dataframes)
        event_no_list = (
            retrieved_event_nos_pd.sample(
                frac=1, replace=False, random_state=rng
            )
            .values.ravel()
            .tolist()
        )

    return event_no_list


def get_equal_proportion_neutrino_indices(
    database: str, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Get indices for neutrino events in equal flavour proportions.

    Args:
        database: Path to database from which to get the event numbers.
        seed: Random number generator seed.

    Returns:
        Tuple containing lists of training and test event numbers, resp.
    """
    # Common variables
    pids = ["12", "14", "16"]
    pid_indicies = {}
    indices = []
    rng = np.random.RandomState(seed=seed)

    # Get a list of all event numbers for each PID
    with sqlite3.connect(database) as conn:
        for pid in pids:
            pid_indicies[pid] = pd.read_sql_query(
                f"SELECT event_no FROM truth where abs(pid) = {pid}", conn
            )

    # Subsample events for each PID to the smallest sample size
    samples_sizes = list(map(len, pid_indicies.values()))
    smallest_sample_size = min(samples_sizes)
    logger.info(f"Smallest sample size: {smallest_sample_size}")

    indices = [
        (
            pid_indicies[pid]
            .sample(smallest_sample_size, replace=False, random_state=rng)
            .reset_index(drop=True)
        )
        for pid in pids
    ]
    indices_equal_proprtions = pd.concat(indices, ignore_index=True)

    # Shuffle and convert to list
    indices_equal_proprtions = (
        indices_equal_proprtions.sample(
            frac=1, replace=False, random_state=rng
        )
        .values.ravel()
        .tolist()
    )

    # Get test indices (?)
    with sqlite3.connect(database) as con:
        train_event_nos = (
            "(" + ", ".join(map(str, indices_equal_proprtions)) + ")"
        )
        query = f"select event_no from truth where abs(pid) != 13 and abs(pid) != 1 and event_no not in {train_event_nos}"
        test = pd.read_sql(query, con).values.ravel().tolist()

    return indices_equal_proprtions, test


def get_even_signal_background_indicies(database: str) -> List[int]:
    """Get event numbers with equal proportion neutrino and muon events.

    Args:
        database: Path to database from which to get the event numbers.

    Returns:
        List of event numbers.
    """
    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 13"
        muons = pd.read_sql(query, con)
    neutrinos_list, _ = get_equal_proportion_neutrino_indices(database)
    neutrinos = pd.DataFrame(neutrinos_list)

    if len(neutrinos) > len(muons):
        neutrinos = neutrinos.sample(len(muons))
    else:
        muons = muons.sample(len(neutrinos))

    indicies = []
    indicies.extend(muons.values.ravel().tolist())
    indicies.extend(neutrinos.values.ravel().tolist())
    df_for_shuffle = pd.DataFrame(indicies).sample(frac=1)
    return df_for_shuffle.values.ravel().tolist()


def get_even_track_cascade_indicies(
    database: str,
) -> Tuple[List[int], List[int]]:
    """Get event numbers with equal proportion CC and NC e/mu neutrino events.

    Args:
        database: Path to database from which to get the event numbers.

    Returns:
        Tuple containing lists of training and test event numbers, resp.
    """
    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 12 and interaction_type = 1"
        nu_e_cc = pd.read_sql(query, con)

    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 12 and interaction_type = 2"
        nu_e_nc = pd.read_sql(query, con)

    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 14 and interaction_type = 1"
        nu_u_cc = pd.read_sql(query, con)

    with sqlite3.connect(database) as con:
        query = "select event_no from truth where abs(pid) = 14 and interaction_type = 2"
        nu_u_nc = pd.read_sql(query, con)

    events = (
        nu_e_nc.append(
            nu_e_cc.sample(len(nu_e_nc)).reset_index(drop=True),
            ignore_index=True,
        )
        .append(
            nu_u_nc.sample(len(nu_e_nc)).reset_index(drop=True),
            ignore_index=True,
        )
        .append(
            nu_u_cc.sample(3 * len(nu_e_nc)).reset_index(drop=True),
            ignore_index=True,
        )
    )
    train_events = events.sample(frac=1).reset_index(drop=True)

    with sqlite3.connect(database) as con:
        query = (
            "select event_no from truth where abs(pid) != 13 and event_no not in %s"
            % str(tuple(train_events["event_no"]))
        )
        test = pd.read_sql(query, con).values.ravel().tolist()

    return train_events.values.ravel().tolist(), test


def get_even_dbang_selection(
    database: str,
    min_max_decay_length: Optional[Tuple[float, float]] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Get event numbers for equal numbers of dbang / non-dbang events.

    Args:
        db : Path to database.
        seed: Random number generator seed. Defaults to 42.

    Returns:
        Tuple containing lists of training and test event numbers, resp.
    """
    # Common variables
    pids = ["12", "14", "16"]
    non_dbangs_indicies = {}
    indices = []
    rng = np.random.RandomState(seed=seed)

    # Get a list of all event numbers for each PID that is not dbang
    with sqlite3.connect(database) as conn:
        for pid in pids:
            non_dbangs_indicies[pid] = pd.read_sql_query(
                f"SELECT event_no FROM truth where abs(pid) = {pid} and dbang_decay_length = -1",
                conn,
            )

    # Subsample events for each PID to the smallest sample size
    samples_sizes = list(map(len, non_dbangs_indicies.values()))
    smallest_sample_size = min(samples_sizes)
    logger.info(f"Smallest non dbang sample size: {smallest_sample_size}")
    indices = [
        (
            non_dbangs_indicies[pid]
            .sample(smallest_sample_size, replace=False, random_state=rng)
            .reset_index(drop=True)
        )
        for pid in pids
    ]
    indices_equal_proprtions = pd.concat(indices, ignore_index=True)

    # Get a list of all event numbers  that is dbang
    if min_max_decay_length is None:
        with sqlite3.connect(database) as conn:
            dbangs_indicies = pd.read_sql_query(
                "SELECT event_no FROM truth where dbang_decay_length != -1",
                conn,
            )
        logger.info(f"dbang sample size: {len(dbangs_indicies)}")
    elif min_max_decay_length[1] is None:
        with sqlite3.connect(database) as conn:
            dbangs_indicies = pd.read_sql_query(
                f"SELECT event_no FROM truth where dbang_decay_length != -1 and dbang_decay_length >= {min_max_decay_length[0]}",
                conn,
            )
    else:
        with sqlite3.connect(database) as conn:
            dbangs_indicies = pd.read_sql_query(
                f"SELECT event_no FROM truth where dbang_decay_length != -1 and dbang_decay_length >= {min_max_decay_length[0]} and dbang_decay_length <= {min_max_decay_length[1]}",
                conn,
            )

    if len(indices_equal_proprtions) > len(dbangs_indicies):
        indices_equal_proprtions = indices_equal_proprtions.sample(
            len(dbangs_indicies)
        ).reset_index(drop=True)
    else:
        dbangs_indicies = dbangs_indicies.sample(
            len(indices_equal_proprtions)
        ).reset_index(drop=True)

    logger.info("dbangs in joint sample: %s" % len(dbangs_indicies))
    logger.info(
        "non-dbangs in joint sample: %s" % len(indices_equal_proprtions)
    )

    joint_indicies = (
        dbangs_indicies.append(indices_equal_proprtions, ignore_index=True)
        .reset_index(drop=True)
        .sample(frac=1, replace=False, random_state=rng)
        .values.ravel()
        .tolist()
    )
    # Shuffle and convert to list

    # Get test indices (?)
    with sqlite3.connect(database) as con:
        train_event_nos = "(" + ", ".join(map(str, joint_indicies)) + ")"
        query = f"select event_no from truth where abs(pid) != 13 and abs(pid) != 1 and event_no not in {train_event_nos}"
        test = pd.read_sql(query, con).values.ravel().tolist()

    return joint_indicies, test
