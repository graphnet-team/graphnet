"""Utility function relevant to the gnn_reco.data package."""

from glob import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import List, Tuple
import sqlite3

def get_equal_proportion_neutrino_indices(db: str, seed: int = 42) -> Tuple[List[int]]:
    """Utility method to get indices for neutrino events in equal flavour proportions.

    Args:
        db (str): Path to database.
        seed (int, optional): Random number generator seed. Defaults to 42.

    Returns:
        tuple: Training and test indices, resp.
    """
    # Common variables
    pids = ['12', '14', '16']
    pid_indicies = {}
    indices = []
    rng = np.random.RandomState(seed=seed)

    # Get a list of all event numbers for each PID
    with sqlite3.connect(db) as conn:
        for pid in pids:
            pid_indicies[pid] = pd.read_sql_query(f"SELECT event_no FROM truth where abs(pid) = {pid}", conn)

    # Subsample events for each PID to the smallest sample size
    samples_sizes = list(map(len, pid_indicies.values()))
    smallest_sample_size = min(samples_sizes)
    print(f"Smallest sample size: {smallest_sample_size}")

    indices = [
        (
            pid_indicies[pid]
            .sample(smallest_sample_size, replace=False, random_state=rng)
            .reset_index(drop=True)
        ) for pid in pids
    ]
    indices_equal_proprtions = pd.concat(indices, ignore_index=True)

    # Shuffle and convert to list
    indices_equal_proprtions = (
        indices_equal_proprtions
        .sample(frac=1, replace=False, random_state=rng)
        .values
        .ravel()
        .tolist()
    )

    # Get test indices (?)
    with sqlite3.connect(db) as con:
        train_event_nos = '(' + ', '.join(map(str, indices_equal_proprtions)) + ')'
        query = f'select event_no from truth where abs(pid) != 13 and event_no not in {train_event_nos}'
        test = pd.read_sql(query, con).values.ravel().tolist()

    return indices_equal_proprtions, test

def get_even_signal_background_indicies(db):
    with sqlite3.connect(db) as con:
        query = 'select event_no from truth where abs(pid) = 13'
        muons = pd.read_sql(query,con)
    neutrinos, _ = get_equal_proportion_neutrino_indices(db)
    neutrinos =  pd.DataFrame(neutrinos)

    if len(neutrinos) > len(muons):
        neutrinos = neutrinos.sample(len(muons))
    else:
        muons = muons.sample(len(neutrinos))

    indicies = []
    indicies.extend(muons.values.ravel().tolist())
    indicies.extend(neutrinos.values.ravel().tolist())
    df_for_shuffle = pd.DataFrame(indicies).sample(frac = 1)
    return df_for_shuffle.values.ravel().tolist()

def get_even_track_cascade_indicies(database):
    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) = 12 and interaction_type = 1'
        nu_e_cc = pd.read_sql(query,con)

    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) = 12 and interaction_type = 2'
        nu_e_nc = pd.read_sql(query,con)

    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) = 14 and interaction_type = 1'
        nu_u_cc = pd.read_sql(query,con)

    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) = 14 and interaction_type = 2'
        nu_u_nc = pd.read_sql(query,con)

    events = nu_e_nc.append(nu_e_cc.sample(len(nu_e_nc)).reset_index(drop = True), ignore_index= True).append(nu_u_nc.sample(len(nu_e_nc)).reset_index(drop = True), ignore_index= True).append(nu_u_cc.sample(3*len(nu_e_nc)).reset_index(drop = True), ignore_index= True)
    train_events = events.sample(frac = 1).reset_index(drop = True)


    with sqlite3.connect(database) as con:
        query = 'select event_no from truth where abs(pid) != 13 and event_no not in %s'%str(tuple(train_events['event_no']))
        test = pd.read_sql(query, con).values.ravel().tolist()

    return events.values.ravel().tolist(), test



def create_out_directory(outdir: str):
    try:
        os.makedirs(outdir)
    except:
        print(f"Directory {outdir} already exists")

def is_gcd_file(filename: str) -> bool:
    """Checks whether `filename` is a GCD file."""
    if re.search('(gcd|geo)', filename.lower()):
        return True
    return False

def is_i3_file(filename: str) -> bool:
    """Checks whether `filename` is an I3 file."""
    if is_gcd_file(filename.lower()):
        return False
    elif re.search(r'\.i3\.', filename.lower()):
        return True
    return False

def has_extension(filename: str, extensions: List[str]) -> bool:
    """Checks whether `filename` has one of the desired extensions."""
    # @TODO: Remove method, as it is not used?
    return re.search('(' + '|'.join(extensions) + ')$', filename) is not None

def pairwise_shuffle(i3_list, gcd_list):
    """Shuffles the I3 file list and the correponding gcd file list.

    This is handy because it ensures a more even extraction load for each worker.

    Args:
        files_list (list): List of I3 file paths.
        gcd_list (list): List of corresponding gcd file paths.

    Returns:
        i3_shuffled (list): List of shuffled I3 file paths.
        gcd_shuffled (list): List of corresponding gcd file paths.
    """
    df = pd.DataFrame({'i3': i3_list, 'gcd': gcd_list})
    df_shuffled = df.sample(frac=1, replace=False)
    i3_shuffled = df_shuffled['i3'].tolist()
    gcd_shuffled = df_shuffled['gcd'].tolist()
    return i3_shuffled, gcd_shuffled

def find_i3_files(directories, gcd_rescue):
    """Finds I3 files and corresponding GCD files in `directories`.

    Finds I3 files in dir and matches each file with a corresponding GCD file if
    present in the directory, matches with gcd_rescue if gcd is not present in
    the directory.

    Args:
        directories (list[str]): Directories to search recursively for I3 files.
        gcd_rescue (str): Path to the GCD that will be default if no GCD is
            present in the directory.

    Returns:
        i3_list (list[str]): Paths to I3 files in `directories`
        gcd_list (list[str]): Paths to GCD files for each I3 file.
    """
    # Output containers
    i3_files = []
    gcd_files = []

    for directory in directories:
        # Recursivley find all I3-like files in `directory`.
        i3_pattern = '*.i3.*'
        paths = list(Path(directory).rglob(i3_pattern))

        # Loop over all folders containing such I3-like files.
        folders = sorted(set([os.path.dirname(path) for path in paths]))
        for folder in folders:
            # List all I3 and GCD files, respectively, in the current folder.
            folder_files = glob(os.path.join(folder, i3_pattern))
            folder_i3_files = list(filter(is_i3_file, folder_files))
            folder_gcd_files = list(filter(is_gcd_file, folder_files))

            # Make sure that no more than one GCD file is found; and use rescue file of none is found.
            assert len(folder_gcd_files) <= 1
            if len(folder_gcd_files) == 0:
                folder_gcd_files = [gcd_rescue]

            # Store list of I3 files and corresponding GCD files.
            folder_gcd_files = folder_gcd_files * len(folder_i3_files)
            gcd_files.extend(folder_gcd_files)
            i3_files.extend(folder_i3_files)
            pass
        pass

    return i3_files, gcd_files

def frame_has_key(frame, key: str):
    """Returns whether `frame` contains `key`."""
    try:
        frame[key]
        return True
    except:
        return False
