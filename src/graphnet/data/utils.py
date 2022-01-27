"""Utility function relevant to the graphnet.data package."""

from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.path as mpath
from pathlib import Path
import re
from typing import List, Tuple
import sqlite3


def get_desired_event_numbers(db_path, desired_size, fraction_noise=0, fraction_nu_e=0, fraction_muon=0, fraction_nu_mu = 0, fraction_nu_tau=0, seed=0):
    assert fraction_nu_e + fraction_muon + fraction_nu_mu + fraction_nu_tau + fraction_noise == 1.0, 'Sum of fractions not equal to one.'
    rng = np.random.RandomState(seed=seed)

    fracs = [fraction_noise,fraction_muon,fraction_nu_e,fraction_nu_mu,fraction_nu_tau]
    numbers_desired = [int(x * desired_size) for x in fracs]
    pids = [1,13,12,14,16]

    with sqlite3.connect(db_path) as con:
        total_query = 'SELECT event_no FROM truth WHERE abs(pid) IN {}'.format(tuple(pids))
        tot_event_nos = pd.read_sql(total_query,con)
        if len(tot_event_nos) < desired_size:
            desired_size = len(tot_event_nos)
            numbers_desired = [int(x * desired_size) for x in fracs]
            print('Only {} events in database, using this number instead.'.format(len(tot_event_nos)))

        list_of_dataframes = []
        restart_trigger = True
        while restart_trigger:
            restart_trigger = False
            for number,particle_type in zip(numbers_desired,pids):
                query_is = 'SELECT event_no FROM truth WHERE abs(pid) == {}'.format(particle_type)
                tmp_dataframe = pd.read_sql(query_is,con)
                try:
                    dataframe = tmp_dataframe.sample(number, replace=False, random_state=rng).reset_index(drop=True)#could add weights (re-weigh) here with replace=True
                except ValueError:
                    if len(tmp_dataframe)==0:
                        print('There are no particles of type {} in this database please make new request.'.format(particle_type))
                        return None
                    print('There have been {} requested of particle {}, we can only supply {}. \nRenormalising...'.format(number,particle_type,len(tmp_dataframe)))
                    numbers_desired = [int(new_x * (len(tmp_dataframe)/number)) for new_x in numbers_desired]
                    restart_trigger = True
                    list_of_dataframes = []
                    break

                list_of_dataframes.append(dataframe)
        retrieved_event_nos_pd = pd.concat(list_of_dataframes)
        event_no_list = retrieved_event_nos_pd.sample(frac=1, replace=False, random_state=rng).values.ravel().tolist()

    return event_no_list


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
        query = f'select event_no from truth where abs(pid) != 13 and abs(pid) != 1 and event_no not in {train_event_nos}'
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
    elif has_extension(filename, ['bz2', 'zst']):
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
        i3_patterns = ['*.bz2', '*.zst']
        for i3_pattern in i3_patterns:
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

def muon_stopped(truth, horizontal_pad = 100, vertical_pad = 100):
    '''
    Determine if a simulated muon stops inside the fiducial volume of the IceCube detector as defined in the RIDE study.
    Vertical and horizontal pad shrink fiducial volume further.
    Using track length, azimuth and zenith travel from the starting/generation point,
    also return x,y,z of muon end point (to be reconstructed)
    '''
    #to do:remove hard-coded border coords and replace with GCD file contents using string no's
    border_coords = np.array([(-256.1400146484375, -521.0800170898438), (-132.8000030517578, -501.45001220703125), (-9.13000011444092, -481.739990234375), (114.38999938964844, -461.989990234375), (237.77999877929688, -442.4200134277344), (361.0, -422.8299865722656), (405.8299865722656, -306.3800048828125), (443.6000061035156, -194.16000366210938), (500.42999267578125, -58.45000076293945), (544.0700073242188, 55.88999938964844), (576.3699951171875, 170.9199981689453), (505.2699890136719, 257.8800048828125), (429.760009765625, 351.0199890136719), (338.44000244140625, 463.7200012207031), (224.5800018310547, 432.3500061035156), (101.04000091552734, 412.7900085449219), (22.11000061035156, 509.5), (-101.05999755859375, 490.2200012207031), (-224.08999633789062, 470.8599853515625), (-347.8800048828125, 451.5199890136719), (-392.3800048828125, 334.239990234375), (-437.0400085449219, 217.8000030517578), (-481.6000061035156, 101.38999938964844), (-526.6300048828125, -15.60000038146973), (-570.9000244140625, -125.13999938964844), (-492.42999267578125, -230.16000366210938), (-413.4599914550781, -327.2699890136719), (-334.79998779296875, -424.5)])
    border_z = [-512.82,524.56]
    border = mpath.Path(border_coords)

    start_pos = np.array([truth['position_x'],
                          truth['position_y'],
                          truth['position_z']])
                          
    travel_vec = -1*np.array([truth['track_length']*np.cos(truth['azimuth'])*np.sin(truth['zenith']),
                           truth['track_length']*np.sin(truth['azimuth'])*np.sin(truth['zenith']),
                           truth['track_length']*np.cos(truth['zenith'])])
    
    end_pos = start_pos+travel_vec

    stopped_xy = border.contains_point((end_pos[0],end_pos[1]),radius=-horizontal_pad) 
    stopped_z = (end_pos[2] > border_z[0] + vertical_pad) * (end_pos[2] < border_z[1] - vertical_pad) 
    stopped_muon = stopped_xy * stopped_z 

    return end_pos,stopped_muon