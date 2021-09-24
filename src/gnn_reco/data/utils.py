"""Utility function relevant to the gnn_reco.data package.
"""

import os
import pandas as pd
import re

def create_out_directory(outdir):
    try:
        os.makedirs(outdir)
    except:
        print(f"Directory {outdir} already exists")

def is_i3_file(filename):
    """Check whether `filename` is an I3 file."""
    if re.search('(gcd|geo)', filename.lower()):
        return False
    elif re.search(r'\.i3\.', filename.lower()):
        return True
    return False

def has_extension(filename, extensions):
    """Checks if the file has the desired extension.

    Args:
        filename (str): File name.
        extensions (list[str]): List of accepted extensions.

    Returns:
        boolean: True if accepted extension is detected, False otherwise
    """
    return re.search('(' + '|'.join(extensions) + ')$', filename) is not None

def pairwiseshuffle(files_list, gcd_list):
    """Shuffles the i3 file list and the correponding gcd file list.
    
    This is handy because it ensures a more even extraction load for each worker

    Args:
        files_list (list): List of i3 file paths.
        gcd_list (list): List of corresponding gcd file paths.

    Returns:
        i3_shuffled (list): List of shuffled i3 file paths.
        gcd_shuffled (list): List of corresponding gcd file paths.
    """
    df = pd.DataFrame({'i3': files_list, 'gcd': gcd_list})
    df_shuffled = df.sample(frac = 1)
    i3_shuffled = df_shuffled['i3'].tolist()
    gcd_shuffled = df_shuffled['gcd'].tolist()
    return i3_shuffled, gcd_shuffled
    
def save_filenames(input_files,outdir, db_name):
    """Saves i3 file names in csv

    Args:
        input_files (list): List of file names.
        outdir (str): Out directory path.
        db_name (str): Name of the database.
    """
    create_out_directory(outdir + '/%s/config'%db_name)
    input_files = pd.DataFrame(input_files)
    input_files.columns = ['filename']
    input_files.to_csv(outdir + '/%s/config/i3files.csv'%db_name)
    return

def find_files(paths,outdir,db_name,gcd_rescue, extensions = None):
    """Loops over paths and returns the corresponding i3 files and gcd files

    Args:
        paths (list): list of paths
        outdir (str): path to out directory
        db_name (str): name of the database
        gcd_rescue (str): path to the gcd file that will be defaulted to if no gcd is present in a directory
        extensions (list, optional): list of accepted extensions. E.g. (i3.zst). Defaults to None.

    Returns:
        input_files (list): a list of paths to i3 files
        gcd_files (list): a list of corresponding gcd files
    """
    print('Counting files in: \n%s\n This might take a few minutes...'%paths)
    if extensions == None:
        extensions = ("i3.bz2",".zst",".gz")
    input_files_mid = []
    input_files = []
    files = []
    gcd_files_mid = []
    gcd_files = []
    for path in paths:
        input_files_mid, gcd_files_mid = find_i3_files(path, extensions, gcd_rescue)
        input_files.extend(input_files_mid)
        gcd_files.extend(gcd_files_mid)

    if len(input_files) > 0:
        input_files, gcd_files = pairwiseshuffle(input_files, gcd_files)
        save_filenames(input_files, outdir, db_name)
    return input_files, gcd_files

def find_i3_files(dir, extensions, gcd_rescue):
    """Finds i3 files in dir and matches each file with a corresponding gcd_file if present in the directory, matches with gcd_rescue if gcd is not present in the directory

    Args:
        dir (str): path to scan recursively (2 layers deep by IceCube convention)
        extensions (list): list of accepted file extensions. E.g. i3.zst
        gcd_rescue (path): path to the gcd that will be default if no gcd is present in the directory

    Returns:
        files_list (list): a list containg paths to i3 files in dir
        GCD_list   (list): a list containing paths to gcd_files for each i3-file in dir
    """
    files_list = []
    GCD_list   = []
    root,folders,root_files = next(os.walk(dir))
    gcds_root = []
    gcd_root = None
    i3files_root = []
    for file in root_files:
        if has_extension(file, extensions):
            if is_i3_file(file):
                i3files_root.append(os.path.join(root,file))
            else:
                gcd_root = os.path.join(root,file)
                gcds_root.append(os.path.join(root,file))
    if gcd_root == None:
        gcd_root = gcd_rescue
    for k in range(len(i3files_root) - len(gcds_root)):
        gcds_root.append(gcd_root)
    files_list.extend(i3files_root)
    GCD_list.extend(gcds_root)
    for folder in folders:
        sub_root, sub_folders, sub_folder_files = next(os.walk(os.path.join(root,folder)))
        gcds_folder = []
        gcd_folder = None
        i3files_folder = []
        for sub_folder_file in sub_folder_files:
            if has_extension(sub_folder_file, extensions):
                if is_i3_file(sub_folder_file):
                    i3files_folder.append(os.path.join(sub_root,sub_folder_file))
                else:
                    gcd_folder = os.path.join(sub_root,sub_folder_file)
                    gcds_folder.append(os.path.join(sub_root,sub_folder_file))
        if gcd_folder == None:
            gcd_folder = gcd_rescue
        for k in range(len(i3files_folder) - len(gcds_folder)):
            gcds_folder.append(gcd_folder)
        files_list.extend(i3files_folder)
        GCD_list.extend(gcds_folder)
    
    files_list, GCD_list = pairwiseshuffle(files_list, GCD_list)
    return files_list, GCD_list
