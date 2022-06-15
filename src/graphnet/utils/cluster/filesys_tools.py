"""
Tools for file system operations

Tom Stuttard
"""

import os, datetime, stat

# from graphnet.utils.cluster import containers

#
# Tools for handling files and directories
#

# Create a directory with robust error handling, including race conditions
def make_dir(dir_path, raise_exception=False):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            return True
        except OSError:
            assert os.path.isdir(
                dir_path
            ), 'Could not create directory "%s"' % (dir_path)
    return False


# Make a directory with the data/time as the name
TMP_FILE_STRFTIME = "%Y-%m-%d_%H-%M-%S"


def make_tmp_dir(parent_dir):
    dir_path = os.path.join(
        parent_dir, datetime.datetime.now().strftime(TMP_FILE_STRFTIME)
    )
    make_dir(dir_path)
    return dir_path


# Check file is executable
def is_executable(file_path):
    if os.path.exists(file_path):
        return os.stat(file_path).st_mode & stat.S_IEXEC > 0
