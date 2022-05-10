"""
Tom Stuttard
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Unix specific-tools
# Tom Stuttard

from future import standard_library

standard_library.install_aliases()
import os


#
# Shebangs
#

BASH_SHEBANG = "#!/usr/bin/env bash"


#
# Implement python versions of unix command line applications
#

# tail
# Stolen from http://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-with-python-similar-to-tail
def tail(filePath, numLines):
    if os.path.exists(filePath):
        stdin, stdout = os.popen2("tail -n %i %s " % (numLines, filePath))
        stdin.close()
        linesInFile = stdout.readlines()
        stdout.close()
        return linesInFile
    else:
        raise Exception('File "%s" does not exist' % (filePath))


# which
# Stolen from: https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
