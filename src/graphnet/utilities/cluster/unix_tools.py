"""
Tom Stuttard
"""
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
