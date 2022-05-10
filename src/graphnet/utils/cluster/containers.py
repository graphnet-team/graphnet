"""
Some custom container types
Fro tools for working with standard containers, see container_tools.py

Tom Stuttard
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
from builtins import dict
from future import standard_library

standard_library.install_aliases()
from builtins import object

# from container_tools import trim_duplicates
# Return the list with duplicates trimmed (still keeping one of each duplicate), maintaining the order of the original list
# Note: Cannot just use list(set(l)), as the order of the list is not maintained by the set
def trim_duplicates(a_list):
    output_list = list()
    unique_elements = set()
    for val in a_list:
        if (
            val not in unique_elements
        ):  # e.g. is this value has not already been seen
            output_list.append(val)
            unique_elements.add(val)
    return output_list


#
# Smart containers
#
