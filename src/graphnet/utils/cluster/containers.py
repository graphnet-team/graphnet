'''
Some custom container types
Fro tools for working with standard containers, see container_tools.py

Tom Stuttard
'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
from builtins import dict
from future import standard_library
standard_library.install_aliases()
from builtins import object

#from container_tools import trim_duplicates
#Return the list with duplicates trimmed (still keeping one of each duplicate), maintaining the order of the original list
#Note: Cannot just use list(set(l)), as the order of the list is not maintained by the set 
def trim_duplicates(a_list) :
    output_list = list()
    unique_elements = set()
    for val in a_list :
        if val not in unique_elements : #e.g. is this value has not already been seen
            output_list.append(val)
            unique_elements.add(val)
    return output_list

#
# Smart containers
#




class Range(object) :
    '''
    Class which tracks a range of something, e.g. keep adding to it and it tracks the total range
    '''

    minVal = None
    maxVal = None

    def __init__(self,val=None) :
        if val is not None: self.add(val)

    def filled(self) :
        return False if self.minVal == None or self.maxVal == None else True 

    def add(self,val) :
        #Input can be list or scalar
        if is_sequence(val) :
            if len(val) == 0 : raise Exception("Range : Cannot add data vector, vector is empty")
            minInputVal = min(val)
            maxInputVal = max(val)
        else :
            minInputVal = val
            maxInputVal = val

        #Get min
        if self.minVal == None : self.minVal = minInputVal
        else : self.minVal = min(self.minVal,minInputVal)

        #Get max
        if self.maxVal == None : self.maxVal = maxInputVal
        else : self.maxVal = max(self.maxVal,maxInputVal)

    @property
    def min(self) : 
        if not self.filled() :
            raise Exception("Cannot return range minimum, have not yet added any values")
        else :
            return self.minVal

    @property
    def max(self) : 
        if not self.filled() :
            raise Exception("Cannot return range maximum, have not yet added any values")
        else :
            return self.maxVal

    @property
    def width(self) : 
        if not self.filled() :
            raise Exception("Cannot return range width, have not yet added any values")
        else :
            return self.maxVal - self.minVal

    @property
    def center(self) : 
        if not self.filled() :
            raise Exception("Cannot return range center, have not yet added any values")
        else :
            return self.min + ( self.width / 2. )

    def scale(self,factor) :
        if not self.filled() :
            raise Exception("Cannot scale range, have not yet added any values")
        else :
            new_width = self.width * factor
            self.minVal = self.center - ( new_width / 2. )
            self.maxVal = self.center + ( new_width / 2. )



