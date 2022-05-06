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

class FiniteList(object) :
    '''
    A list which can become full
    '''

    def __init__(self,limit) :
        self.data = list()
        self.limit = limit

    def __init__(self,limit) :
        return len(self.data)

    def full(self) :
        if len(self.data) > limit :
            raise Exception("Finite list has overflowed")
        else :
            return len(self.data) == limit

    def append(self,v) :
        if self.full() :
            return False
        else :
            self.data.append(v)
            return True



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


class RelationalDict(object) :
    '''
    A relational dict, a powerful container
    Acts like a relational database (like SQL), e.g. data is stored as rows of columns, and can lookup based on table values

    #TODO Could this be replaced with Pandas DataFrame?
    #TODO Have issue that cannot modify mutable types in the returned row, due to fact that I build a dict for that row. Think about how do do this better...
    #TODO Add option to add new columns
    #TODO Add a bunch of clever >/< etc operations (maybe call it 'filter'), or exclude options (could pass a python string expression with the action, or a lambda)
    #TODO Add to/from json option
    #TODO Add option to draw as matplotlib table (maybe by integrating with utils.table.Table class)
    '''

    def __init__(self,column_headers) :

        #TODO use dicts of dicts of dicts etc for faster look up? Might be massive memory bartprint though...

        if len(column_headers) == 0 : 
            raise Exception("Cannot init RelationalDict : User did not provide any column headers")

        if "index" in column_headers :
            raise Exception("Cannot init RelationalDict : 'index' is a protected heading")

        #Store table as a dict, where the keys are the column headers and each key value is a list representing with one element per row
        self.table = dict( [ (ch,[]) for ch in column_headers ] )


    def get_num_rows(self) :
        return len(list(self.table.values())[0])


    def __len__(self) :
        return self.get_num_rows()


    def get_row(self,i) : #returns a row as a dict, where the keys are the column headers and the values are the values for that column in that row
        if i >= self.get_num_rows() :
            raise Exception( "Cannot get row from RelationalDict : Requested row %i is out of range [0,%i]" % (i,self.get_num_rows()) )
        row = dict( (header,column[i]) for header,column in list(self.table.items()) ) 
        row["index"] = i
        return row


    def get_column_headers(self) :
        return list(self.table.keys())


    def insert(self,table_vals) : #Use as: myTable.insert( { "headerA":1 , "headerB":4 } ), must specify all values

        #Check input columns are same length as table columns
        if len(self.table) != len(table_vals) : 
            raise Exception( "Cannot insert into RelationalDict : Mismatch in number of columns : %s vs %s" % (list(self.table.keys()),list(table_vals.keys())) )

        #Check all headings are present
        for column_header in self.get_column_headers() :
            if column_header not in table_vals : raise Exception( "Cannot insert into RelationalDict : Column '%s' is not specified" % column_header )

        #Loop over columns
        for column_header in self.get_column_headers() :

            #Add value to column as a new row
            self.table[column_header].append( table_vals[column_header] )


    #Set particular element in a row (e.g. the value at a single column in that row) #TOD Maybe use getIndex instead? Or somehow return the whole row as reference? (this is a bit clumsy and could do with a rethink, maybe store rows instead of columns)
    def set_element(self,search_terms,column_header,val) :

        row_nums = self._find_row_nums(search_terms)

        if len(row_nums) == 0 :
            raise Exception("Cannot set element in column %s in row in RelationalDict : Cannot find row" % column_header )

        elif len(row_nums) > 1 :
            raise Exception("Cannot set element in column %s in row in RelationalDict : %i matches for row" % (column_header,len(row_nums)) ) 

        else :
            rowNum = row_nums[0]
            if column_header not in self.get_column_headers() :
                raise Exception("Cannot set element in column %s in row in RelationalDict : Column does not exist" % column_header ) 
            else :
                self.table[column_header][rowNum] = val


    #Set particular element in a row (e.g. the value at a single column in that row) #TOD Maybe use getIndex instead? Or somehow return the whole row as reference? (this is a bit clumsy and could do with a rethink, maybe store rows instead of columns)
    def set_row(self,row) :

        for k,v in list(row.items()) :

            if k == "index" : continue

            if k not in list(self.table.keys()) :
                raise Exception("Cannot set element in column %s in row in RelationalDict : Column does not exist" % k ) 
            else :
                self.table[k][row["index"]] = v


    def _find_row_nums(self,search_terms=None) :

        #TODO use index

        output_row_nums = list()

        #Check no unknown table columns are specified
        if search_terms != None :
            for column_header in list(search_terms.keys()) :
                if column_header not in self.table : raise Exception( "Cannot find in RelationalDict : Unknown column %s specified" % column_header )

        #Loop over rows
        for i in range(0,self.get_num_rows()) :

            #Check if row matches column values specified 
            #To do this, loop through all search terms and check every one of them matches the row
            this_row_matches = True
            if search_terms is not None :
                for column_header in list(search_terms.keys()) :
                    try : #Use try block to handle thing with no comparison operation possible
                        if self.table[column_header][i] != search_terms[column_header] :
                            this_row_matches = False
                            break
                    except : 
                        pass

            #If found match, add this row to the output data
            if this_row_matches : 
                output_row_nums.append(i)

        return output_row_nums


    #Find all rows matching the search terms and return them
    def find(self,search_terms=None) :
        return [ self.get_row(i) for i in self._find_row_nums(search_terms) ]


    #Return all rows
    def get_all(self) :
        return self.find(search_terms=None)


    #Get a specific element that uniquely matches the search terms, and throw if fails
    def get(self,search_terms) :
        rows = self.find(search_terms)
        if len(rows) == 0 :
            raise Exception( "Cannot get from RelationalDict : No matches for %s" % search_terms )
        elif len(rows) > 1 :
            raise Exception( "Cannot get from RelationalDict : %i matches for %s" % (len(rows),search_terms) )
        else :
            return rows[0]


    def get_column_values(self,column_header,search_terms=None,unique=False) : #get all values found in a column (optionally can get only unique values, e.g. like a set)

        #Check column exists
        if column_header not in self.table : raise Exception( "Cannot get column values from RelationalDict : Unknown column '%s' specified" % column_header )

        #Get the column values (matching the search terms if user provided any)
        rows = self.find(search_terms)
        columnValues = [ row[column_header] for row in rows ]

        #Trim list to give only unique values if required
        if unique == True : 
            columnValues = trim_duplicates(columnValues)

        return columnValues

