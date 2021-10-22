import os
import pandas as pd
import sqlite3

def check_db_size(db):
    max_size = 5000000
    with sqlite3.connect(db) as con:
        query = 'select event_no from truth'
        events =  pd.read_sql(query,con)
    if len(events) > max_size:
        events = events.sample(max_size)
    return events        
