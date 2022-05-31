import os
import pandas as pd
import sqlite3
from copy import deepcopy

from graphnet.utilities.logging import get_logger


logger = get_logger()


def check_db_size(db):
    max_size = 5000000
    with sqlite3.connect(db) as con:
        query = "select event_no from truth"
        events = pd.read_sql(query, con)
    if len(events) > max_size:
        events = events.sample(max_size)
    return events


def fit_scaler(db, features, truth, pulsemap):
    features = deepcopy(features)
    truth = deepcopy(truth)
    # features.remove('event_no')
    # truth.remove('event_no')
    truth = ", ".join(truth)
    features = ", ".join(features)

    outdir = "/".join(db.split("/")[:-2])
    logger.info(os.path.exists(outdir + "/meta/transformers.pkl"))
    if os.path.exists(outdir + "/meta/transformers.pkl"):
        comb_scalers = pd.read_pickle(outdir + "/meta/transformers.pkl")
    else:
        logger.warning(
            "Directory '" + outdir + "/meta/transformers.pkl' doesn't exist."
        )

    return comb_scalers
