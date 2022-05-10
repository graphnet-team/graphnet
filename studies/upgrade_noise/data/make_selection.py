from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import sqlite3
import tqdm


def parallel_this(settings):
    events, id, tmp_dir, db = settings
    con = sqlite3.connect(db)
    labels = []
    for index in tqdm.tqdm(events):
        features = con.execute(
            "SELECT event_no FROM SplitInIcePulses_GraphSage_Pulses WHERE event_no = {}".format(
                index
            )
        )
        features = features.fetchall()
        if len(features) > 10:
            labels.append(index)
    labels = pd.DataFrame(data=labels, columns=["event_no"])
    labels.to_csv("%s/tmp_%s.csv" % (tmp_dir, id))
    return


def merge_tmp(tmp_dir):
    tmps = os.listdir(tmp_dir)
    is_first = True
    for file in tmps:
        if ".csv" in file:
            if is_first:
                df = pd.read_csv(tmp_dir + "/" + file)
                is_first = False
            else:
                df = df.append(
                    pd.read_csv(tmp_dir + "/" + file), ignore_index=True
                )
    df = df.sort_values("event_no").reset_index(drop=True)
    return df


def over_10_pulses(n_workers, path, tmp_dir, db, events):
    settings = []
    event_batches = np.array_split(events.values.ravel().tolist(), n_workers)
    for i in range(n_workers):
        settings.append([event_batches[i], i, tmp_dir, db])
    p = Pool(processes=n_workers)
    p.map_async(parallel_this, settings)
    p.close()
    p.join()
    selection = merge_tmp(tmp_dir)
    selection.to_csv(path + "/over10pulses.csv")
    return


def make_even_track_cascade(events, db):
    with sqlite3.connect(db) as con:
        query = (
            "select event_no from truth where abs(pid) = 14 and interaction_type = 1 and event_no in %s"
            % str(tuple(events["event_no"]))
        )
        tracks = pd.read_sql(query, con)
        query = (
            "select event_no from truth where event_no not in %s and event_no in %s"
            % (str(tuple(tracks["event_no"])), str(tuple(events["event_no"])))
        )
        cascades = pd.read_sql(query, con)
    print("found %s tracks" % len(tracks))
    print("found %s cascades" % len(cascades))
    if len(tracks) > len(cascades):
        return (
            pd.concat(
                [tracks.sample(len(cascades)), cascades], ignore_index=True
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )
    else:
        return (
            pd.concat(
                [tracks, cascades.sample(len(tracks))], ignore_index=True
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    n_workers = 50
    db = "/mnt/scratch/rasmus_orsoe/databases/dev_step4_numu_140021_second_run/data/dev_step4_numu_140021_second_run.db"
    tmp_dir = "/home/iwsatlas1/oersoe/phd/upgrade_noise/tmp"
    path = "/mnt/scratch/rasmus_orsoe/databases/dev_step4_numu_140021_second_run/selection"
    with sqlite3.connect(db) as con:
        query = "select event_no from truth"
        events = pd.read_sql(query, con)
    over_10_pulses(n_workers, path, tmp_dir, db, events)
    events = pd.read_csv(
        path + "/over10pulses.csv",
    )
    selection = make_even_track_cascade(events, db)
    selection.to_csv(path + "/even_track_cascade_over10pulses.csv")
