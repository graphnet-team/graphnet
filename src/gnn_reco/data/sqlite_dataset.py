from torch_geometric.data import Dataset
import sqlite3
import pandas as pd

class SQLiteDataset(Dataset):
    """Sqlite dataset class. Queries the database and extracts a batch of events as a pair of pandas.DataFrames

    Args:
        Dataset (torch.Dataset): [description]
    """
    def __init__(self, root, database, selection, pulsemap, batch_size, transform=None, pre_transform=None):
        """SQLite Dataset. Queries database and extracts a batch of events from selection as a pair of pandas.Dataframes

        Args:
            root (str): string to root directory. Currently unused
            database (str): path to database file.
            selection (str): path to .csv file containing the event_no from database that is desired for extraction
            pulsemap (str): the name of the pulse map, e.g. SRTInIcePulses
            batch_size (int): the size of the batch
            transform ([type], optional): [description]. Defaults to None.
            pre_transform ([type], optional): [description]. Defaults to None.
        """
        super().__init__(root,transform, pre_transform)
        self.database = database
        self.selection = pd.read_csv(selection).reset_index(drop = True)
        self.pulsemap = pulsemap
        self.batch_size = batch_size
        self._sample_generator()
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def len(self):
        """Calculates the length of the dataset, e.g. the number of batches given self.batch_size

        Returns:
            int: length of the dataset
        """
        return len(self.event_batches)

    def _sample_generator(self):
        """Uses self.batch_size to build batches from self.selection
        """
        self.event_batches = []
        for i in range(0, len(self.selection), self.batch_size):
            self.event_batches.append(str(tuple(self.selection['event_no'][i:i + self.batch_size])))
        return
    def get(self, idx):
        """Queries self.database and extracts the idx'th batch from self.event_batches

        Args:
            idx (int): batch index

        Returns:
            features (pandas.DataFrame): the pulsemap features 
            truth    (pandas.DataFrame): the truth variables
        """
        events = self.event_batches[idx]
        with sqlite3.connect(self.database) as con:
            query = 'select event_no, dom_x, dom_y, dom_z,dom_time, charge, rde, pmt_area from %s where event_no in %s'%(self.pulsemap, events)
            features = pd.read_sql(query, con)
            query = 'select event_no, energy, position_x, position_y, position_z, azimuth, zenith, pid, elasticity from truth where event_no in %s'%(events)
            truth = pd.read_sql(query,con)
        return features, truth