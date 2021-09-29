import torch
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')
from gnn_reco.data.sqlite_dataset import SQLiteDataset

class SQLiteIterator:
   ''' Iterator class '''
   def __init__(self, SQLiteDataLoader):
       self._SQLiteDataLoader = SQLiteDataLoader
       self._index = 0
   def __next__(self):
       ''''Returns the next batch from self._SQLiteDataloader.queue '''
       if self._index < (self._SQLiteDataLoader.len) :
           batch_of_graphs = self._SQLiteDataLoader._grab_graphs()
           self._index +=1
           return batch_of_graphs
       # End of Iteration
       raise StopIteration

class SQLiteDataLoader(object):
    """SQLite specific dataloader. Extracts truth and pulsemap data from the SQLiteDataset using multiprocessing.

        Args:
            dataset (SQLiteDataset): A SQLiteDataset instance
            batch_size (int): size of each batch
            num_workers (int): the number of workers used to extract from the dataset
            shuffle (bool, optional): shuffles the dataset.selection if True. Defaults to True.
    """
    def __init__(self,dataset, batch_size, num_workers, shuffle = True):
       super(SQLiteDataLoader, self).__init__()
       self.dataset = dataset
       self.batch_size = batch_size
       self.num_workers  = num_workers
       self.shuffle    = shuffle
       self.len = self.dataset.len()
       self.manager = torch.multiprocessing.Manager()
       self._setup_queue()     
    def _grab_graphs(self):
        """Extracts a batch of graphs froms self.queue.

        Returns:
            batch_of_graphs: A batch of graphs
        """
        queue_empty = self.queue.empty()
        while(queue_empty):
            queue_empty = self.queue.empty()
        batch_of_graphs = self.queue.get()
        return batch_of_graphs

    def _setup_queue(self):
        """Initializes the queue and spawns processes.
        """
        self.queue =  self.manager.Queue()
        workers = []
        batch_indicies = np.arange(0,self.len,1)
        if self.shuffle:
            np.random.shuffle(batch_indicies)

        batch_list = np.array_split(batch_indicies, self.num_workers)

        for i in range(self.num_workers):
            workers.append(torch.multiprocessing.Process(target=self._parallel_sqlite_extraction, args=([batch_list[i]])))
        for worker in workers:
            worker.start()
        return
    def _parallel_sqlite_extraction(self, settings):
        """The sqlite extraction function run in parallel by every worker. Reads self.dataset and puts the extracted quantities in self.queue

        Args:
            settings (list): a list of arguments
        """
        batch_list = settings
        for batch in batch_list:
            self.queue.put([self.dataset.get(batch)])
        return
        
    def __iter__(self):
        return SQLiteIterator(self)

    def __len__(self):
        return self.len