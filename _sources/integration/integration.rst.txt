.. include:: ../substitutions.rst

Integrating New Experiments into GraphNeT\ |graphnet-header|
============================================================

GraphNeT is built to host data conversion, model and deployment code from different neutrino telescopes and related experiments.
Part of the design is to minimize the technical overhead of implementing support for an experiment, and can typically be done with 200 - 300 lines of code.

.. note:: 
    The GraphNeT development team is willing to support the integration efforts of new experiments, so please consider reaching out to us if you need help.

A general outline of the steps to integrate an experiment into GraphNeT is outlined below.

**1) Adding Support for Your Data**
----------------------------

The most critical element of implementing support for an experiment into graphnet is an interface to data from your experiment.
This can be done in two ways:

- **a)** Adding a :code:`Dataset` class that are able to read your data directly during training
- **b)** Adding a :code:`GraphNeTFileReader` and associated :code:`Extractors` to convert your data to a supported data format.

Option **a)** is only viable if the data format from your experiment is suitable for deep learning. 

Option **b)** requires adding your own reader and defining extractors. Below is a step-by-step example 




Writing your own :code:`Extractor` and :code:`GraphNeTFileReader`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    We recommend visiting the :code:`DataConverter` documentation before reading this example

Suppose you want to add a reader and extractor for MyExperiment and that the data
is stored as pickled dictionaries with two kinds of entries:

**hits**: a single pandas.DataFrame with dimensions [n_pulses,d] where `n_pulses` is the number of observed Cherenkov pulses across all events in the file, and `d` is the number of features we know about each measurement.
These features would normally include the sensor position, time of measurement, event id etc. 

**truth**: a single pandas.DataFrame with dimensions [n_events, t] where `n_events` denotes the total number of events in the file and `t` is the number of event-level truth variables we have available for this simulation.
Ordinarily, these truth variables would include particle id, energy, direction etc. 


To convert these pickle files to a supported backend in GraphNeT, we first have to define our reader. This reader should open a pickle file and apply the :code:`Extractor` that we must also implement. Lets start with the reader:

.. code-block:: python

    from typing import List, Union, Dict
    import pandas as pd
    import pickle

    # Import the generic file reader
    from .graphnet_file_reader import GraphNeTFileReader

    # Import your own extractor
    from graphnet.data.extractors.myexperiment import MyExtractor

    class MyReader(GraphNeTFileReader):
        """A class for reading my pickle files from MyExperiment."""

        _accepted_file_extensions = [".pickle"]
        _accepted_extractors = [MyExperiment]

        def __call__(self, file_path: str) -> Dict[str, pd.DataFrame]:
            """Extract data from single pickle file.

            Args:
                file_path: Path to pickle file.

            Returns:
                Extracted data.
            """

            # Open file
            file = open(file_path,'r')
            data = pickle.load(file)

            # Apply extractors
            outputs = {}
            for extractor in self._extractors:
                output = extractor(data)
                if output is not None:
                    outputs[extractor._extractor_name] = output
            return outputs

When the :code:`DataConverter` is instantiated, it will set the :code:`Extractors` that it was instantiated with as member
variables of our :code:`GraphNeTFileReader`, making them available to us under :code:`self._extractors`. When the conversion is running, 
the :code:`DataConverter` will pass a :code:`file_path` to our :code:`__call__` function, and it is the job of our reader to open this
file and apply extractors to it. These calls will happen in parallel automatically.

So - the reader above first opens the `.pickle` file, and then applies the extractors. Job done! Let's now define the extractor:


The purpose of an :code:`Extractor` is to extract only part of the information available in files. When an :code:`Extractor`  is instantiated, it is given a name:

.. code-block:: python

    extractor = MyExtractor(extractor_name = "hits")

and this name is used to select specific tables in the file.

.. code-block:: python

    from typing import Dict

    from graphnet.data.extractors import Extractor

    class MyExtractor(Extractor):
        """
        Class for extracting information from pickles files in MyExperiment
        """

        def __call__(self, dictionary: Dict[str, pd.DataFrame]) -> pd.DataFrame:
            """Extract information from pickle file."""

            # Check if the table is in the dict
            if self._extractor_name in dictionary.keys():
                return dictionary[self._extractor_name]
            else:
                return None

We defined our reader in such a way that our extractor recieves a :code:`dictionary: Dict[str, pd.DataFrame]` argument. Our extractor therefore only have to find the field it needs to extract and return it.


With both a reader and extractor defined, we're ready to convert data to a supported backend in GraphNeT! Below is an example of using the code above in conversion:

.. code-block::

    from graphnet.data.extractors.myexperiment import MyExtractor
    from graphnet.data.dataconverter import DataConverter
    from graphnet.data.readers import MyReader
    from graphnet.data.writers import ParquetWriter

    # Your settings
    dir_with_files = '/home/my_files'
    outdir = '/home/my_outdir'
    num_workers = 5

    # Instantiate DataConverter - exports data from MyExperiment to Parquet
    converter = DataConverter(file_reader = MyReader(),
                              save_method = ParquetWriter(),
                              extractors=[MyExtractor('hits'), MyExtractor('truth')],
                              outdir=outdir,
                              num_workers=num_workers,
                            )
    # Run Converter
    converter(input_dir = dir_with_files)
    # Merge files (Optional)
    converter.merge_files()





**2) Implementing a Detector Class**
-----------------------------

GraphNeT requires a :code:`Detector` class to represent details that are specific to your experiment.

a :code:`Detector` holds a geometry table, standardization functions that maps your raw data into a numerical range suitable for deep learning, and names of important columns in your data.

Below is an example of a :code:`Detector` class:

.. code-block:: python

    from graphnet.models.detector import Detector

    class MyDetector(Detector):
        """`Detector` class for my experiment."""

        geometry_table_path = "path_to_geometry_table.parquet"
        xyz = ["sensor_x", "sensor_y", "sensor_z"]
        string_id_column = "string_id"
        sensor_id_column = "sensor_id"

        def feature_map(self) -> Dict[str, Callable]:
            """Map standardization functions to each dimension of input data."""
            feature_map = {
                "sensor_x": self._sensor_xyz,
                "sensor_y": self._sensor_xyz,
                "sensor_z": self._sensor_xyz,
                "sensor_time": self._sensor_time,
            }
            return feature_map

        def _sensor_xyz(self, x: torch.tensor) -> torch.tensor:
            return x / 500.0

        def _sensor_time(self, x: torch.tensor) -> torch.tensor:
            return (x - 1.0e04) / 3.0e4

:code:`feature_map` is a function that maps a standardization function to each possible feature from your experiment. 
The class variable :code:`xyz` contains the names of the xyz-position of sensors in your detector.
:code:`string_id_column` and :code:`sensor_id_column` holds the names of the columns in your input data that contain the string and sensor ids, respectively.

Lastly, :code:`geometry_table_path` points to a file that you should add to :code:`graphnet/data/geometry_tables/name-of-your-experiment/name-of-detector.parquet`.
A geometry table is an array containing all sensors in your experiment and has dimensions [n, d] where `n` denotes the number of sensors in your detector and `d` is the number of available features.

Suppose the detector represented by the Detector class above had 5 sensors in total on one string, then the corresponding geometry table would be:

.. list-table:: Example of geometry table before applying multi-index
   :widths: 20 20 20 20 20 20
   :header-rows: 1

   * - sensor_x
     - sensor_y
     - sensor_z
     - sensor_time
     - string_id
     - sensor_id
   * - 10
     - 10
     - 10
     - 1
     - 0
     - 0
   * - 20
     - 20
     - 20
     - 1
     - 0
     - 1
   * - 30
     - 30
     - 30
     - 1
     - 0
     - 2
   * - 40
     - 40
     - 40
     - 1
     - 0
     - 3
   * - 50
     - 50
     - 50
     - 1
     - 0
     - 4
 
Here, every row represents a unique sensor identified by :code:`sensor_id`.
GraphNeT will use this id to add/remove/filter sensors from your training examples, if you specify so in your data representations.

To convert the table above into a geometry table, you must set a :code:`MultiIndex` on the xyz position variables, and save it as :code:`.parquet`:

.. code-block:: python

    import pandas as pd

    path_to_array = 'my_table.csv'

    table_without_index = pd.read_csv(path_to_array)
    geometry_table = table_without_index.set_index(['sensor_x','sensor_y','sensor_z'])
    geometry_table.to_parquet('my_geometry_table.parquet')

here :code:`'my_table.csv'` refers to the table above, and the resulting :code:`'my_geometry_table.parquet'` would be the file to include under :code:`graphnet/data/geometry_tables/name-of-your-experiment/`


