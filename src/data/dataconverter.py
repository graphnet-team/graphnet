from abc import ABCMeta, abstractmethod
from utils import I3Extractor
from utils import load_geospatial_data
from icecube import dataio

class DataConverter(ABCMeta):
    def __init__(self, paths, mode, pulsemap):
        self.paths = paths
        self._extractor = I3Extractor()
        self.mode = mode
        self.pulsemap = pulsemap

    def process_file(self, i3_file, gcd_file, mode, pulsemap):
        gcd_dict, calibration = load_geospatial_data(gcd_file)
        i3_file = dataio.I3File(i3_file, "r")

        while i3_file.more():
            try:
                frame = i3_file.pop_physics()
            except: 
                continue
            array = self._extractor(frame, mode, pulsemap, gcd_dict, calibration, i3_file)
            self._save(array)

    @abstractmethod
    def _processfiles(self):
        pass

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _initialise(self):
        pass