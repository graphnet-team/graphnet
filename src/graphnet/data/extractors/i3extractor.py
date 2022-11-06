"""Base I3Extractor class(es)."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import LoggerMixin

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class I3Extractor(ABC, LoggerMixin):
    """Base class for extracting information from physics I3-frames.

    All classes inheriting from `I3Extractor` should implement the `__call__`
    method, and can be applied directly on `icetray.I3Frame` objects to return
    extracted, pure-python data.
    """

    def __init__(self, name: str):
        """Construct I3Extractor.

        Args:
            name: Name of the `I3Extractor` instance. Used to keep track of the
                provenance of different data, and to name tables to which this
                data is saved.
        """
        # Member variable(s)
        self._i3_file: str = ""
        self._gcd_file: str = ""
        self._gcd_dict: Dict[int, Any] = {}
        self._calibration: Optional["icetray.I3Frame.Calibration"] = None
        self._name: str = name

    def set_files(self, i3_file: str, gcd_file: str) -> None:
        """Store references to the I3- and GCD-files being processed."""
        # @TODO: Is it necessary to set the `i3_file`? It is only used in one
        #        place in `I3TruthExtractor`, and there only in a way that might
        #        be solved another way.
        self._i3_file = i3_file
        self._gcd_file = gcd_file
        self._load_gcd_data()

    def _load_gcd_data(self) -> None:
        """Load the geospatial information contained in the GCD-file."""
        # If no GCD file is provided, search the I3 file for frames containing
        # geometry (G) and calibration (C) information.
        gcd_file = dataio.I3File(self._gcd_file or self._i3_file)

        try:
            g_frame = gcd_file.pop_frame(icetray.I3Frame.Geometry)
        except RuntimeError:
            self.error(
                "No GCD file was provided and no G-frame was found. Exiting."
            )
            raise
        else:
            self._gcd_dict = g_frame["I3Geometry"].omgeo

        try:
            c_frame = gcd_file.pop_frame(icetray.I3Frame.Calibration)
        except RuntimeError:
            self.warning("No GCD file was provided and no C-frame was found.")
        else:
            self._calibration = c_frame["I3Calibration"]

    @abstractmethod
    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract information from frame."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the `I3Extractor` instance."""
        return self._name


class I3ExtractorCollection(list):
    """Class to manage multiple I3Extractors."""

    def __init__(self, *extractors: I3Extractor):
        """Construct I3ExtractorCollection.

        Args:
            *extractors: List of `I3Extractor`s to be treated as a single
            collection.
        """
        # Check(s)
        for extractor in extractors:
            assert isinstance(extractor, I3Extractor)

        # Base class constructor
        super().__init__(extractors)

    def set_files(self, i3_file: str, gcd_file: str) -> None:
        """Store references to the I3- and GCD-files being processed."""
        for extractor in self:
            extractor.set_files(i3_file, gcd_file)

    def __call__(self, frame: "icetray.I3Frame") -> List[dict]:
        """Extract information from frame for each member `I3Extractor`."""
        return [extractor(frame) for extractor in self]
