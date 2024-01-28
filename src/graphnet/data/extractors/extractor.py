"""Base I3Extractor class(es)."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class Extractor(ABC, Logger):
    """Base class for extracting information from data files.

    All classes inheriting from `Extractor` should implement the `__call__`
    method, and should return a pure python dictionary on the form

    output = [{'var1: ..,
                 ... ,
              'var_n': ..}]

    Variables can be scalar or array-like of shape [n, 1], where n denotes the
    number of elements in the array, and 1 the number of columns.

    An extractor is used in conjunction with a specific `FileReader`.
    """

    def __init__(self, extractor_name: str):
        """Construct Extractor.

        Args:
            extractor_name: Name of the `Extractor` instance. Used to keep track of the
                provenance of different data, and to name tables to which this
                data is saved. E.g. "mc_truth".
        """
        # Member variable(s)
        self._extractor_name: str = extractor_name

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract information from frame."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the `I3Extractor` instance."""
        return self._extractor_name


class I3Extractor(Extractor):
    """Base class for extracting information from physics I3-frames.

    Contains functionality required to extract data from i3 files, used by
    the IceCube Neutrino Observatory.

    All classes inheriting from `I3Extractor` should implement the `__call__`
    method.
    """

    def __init__(self, extractor_name: str):
        """Construct I3Extractor.

        Args:
            extractor_name: Name of the `I3Extractor` instance. Used to keep track of the
                provenance of different data, and to name tables to which this
                data is saved.
        """
        # Member variable(s)
        self._i3_file: str = ""
        self._gcd_file: str = ""
        self._gcd_dict: Dict[int, Any] = {}
        self._calibration: Optional["icetray.I3Frame.Calibration"] = None

        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def set_gcd(self, gcd_file: str, i3_file: str) -> None:
        """Load the geospatial information contained in the GCD-file."""
        # If no GCD file is provided, search the I3 file for frames containing
        # geometry (G) and calibration (C) information.
        gcd = dataio.I3File(gcd_file or i3_file)

        try:
            g_frame = gcd.pop_frame(icetray.I3Frame.Geometry)
        except RuntimeError:
            self.error(
                "No GCD file was provided and no G-frame was found. Exiting."
            )
            raise
        else:
            self._gcd_dict = g_frame["I3Geometry"].omgeo

        try:
            c_frame = gcd.pop_frame(icetray.I3Frame.Calibration)
        except RuntimeError:
            self.warning("No GCD file was provided and no C-frame was found.")
        else:
            self._calibration = c_frame["I3Calibration"]

    @abstractmethod
    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract information from frame."""
        pass
