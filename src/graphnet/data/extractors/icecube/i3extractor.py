"""Base I3Extractor class(es)."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors import Extractor

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


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

    def set_gcd(self, i3_file: str, gcd_file: Optional[str] = None) -> None:
        """Extract GFrame and CFrame from i3/gcd-file pair.

           Information from these frames will be set as member variables of
           `I3Extractor.`

        Args:
            i3_file: Path to i3 file that is being converted.
            gcd_file: Path to GCD file. Defaults to None. If no GCD file is
                      given, the method will attempt to find C and G frames in
                      the i3 file instead. If either one of those are not
                      present, `RuntimeErrors` will be raised.
        """
        if gcd_file is None:
            # If no GCD file is provided, search the I3 file for frames
            # containing geometry (GFrame) and calibration (CFrame)
            gcd = dataio.I3File(i3_file)
        else:
            # Ideally ends here
            gcd = dataio.I3File(gcd_file)

        # Get GFrame
        try:
            g_frame = gcd.pop_frame(icetray.I3Frame.Geometry)
            # If the line above fails, it means that no gcd file was given
            # and that the i3 file does not have a G-Frame in it.
        except RuntimeError as e:
            self.error(
                "No GCD file was provided "
                f"and no G-frame was found in {i3_file.split('/')[-1]}."
            )
            raise e

        # Get CFrame
        try:
            c_frame = gcd.pop_frame(icetray.I3Frame.Calibration)
            # If the line above fails, it means that no gcd file was given
            # and that the i3 file does not have a C-Frame in it.
        except RuntimeError as e:
            self.warning(
                "No GCD file was provided and no C-frame "
                f"was found in {i3_file.split('/')[-1]}."
            )
            raise e

        # Save information as member variables of I3Extractor
        self._gcd_dict = g_frame["I3Geometry"].omgeo
        self._calibration = c_frame["I3Calibration"]

    @abstractmethod
    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract information from frame."""
        pass
