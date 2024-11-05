"""Module containing different I3Reader."""

from typing import List, Union, OrderedDict, Optional

from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube.utilities.i3_filters import (
    I3Filter,
    NullSplitI3Filter,
)
from graphnet.data.extractors.icecube import I3Extractor
from graphnet.data.dataclasses import I3FileSet
from graphnet.utilities.filesys import find_i3_files
from .graphnet_file_reader import GraphNeTFileReader


if has_icecube_package():
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class I3Reader(GraphNeTFileReader):
    """A class for reading .i3 files from the IceCube Neutrino Observatory.

    Note that this class relies on IceCube-specific software, and therefore
    must be run in a software environment that contains IceTray.
    """

    def __init__(
        self,
        gcd_rescue: str,
        i3_filters: Optional[Union[I3Filter, List[I3Filter]]] = None,
        icetray_verbose: int = 0,
    ):
        """Initialize `I3Reader`.

        Args:
            gcd_rescue: Path to a GCD file that will be used if no GCD file is
                        found in subfolder. `I3Reader` will recursively search
                        the input directory for I3-GCD file pairs. By IceCube
                        convention, a folder containing i3 files will have an
                        accompanying GCD file. However, in some cases, this
                        convention is broken. In cases where a folder contains
                        i3 files but no GCD file, the `gcd_rescue` is used
                        instead.
            i3_filters: Instances of `I3Filter` to filter PFrames. Defaults to
                        `NullSplitI3Filter`.
            icetray_verbose: Set the level of verbosity of icetray.
                             Defaults to 0.
        """
        # checks
        assert isinstance(gcd_rescue, str)
        # Set verbosity
        if icetray_verbose == 0:
            icetray.I3Logger.global_logger = icetray.I3NullLogger()

        if i3_filters is None:
            i3_filters = [NullSplitI3Filter()]
        # Set Member Variables
        self._accepted_file_extensions = [".bz2", ".zst", ".gz"]
        self._accepted_extractors = [I3Extractor]
        self._gcd_rescue = gcd_rescue
        self._i3filters = (
            i3_filters if isinstance(i3_filters, list) else [i3_filters]
        )

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def __call__(
        self, file_path: I3FileSet
    ) -> List[OrderedDict]:  # noqa: E501  # type: ignore
        """Extract data from single I3 file.

        Args:
            fileset: Path to I3 file and corresponding GCD file.

        Returns:
            Extracted data.
        """
        # Set I3-GCD file pair in extractor
        for extractor in self._extractors:
            assert isinstance(extractor, I3Extractor)
            extractor.set_gcd(
                i3_file=file_path.i3_file, gcd_file=file_path.gcd_file
            )

        # Open I3 file
        i3_file_io = dataio.I3File(file_path.i3_file, "r")
        data = list()
        while i3_file_io.more():
            try:
                frame = i3_file_io.pop_physics()
            except Exception as e:
                if "I3" in str(e):
                    continue
            # check if frame should be skipped
            if self._skip_frame(frame):
                continue

            # Try to extract data from I3Frame
            results = [extractor(frame) for extractor in self._extractors]

            data_dict = OrderedDict(zip(self.extracor_names, results))

            data.append(data_dict)
        return data

    def find_files(self, path: Union[str, List[str]]) -> List[I3FileSet]:
        """Recursively search directory for I3 and GCD file pairs.

        Args:
            path: directory to search recursively.

        Returns:
            List I3 and GCD file pairs as I3FileSets
        """
        # Find all I3 and GCD files in the specified directories.
        i3_files, gcd_files = find_i3_files(
            path,
            self._gcd_rescue,
        )
        # checks
        assert len(i3_files) == len(gcd_files)

        # Pack as I3FileSets
        filesets = []
        for i3_file, gcd_file in zip(i3_files, gcd_files):
            assert isinstance(i3_file, str)
            assert isinstance(gcd_file, str), print(gcd_file, self._gcd_rescue)
            filesets.append(I3FileSet(i3_file, gcd_file))

        return filesets

    def _skip_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check the user defined filters.

        Returns:
            bool: True if frame should be skipped, False otherwise.
        """
        if self._i3filters is None:
            return False  # No filters defined, so we keep the frame

        for filter in self._i3filters:
            if not filter(frame):
                return True  # keep_frame call false, skip the frame.
        return False  # All filter keep_frame calls true, keep the frame.
