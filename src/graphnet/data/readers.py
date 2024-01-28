"""Module containing different FileReader classes in GraphNeT.

These methods are used to open and apply `Extractors` to experiment-specific
file formats.
"""

from typing import List, Union, OrderedDict, Type
from abc import abstractmethod, ABC
import glob
import os

from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger
from graphnet.utilities.imports import has_icecube_package
from graphnet.data.filters import I3Filter, NullSplitI3Filter

from .dataclasses import I3FileSet

from .extractors.extractor import (
    Extractor,
    I3Extractor,
)  # , I3GenericExtractor
from graphnet.utilities.filesys import find_i3_files

if has_icecube_package():
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class GraphNeTFileReader(Logger, ABC):
    """A generic base class for FileReaders in GraphNeT.

    Classes inheriting from `GraphNeTFileReader` must implement a
    `__call__` method that opens a file, applies `Extractor`(s) and returns
    a list of ordered dictionaries.

    In addition, Classes inheriting from `GraphNeTFileReader` must set
    class properties `accepted_file_extensions` and `accepted_extractors`.
    """

    @abstractmethod
    def __call__(self, file_path: str) -> List[OrderedDict]:
        """Open and apply extractors to a single file.

        The `output` must be a list of dictionaries, where the number of events
        in the file `n_events` satisfies `len(output) = n_events`. I.e each
        element in the list is a dictionary, and each field in the dictionary
        is the output of a single extractor.
        """

    @property
    def accepted_file_extensions(self) -> List[str]:
        """Return list of accepted file extensions."""
        return self._accepted_file_extensions  # type: ignore

    @property
    def accepted_extractors(self) -> List[Extractor]:
        """Return list of compatible `Extractor`(s)."""
        return self._accepted_extractors  # type: ignore

    @property
    def extracor_names(self) -> List[str]:
        """Return list of table names produced by extractors."""
        return [extractor.name for extractor in self._extractors]  # type: ignore

    def find_files(
        self, path: Union[str, List[str]]
    ) -> Union[List[str], List[I3FileSet]]:
        """Search directory for input files recursively.

        This method may be overwritten by custom implementations.

        Args:
            path: path to directory.

        Returns:
            List of files matching accepted file extensions.
        """
        if isinstance(path, str):
            path = [path]
        files = []
        for dir in path:
            for accepted_file_extension in self.accepted_file_extensions:
                files.extend(glob.glob(dir + f"/*{accepted_file_extension}"))

        # Check that files are OK.
        self.validate_files(files)
        return files

    @final
    def set_extractors(self, extractors: List[Extractor]) -> None:
        """Set `Extractor`(s) as member variable.

        Args:
            extractors: A list of `Extractor`(s) to set as member variable.
        """
        self._validate_extractors(extractors)
        self._extractors = extractors

    @final
    def _validate_extractors(self, extractors: List[Extractor]) -> None:
        for extractor in extractors:
            try:
                assert isinstance(extractor, tuple(self.accepted_extractors))  # type: ignore
            except AssertionError as e:
                self.error(
                    f"{extractor.__class__.__name__} is not supported by {self.__class__.__name__}"
                )
                raise e

    @final
    def validate_files(
        self, input_files: Union[List[str], List[I3FileSet]]
    ) -> None:
        """Check that the input files are accepted by the reader.

        Args:
            input_files: Path(s) to input file(s).
        """
        for input_file in input_files:
            # Handle filepath vs. FileSet cases
            if isinstance(input_file, I3FileSet):
                self._validate_file(input_file.i3_file)
                self._validate_file(input_file.gcd_file)
            else:
                self._validate_file(input_file)

    @final
    def _validate_file(self, file: str) -> None:
        """Validate a single file path.

        Args:
            file: path to file.

        Returns:
            None
        """
        try:
            assert file.lower().endswith(tuple(self.accepted_file_extensions))
        except AssertionError:
            self.error(
                f'{self.__class__.__name__} accepts {self.accepted_file_extensions} but {file.split("/")[-1]} has extension {os.path.splitext(file)[1]}.'
            )


class I3Reader(GraphNeTFileReader):
    """A class for reading .i3 files from the IceCube Neutrino Observatory.

    Note that this class relies on IceCube-specific software, and therefore
    must be run in a software environment that contains IceTray.
    """

    def __init__(
        self,
        gcd_rescue: str,
        i3_filters: Union[
            Type[I3Filter], List[Type[I3Filter]]
        ] = NullSplitI3Filter,
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
        # Set verbosity
        if icetray_verbose == 0:
            icetray.I3Logger.global_logger = icetray.I3NullLogger()

        # Set Member Variables
        self._accepted_file_extensions = [".bz2", ".zst", ".gz"]
        self._accepted_extractors = [I3Extractor]
        self._gcd_rescue = gcd_rescue
        self._i3filters = (
            i3_filters if isinstance(i3_filters, list) else [i3_filters]
        )

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def __call__(self, file_path: I3FileSet) -> List[OrderedDict]:  # type: ignore
        """Extract data from single I3 file.

        Args:
            fileset: Path to I3 file and corresponding GCD file.

        Returns:
            Extracted data.
        """
        # Set I3-GCD file pair in extractor
        for extractor in self._extractors:
            extractor.set_files(file_path.i3_file, file_path.gcd_file)  # type: ignore

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

            # If an I3GenericExtractor is used, we want each automatically
            # parsed key to be stored as a separate table.
            # for extractor in self._extractors:
            #    if isinstance(extractor, I3GenericExtractor):
            #        data_dict.update(data_dict.pop(extractor._name))

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

        # Pack as I3FileSets
        filesets = [
            I3FileSet(i3_file, gcd_file)
            for i3_file, gcd_file in zip(i3_files, gcd_files)
        ]
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
