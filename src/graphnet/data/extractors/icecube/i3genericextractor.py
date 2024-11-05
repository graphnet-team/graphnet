"""I3Extractor class(es) for generic data extraction."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from graphnet.data.extractors.icecube import I3Extractor
from graphnet.data.extractors.icecube.utilities.types import (
    cast_object_to_pure_python,
    cast_pulse_series_to_pure_python,
)
from graphnet.data.extractors.icecube.utilities.collections import (
    transpose_list_of_dicts,
    serialise,
    flatten_nested_dictionary,
)

from graphnet.utilities.imports import has_icecube_package


if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        dataclasses,
        icetray,
    )  # pyright: reportMissingImports=false


GENERIC_EXTRACTOR_NAME = "GENERIC"


class I3GenericExtractor(I3Extractor):
    """Dynamically and generically extract information from frames.

    This class parses all keys in the I3Frame objects it is called on, and
    tries to automatically cast all of the available information to pure-python
    classes. This is done recursively, for each object in the I3Frame, by
    looking for member variables that can be parsed; by looking for objects
    that have signatures similar to python lists or dicts; and by handling a
    handful of special cases:
    - Pulse series maps,
    - Per-pulse maps,
    - MC tree, and
    - Triggers.
    """

    def __init__(
        self,
        keys: Optional[Union[str, List[str]]] = None,
        exclude_keys: Optional[Union[str, List[str]]] = None,
        extractor_name: str = GENERIC_EXTRACTOR_NAME,
    ):
        """Construct I3GenericExtractor.

        Args:
            keys: List of keys in `I3Frame` to be parsed. Defaults to all keys.
            exclude_keys: List of keys in `I3Frame` to exclude while parsing.

        Raises:
            ValueError: If both `keys` and `exclude_keys` are set.
        """
        # Check(s)
        if (keys is not None) and (exclude_keys is not None):
            raise ValueError(
                "Only one of `keys` and `exclude_keys` should be set."
            )

        # Cast(s)
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]

        # Reference to frame currently being processed
        self._keys: Optional[List[str]] = keys
        self._exclude_keys: Optional[List[str]] = exclude_keys

        # Base class constructor
        super().__init__(extractor_name)

    def _get_keys(self, frame: "icetray.I3Frame") -> List[str]:
        """Get the list of keys to be queried from `frame`.

        If a list of keys was provided by the user, return this. Otherwise,
        return all keys, possibly except ones that the user have explicitly
        excluded.
        """
        if self._keys is None:
            keys = list(frame.keys())
            if self._exclude_keys is not None:
                keys = [key for key in keys if key not in self._exclude_keys]
        else:
            keys = self._keys
        return keys

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract all possible data from `frame`.

        The following types of objects are handled as special cases:
        - Pulse series maps,
        - Per-pulse maps,
        - MC tree, and
        - Triggers.

        All other fields are cast to pure-python classes by generically parsing
        member variables, and checking if the object has a signature similar to
        python lists or dicts.

        Returns:
            Dictionary containing each parsed key in `frame`, and the
                corresponding, extracted data in pure-python format.
        """
        results = {}
        for key in self._get_keys(frame):

            # Extract object from frame
            try:
                obj = frame[key]
            except RuntimeError:
                self.debug(f"Key {key} in frame not supported. Skipping.")
            except KeyError:
                if self._keys is not None:
                    self.warning(f"Key {key} not in frame. Skipping")
                continue

            # Special case(s)
            # -- Pulse series map
            if isinstance(
                obj,
                (
                    dataclasses.I3DOMLaunchSeriesMap,
                    dataclasses.I3RecoPulseSeriesMap,
                    dataclasses.I3RecoPulseSeriesMapMask,
                    dataclasses.I3RecoPulseSeriesMapUnion,
                ),
            ):
                result = self._extract_pulse_series_map(frame, key)

            # -- Per-pulse attribute
            elif isinstance(
                obj,
                (
                    dataclasses.I3MapKeyUInt,
                    dataclasses.I3MapKeyDouble,
                    dataclasses.I3MapKeyVectorInt,
                    dataclasses.I3MapKeyVectorDouble,
                ),
            ):
                result = self._extract_per_pulse_attribute(frame, key)

            # -- MC Tree
            elif isinstance(obj, dataclasses.I3MCTree):
                result = self._cast_mc_tree(obj)

            # -- Triggers
            elif isinstance(obj, dataclasses.I3TriggerHierarchy):
                result = self._cast_triggers(obj)

            # -- Generic case
            else:
                result = cast_object_to_pure_python(obj)

            # Skip empty extractions
            if result is None:
                continue

            # Flatten and transpose MC Tree
            if isinstance(obj, dataclasses.I3MCTree):
                (
                    results[key + "__primaries"],
                    results[key + "__particles"],
                ) = self._flatten_result_mctree(result)

            # Flatten all other objects
            else:
                results[key] = self._flatten_result(result)
                if (
                    isinstance(results[key], dict)
                    and "value" in results[key]
                    and len(results[key]) == 1
                ):
                    results[key] = results[key]["value"]

        # Serialise list of iterables to JSON
        results = {key: serialise(value) for key, value in results.items()}

        return results

    def _extract_pulse_series_map(
        self, frame: "icetray.I3Frame", key: str
    ) -> Optional[Dict[str, Any]]:
        """Extract pulse-series map `key` from `frame`."""
        result = cast_pulse_series_to_pure_python(
            frame, key, self._calibration, self._gcd_dict
        )

        if result is None:
            self.debug(f"Pulse map {key} didn't return anything.")

        return result

    def _extract_per_pulse_attribute(
        self, frame: "icetray.I3Frame", key: str
    ) -> Optional[Dict[str, Any]]:
        """Extract per-pulse attribute `key` from `frame`.

        A per-pulse attribute (e.g., dataclasses.I3MapKeyUInt) is a dictionary-
        like mapping from an OM key to some attribute, e.g., an integer or a
        vector properties.
        """
        result = self._extract_pulse_series_map(frame, key)

        if result is not None:
            # If we get a per-pulse attribute map, which isn't a
            # "I3RecoPulseSeriesMap*", we don't care about area,
            # direction, orientation, and position -- we only care
            # about the OM index for future reference. We therefore
            # only keep these indices and the associated mapping value.
            keep_keys = ["value"] + [
                key_ for key_ in result if key_.startswith("index.")
            ]
            result = {key_: result[key_] for key_ in keep_keys}

        return result

    def _cast_mc_tree(self, obj: "dataclasses.I3MCTree") -> Dict[str, Any]:
        """Cast I3MCTree to dict."""
        result = cast_object_to_pure_python(obj)

        # Assign parent and children links to all particles in tree
        result["particles"] = result.pop("_list")
        for ix, particle in enumerate(obj):
            try:
                parent = obj.parent(particle).minor_id
            except IndexError:
                parent = None

            children = [p.minor_id for p in obj.children(particle)]

            result["particles"][ix]["parent"] = parent
            result["particles"][ix]["children"] = children

        return result

    def _cast_triggers(
        self, obj: "dataclasses.I3TriggerHierarchy"
    ) -> Dict[str, List[Any]]:
        """Cast trigger hierarchy to dict."""
        result = cast_object_to_pure_python(obj)
        assert isinstance(result, list)
        result = transpose_list_of_dicts(result)
        return result

    def _flatten_result_mctree(
        self, result: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Flatten results from casting I3MCTree to pure python."""
        # Flatten and transpose MC Tree
        assert len(result.keys()) == 2
        result_primaries: List[Dict[str, Any]] = result["primaries"]
        result_particles: List[Dict[str, Any]] = result["particles"]

        result_primaries = [
            flatten_nested_dictionary(res) for res in result_primaries
        ]
        result_particles = [
            flatten_nested_dictionary(res) for res in result_particles
        ]

        result_primaries_transposed: Dict[str, List[Any]] = (
            transpose_list_of_dicts(result_primaries)
        )
        result_particles_transposed: Dict[str, List[Any]] = (
            transpose_list_of_dicts(result_particles)
        )

        # Remove `majorID`, which has unsupported unit64 dtype.
        # Keep only one instances of `minorID`.
        del result_primaries_transposed["id__minorID"]
        del result_particles_transposed["id__minorID"]
        del result_primaries_transposed["id__majorID"]
        del result_particles_transposed["id__majorID"]
        del result_primaries_transposed["major_id"]
        del result_particles_transposed["major_id"]

        return result_primaries_transposed, result_particles_transposed

    def _flatten_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten results from casting any other instance to pure python."""
        result = flatten_nested_dictionary(result)

        # If the object is a non-dict object, ensure that it has a non-
        # empty key (required for saving).
        if list(result.keys()) == [""]:
            result["value"] = result.pop("")

        return result
