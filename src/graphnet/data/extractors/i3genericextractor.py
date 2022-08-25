from typing import Any, Dict, List, Optional, Union

from graphnet.data.extractors.i3extractor import I3Extractor
from graphnet.data.extractors.utilities.types import (
    cast_object_to_pure_python,
    cast_pulse_series_to_pure_python,
)
from graphnet.data.extractors.utilities.collections import (
    transpose_list_of_dicts,
    serialise,
    flatten_nested_dictionary,
)

from graphnet.utilities.logging import get_logger

logger = get_logger()

try:
    from icecube import (
        dataclasses,
        icetray,
    )  # pyright: reportMissingImports=false
except ImportError:
    logger.warning("icecube package not available.")


GENERIC_EXTRACTOR_NAME = "<GENERIC>"


class I3GenericExtractor(I3Extractor):
    """Dynamically and generically extract all information from frames."""

    def __init__(
        self,
        keys: Optional[Union[str, List[str]]] = None,
        exclude_keys: Optional[Union[str, List[str]]] = None,
    ):
        """Constructor."""
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
        self._keys: List[str] = keys
        self._exclude_keys: List[str] = exclude_keys

        super().__init__(GENERIC_EXTRACTOR_NAME)

    def _get_keys(self, frame: "icetray.I3Frame") -> List[str]:
        """Get the effective list of keys to be queried from `frame`."""
        if self._keys is None:
            keys = list(frame.keys())
            if self._exclude_keys is not None:
                keys = [key for key in keys if key not in self._exclude_keys]
        else:
            keys = self._keys
        return keys

    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract all possible data from `frame`."""

        results = {}
        for key in self._get_keys(frame):

            # Extract object from frame
            try:
                obj = frame[key]
            except RuntimeError:
                self.logger.debug(
                    f"Key {key} in frame not supported. Skipping."
                )
            except KeyError:
                if self._keys is not None:
                    self.logger.warning(f"Key {key} not in frame. Skipping")
                continue

            # Special case(s)
            # -- Pulse series mao
            if isinstance(
                obj,
                (
                    dataclasses.I3DOMLaunchSeriesMap,
                    dataclasses.I3RecoPulseSeriesMap,
                    dataclasses.I3RecoPulseSeriesMapMask,
                    dataclasses.I3RecoPulseSeriesMapUnion,
                ),
            ):
                result = cast_pulse_series_to_pure_python(
                    frame, key, self._calibration, self._gcd_dict
                )

                if result is None:
                    self.logger.debug(
                        f"Pulse series map {key} didn't return anything."
                    )

            # -- Per-pulse map
            elif isinstance(
                obj,
                (
                    dataclasses.I3MapKeyUInt,
                    dataclasses.I3MapKeyDouble,
                    dataclasses.I3MapKeyVectorInt,
                    dataclasses.I3MapKeyVectorDouble,
                ),
            ):
                result = cast_pulse_series_to_pure_python(
                    frame, key, self._calibration, self._gcd_dict
                )

                if result is None:
                    self.logger.debug(
                        f"Per-pulse map {key} didn't return anything."
                    )

                else:
                    # If we get a per-pulse map, which isn't a
                    # "I3RecoPulseSeriesMap*", we don't care about area,
                    # direction, orientation, and position -- we only care
                    # about the OM index for future reference. We therefore
                    # only keep these indices and the associated mapping value.
                    keep_keys = ["value"] + [
                        key_ for key_ in result if key_.startswith("index.")
                    ]
                    result = {key_: result[key_] for key_ in keep_keys}

            # -- MC Tree
            elif isinstance(obj, dataclasses.I3MCTree):
                result = cast_object_to_pure_python(obj)

                # Assing parent and children links to all particles in tree
                result["particles"] = result.pop("_list")
                for ix, particle in enumerate(obj):
                    try:
                        parent = obj.parent(particle).minor_id
                    except IndexError:
                        parent = None

                    children = [p.minor_id for p in obj.children(particle)]

                    result["particles"][ix]["parent"] = parent
                    result["particles"][ix]["children"] = children

            # -- Triggers
            elif isinstance(obj, dataclasses.I3TriggerHierarchy):
                result = cast_object_to_pure_python(obj)
                assert isinstance(result, list)
                result = transpose_list_of_dicts(result)

            # -- Generic case
            else:

                self.logger.debug(f"Got generic object {key} in frame.")
                result = cast_object_to_pure_python(obj)

            # Skip empty extractions
            if result is None:
                continue

            # Flatten and transpose MC Tree
            if isinstance(obj, dataclasses.I3MCTree):
                assert len(result.keys()) == 2
                result_primaries: List[Dict[str, Any]] = result["primaries"]
                result_particles: List[Dict[str, Any]] = result["particles"]

                result_primaries: List[Dict[str, Any]] = [
                    flatten_nested_dictionary(res) for res in result_primaries
                ]
                result_particles: List[Dict[str, Any]] = [
                    flatten_nested_dictionary(res) for res in result_particles
                ]

                result_primaries: Dict[
                    str, List[Any]
                ] = transpose_list_of_dicts(result_primaries)
                result_particles: Dict[
                    str, List[Any]
                ] = transpose_list_of_dicts(result_particles)

                results[key + ".particles"] = result_particles
                results[key + ".primaries"] = result_primaries

            # Flatten all other objects
            else:
                result = flatten_nested_dictionary(result)

                # If the object is a non-dict object, ensure that it has a non-
                # empty key (required for saving).
                if list(result.keys()) == [""]:
                    result["value"] = result.pop("")

                results[key] = result

        # Serialise list of iterables to JSON
        results = {key: serialise(value) for key, value in results.items()}

        return results
