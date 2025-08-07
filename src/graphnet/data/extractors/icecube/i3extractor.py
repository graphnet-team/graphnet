"""Base I3Extractor class(es)."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, List

from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors import Extractor
import numpy as np

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (  # noqa: F401
        icetray,
        dataio,
        dataclasses,
    )  # pyright: reportMissingImports=false


class I3Extractor(Extractor):
    """Base class for extracting information from physics I3-frames.

    Contains functionality required to extract data from i3 files, used by
    the IceCube Neutrino Observatory.

    All classes inheriting from `I3Extractor` should implement the `__call__`
    method.
    """

    def __init__(
        self,
        extractor_name: str,
        exclude: list = [None],
        is_corsika: bool = False,
    ):
        """Construct I3Extractor.

        Args:
            extractor_name: Name of the `I3Extractor` instance. Used to keep
                track of the provenance of different data, and to name tables
                to which this data is saved.
            exclude: List of keys to exclude from the extracted data.
            is_corsika: Boolean indicating if the event files being processed
                are Corsika simulations. Defaults to False.
        """
        # Member variable(s)
        self._i3_file: str = ""
        self._gcd_file: str = ""
        self._gcd_dict: Dict[int, Any] = {}
        self._calibration: Optional["icetray.I3Frame.Calibration"] = None
        self._is_corsika: bool = is_corsika

        # Base class constructor
        super().__init__(extractor_name=extractor_name, exclude=exclude)

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
        if gcd_file is not None:
            self._gcd_file = gcd_file
        if i3_file is not None:
            self._i3_file = i3_file

    @abstractmethod
    def __call__(self, frame: "icetray.I3Frame") -> dict:
        """Extract information from frame."""
        pass

    def check_primary_energy(
        self,
        frame: "icetray.I3Frame",
        primaries: Union[
            "dataclasses.ListI3Particle", "dataclasses.I3Particle"
        ],
    ) -> Union["dataclasses.ListI3Particle", "dataclasses.I3Particle"]:
        """Check that primary energy exists for the particle(s).

        If the primary energy is not available, the daughters of the
        primary particle(s) are returned instead.

        Args:
            frame: I3Frame object.
            primaries: Primary particle or a list of primary particles.
        """
        assert hasattr(
            self, "mctree"
        ), "mctree should be instantiated by subclass"

        if isinstance(primaries, dataclasses.ListI3Particle):
            new_primaries = dataclasses.ListI3Particle()
            for primary in primaries:
                primary = self.check_primary_energy(frame, primary)
                if isinstance(primary, dataclasses.ListI3Particle):
                    new_primaries.extend(primary)
                elif isinstance(primary, dataclasses.I3Particle):
                    new_primaries.append(primary)
                else:
                    raise ValueError(
                        "primaries must be a particle or a list of particles"
                    )
            return new_primaries
        elif isinstance(primaries, dataclasses.I3Particle):
            primary = primaries
        else:
            raise ValueError(
                "primaries must be a particle or a list of particles"
            )

        if primary.energy != primary.energy:
            self.warning_once("Primary energy is nan checking daughters")
            daughters = dataclasses.I3MCTree.get_daughters(
                frame[self.mctree], primary
            )
            if len(daughters) == 0:
                raise ValueError(
                    "Primary energy is nan and no daughters found"
                )
            primary = dataclasses.ListI3Particle()
            for daughter in daughters:
                primary.append(daughter)
        return primary

    def get_primaries(
        self,
        frame: "icetray.I3Frame",
        daughters: bool = False,
        highest_energy_primary: bool = True,
    ) -> "dataclasses.ListI3Particle":
        """Get the primary particles in the event.

        For Corsika events the primary particles are all the primaries,
        for Nugen we are only interested in the in-ice neutrino.
        Input:
        frame: I3Frame object
        daughters: If True, then ensure that for nugen the primaries are
            only the in-ice neutrinos, otherwise all primaries are returned.
        highest_energy_primary: If True, return the primary with the highest
            energy. If False, return all primaries.
            NOTE: only used for non corsika events and only makes a difference
            if daughters is False.
        """
        assert hasattr(
            self, "mctree"
        ), "mctree should be instantiated by subclass"

        if not self._is_corsika:
            primaries = frame[self.mctree].get_primaries()
            if daughters:
                primaries = [
                    p
                    for p in primaries
                    if (
                        p.is_neutrino
                        & (
                            p.location_type
                            == dataclasses.I3Particle.LocationType.InIce.real
                        )
                    )
                ]

                # Some times the primary neutrino is not in-ice,
                # but it has exactly one daughter that is an
                # in-ice neutrino, so we check for that.
                if len(primaries) == 0:

                    # Check if the neutrino primary has daughters
                    primary_nus = [
                        p
                        for p in frame[self.mctree].get_primaries()
                        if p.is_neutrino
                    ]

                    daughters_parts: List["dataclasses.I3Particle"] = []
                    for p in primary_nus:
                        daughters_parts.extend(
                            dataclasses.I3MCTree.get_daughters(
                                frame[self.mctree], p.id
                            )
                        )

                    primaries = [
                        d
                        for d in daughters_parts
                        if (
                            d.is_neutrino
                            & (
                                d.location_type
                                == dataclasses.I3Particle.LocationType.InIce.real  # noqa: E501
                            )
                        )
                    ]

                if len(primaries) == 0:
                    self.warning_once(
                        "No in-ice primary neutrino found, "
                        "no daughters of neutrino primaries either. "
                        "Returning all neutrino primaries. "
                        "NOTE: This is most likely not the intended behaviour."
                    )
                    primaries = primary_nus

                    primaries = [
                        p
                        for p in frame[self.mctree]
                        if (
                            p.is_neutrino
                            & (
                                p.location_type
                                == dataclasses.I3Particle.LocationType.InIce.real  # noqa: E501
                            )
                        )
                    ]

            if len(primaries) == 0:
                self.warning_once("No in-ice neutrino found for NuGen event")
                return dataclasses.ListI3Particle()

            if highest_energy_primary:
                energies = np.array([p.energy for p in primaries])

                primaries = [np.array(primaries)[np.argmax(energies)]]

            primaries = dataclasses.ListI3Particle(primaries)
        if self._is_corsika:
            primaries = frame[self.mctree].get_primaries()
        return primaries
