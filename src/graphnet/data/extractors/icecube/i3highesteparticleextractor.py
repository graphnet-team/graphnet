"""Extract the highest energy particle in the event."""

from typing import Dict, Any, List, TYPE_CHECKING

from .utilities import GCD_hull
from .i3extractor import I3Extractor

import numpy as np

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
        MuonGun,
    )  # pyright: reportMissingImports=false


class I3HighestEparticleExtractor(I3Extractor):
    """Extract the highest energy particle in the event."""

    def __init__(
        self,
        hull: GCD_hull,
        mctree: str = "I3MCTree",
        mmctracklist: str = " MMCTrackList",
        extractor_name: str = "HighestEInVolumeParticle",
        daughters: bool = False,
    ):
        """Initialize the extractor.

        Args:
        hull: GCD_hull object
        mctree: Name of the MCTree object
        mmctracklist: Name of the MMCTrackList object
        extractor_name: Name of the extractor
        daughters: forces the extractor to only consider daughters
                   of the primary particle
        """
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract the highest energy particle in the event."""
        output = {}
        if self.frame_contains_info(frame):
            HEParticle = dataclasses.I3Particle()
            HEParticle.energy = 0
            primary_energy = frame[self.mctree].get_primaries()[0].energy
            distance = -1
            EonEntrance = 0
            # this part handles track particles
            HEParticle, EonEntrance, distance, checked_id_list = (
                self.highest_energy_track(frame)
            )
            # this part handles non-track particles
            HEParticle, EonEntrance, distance, checked_id_list = (
                self.highest_energy_cascade(frame, checked_id_list)
            )
            output.update(
                {
                    "e_fraction_"
                    + self._extractor_name: HEParticle.energy / primary_energy,
                    "distance_" + self._extractor_name: distance,
                    "e_on_entrance_" + self._extractor_name: EonEntrance,
                    "zenith_" + self._extractor_name: HEParticle.dir.zenith,
                    "azimuth_" + self._extractor_name: HEParticle.dir.azimuth,
                    "dir_x_" + self._extractor_name: HEParticle.dir.x,
                    "dir_y_" + self._extractor_name: HEParticle.dir.y,
                    "dir_z_" + self._extractor_name: HEParticle.dir.z,
                    "pos_x_" + self._extractor_name: HEParticle.pos.x,
                    "pos_y_" + self._extractor_name: HEParticle.pos.y,
                    "pos_z_" + self._extractor_name: HEParticle.pos.z,
                    "time_" + self._extractor_name: HEParticle.time,
                    "speed_" + self._extractor_name: HEParticle.speed,
                    "energy_" + self._extractor_name: HEParticle.energy,
                }
            )
        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the MCTree."""
        return all(
            [self.mctree in frame.keys(), self.mmctracklist in frame.keys()]
        )

    def highest_energy_track(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> dataclasses.I3Particle:
        """Get the highest energy track in the event.

        Args:
        frame: I3Frame object
        checked_id_list: list of already checked particle ids
        """
        particle = dataclasses.I3Particle()
        EonEntrance = 0
        checked_id_list = []
        primary = frame[self.mctree].get_primaries()[0]
        for track in frame[self.mmctracklist]:
            track_particle = track.GetI3Particle()
            checked_id_list.append(track_particle.id)
            if self.daughters:
                if (
                    dataclasses.I3MCTree.parent(
                        frame[self.mctree], track_particle
                    )
                    == primary
                ):
                    continue
            if track_particle.energy > EonEntrance:
                intersections = self.hull.surface.intersection(
                    track_particle.pos, track_particle.dir
                )
                if intersections.first is not np.nan:
                    found = False
                    for Mtrack in MuonGun.Track.harvest(
                        frame[self.mctree], frame[self.mmctracklist]
                    ):
                        if Mtrack.id == track_particle.id:
                            Mtrack = Mtrack
                            found = True
                            break
                    if found:
                        if (
                            Mtrack.get_energy(intersections.first)
                            > EonEntrance
                        ):
                            particle = track_particle
                            distance = np.sqrt(
                                sum(
                                    np.array(
                                        [
                                            track.GetXc(),
                                            track.GetYc(),
                                            track.GetZc(),
                                        ]
                                    )
                                    ** 2
                                )
                            )
                            EonEntrance = Mtrack.get_energy(
                                intersections.first
                            )
        return particle, EonEntrance, distance, checked_id_list

    def highest_energy_cascade(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> dataclasses.I3Particle:
        """Get the highest energy cascade in the event.

        Args:
        frame: I3Frame object
        checked_id_list: list of already checked particle ids
        """
        HEparticle = dataclasses.I3Particle()
        EonEntrance = 0
        if self.daughters:
            particles = dataclasses.I3MCTree.get_daughters(
                frame[self.mctree], frame[self.mctree].get_primaries()[0]
            )
        else:
            particles = frame[self.mctree]

        for particle in particles:
            if (
                (particle.id not in checked_id_list)
                & (not particle.is_track)
                & (not particle.is_neutrino)
            ):
                checked_id_list.append(particle.id)
                if particle.length == np.nan:
                    pos = particle.pos
                else:
                    pos = particle.pos + particle.dir * particle.length
                if particle.energy > EonEntrance:
                    if self.hull.point_in_hull(pos):
                        HEparticle = particle
                        distance = np.sqrt(
                            (particle.pos.x**2)
                            + (particle.pos.y**2)
                            + (particle.pos.z**2)
                        )
                        EonEntrance = particle.energy
        return HEparticle, EonEntrance, distance, checked_id_list
