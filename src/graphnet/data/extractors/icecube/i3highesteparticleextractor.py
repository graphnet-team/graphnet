"""Extract the highest energy particle in the event."""

from typing import Dict, Any, List, TYPE_CHECKING


from .i3extractor import I3Extractor
from .utilities.gcd_hull import GCD_hull

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
        mmctracklist: str = "MMCTrackList",
        extractor_name: str = "HighestEInVolumeParticle",
        daughters: bool = False,
        exclude: list = [None],
    ):
        """Initialize the extractor.

        Args:
            hull: GCD_hull object
            mctree: Name of the MCTree object
            mmctracklist: Name of the MMCTrackList object
            extractor_name: Name of the extractor
            daughters: forces the extractor to only consider daughters
                       of the primary particle
            exclude: List of keys to exclude from the extracted data.
        """
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        # Base class constructor
        super().__init__(extractor_name=extractor_name, exclude=exclude)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract the highest energy particle in the event."""
        output = {}
        if self.frame_contains_info(frame):
            HEParticle = dataclasses.I3Particle()
            HEParticle.energy = 0
            primary_energy = self.check_primary_energy(
                frame, frame[self.mctree].get_primaries()[0]
            ).energy
            distance = -1
            EonEntrance: int = 0
            is_track = -1
            track_length = -1
            # this part handles track particles
            (
                HEParticleT,
                EonEntranceT,
                distanceT,
                track_length,
                checked_id_list,
            ) = self.highest_energy_track(frame)
            # this part handles non-track particles
            HEParticleC, EonEntranceC, distanceC, checked_id_list = (
                self.highest_energy_cascade(frame, checked_id_list)
            )

            if EonEntranceT >= EonEntranceC:
                HEParticle = HEParticleT
                EonEntrance = EonEntranceT
                distance = distanceT
                is_track = 1
            else:
                HEParticle = HEParticleC
                EonEntrance = EonEntranceC
                distance = distanceC
                is_track = 0

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
                    "length_" + self._extractor_name: HEParticle.length,
                    "is_track_" + self._extractor_name: is_track,
                    "interaction_shape_"
                    + self._extractor_name: int(
                        dataclasses.I3Particle.ParticleShape(HEParticle.shape)
                    ),
                    "particle_type_"
                    + self._extractor_name: int(
                        dataclasses.I3Particle.ParticleType(HEParticle.type)
                    ),
                }
            )

            # convert missing values padded with -1 to None
            for key, value in output.items():
                if value == -1:
                    output[key] = None
        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the MCTree."""
        return all(
            [self.mctree in frame.keys(), self.mmctracklist in frame.keys()]
        )

    def highest_energy_track(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> "dataclasses.I3Particle":
        """Get the highest energy track in the event.

        Args:
        frame: I3Frame object
        checked_id_list: list of already checked particle ids
        """
        particle = dataclasses.I3Particle()
        EonEntrance = 0
        distance = -1
        checked_id_list = []
        primary = self.check_primary_energy(
            frame, frame[self.mctree].get_primaries()[0]
        )
        track_length = -1
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
                            track_length = (
                                intersections.first - intersections.second
                            )
        return particle, EonEntrance, distance, track_length, checked_id_list

    def highest_energy_cascade(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> "dataclasses.I3Particle":
        """Get the highest energy cascade in the event.

        Args:
        frame: I3Frame object
        checked_id_list: list of already checked particle ids
        """
        EonEntrance = 0
        HEparticle = dataclasses.I3Particle()
        distance = -1
        if self.daughters:
            particles = dataclasses.I3MCTree.get_daughters(
                frame[self.mctree],
                self.check_primary_energy(
                    frame, frame[self.mctree].get_primaries()[0]
                ),
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
                        distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
                        EonEntrance = particle.energy
        return HEparticle, EonEntrance, distance, checked_id_list
