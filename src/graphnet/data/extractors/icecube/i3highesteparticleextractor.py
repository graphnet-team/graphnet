"""Extract the highest energy particle in the event."""

from typing import Dict, Any, TYPE_CHECKING, Tuple


from .i3extractor import I3Extractor
from .utilities.gcd_hull import GCD_hull
from .utilities.track_containment import track_containment

import numpy as np

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
        MuonGun,
    )  # pyright: reportMissingImports=false
    from icecube.sim_services.label_events.enums import (
        containments_types,
    )  # pyright: reportMissingImports=false

partly_contained_cascade = max([val.value for val in containments_types]) + 1


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
        min_e: float = 5,
    ):
        """Initialize the extractor.

        Args:
            hull: GCD_hull object
            mctree: Name of the MCTree object.
            mmctracklist: Name of the MMCTrackList object.
            extractor_name: Name of the extractor.
            daughters: forces the extractor to only consider daughters
                       of the primary particle.
            exclude: List of keys to exclude from the extracted data.
            min_e: minimum energy for a particle to be considered,
                     default is 5.
        """
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        self.min_e = min_e
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
            distance = -1.0
            EonEntrance = 0.0
            is_track = -1
            track_length = -1
            # this part handles track particles
            (
                HEParticleT,
                EonEntranceT,
                distanceT,
                track_length,
                containmentT,
            ) = self.highest_energy_track(frame, self.min_e)
            (HEParticleC, EonEntranceC, distanceC, containmentC) = (
                self.highest_energy_cascade(
                    frame, min_e=max(EonEntranceT, self.min_e)
                )
            )

            if EonEntranceT >= EonEntranceC:
                HEParticle = HEParticleT
                EonEntrance = EonEntranceT
                distance = distanceT
                is_track = 1
                containment = containmentT
            else:
                HEParticle = HEParticleC
                EonEntrance = EonEntranceC
                distance = distanceC
                is_track = 0
                containment = containmentC

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
                    "track_length_" + self._extractor_name: track_length,
                    "is_track_" + self._extractor_name: is_track,
                    "interaction_shape_"
                    + self._extractor_name: int(
                        dataclasses.I3Particle.ParticleShape(HEParticle.shape)
                    ),
                    "particle_type_"
                    + self._extractor_name: int(
                        dataclasses.I3Particle.ParticleType(HEParticle.type)
                    ),
                    "containment_" + self._extractor_name: int(containment),
                    "parent_type_"
                    + self._extractor_name: int(
                        dataclasses.I3Particle.ParticleType(
                            dataclasses.I3MCTree.parent(
                                frame[self.mctree], HEParticle.id
                            ).type
                        )
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
        self, frame: "icetray.I3Frame", min_e: float = 0
    ) -> "dataclasses.I3Particle":
        """Get the highest energy track in the event.

        Args:
        frame: I3Frame object
        checked_id: dict of already checked particle ids
        min_e: minimum energy for a particle to be considered
        """
        particle = dataclasses.I3Particle()
        EonEntrance = 0
        distance = -1
        containment = -1
        # checked_id = {"major": np.array([]), "minor": np.array([])}
        primary = self.check_primary_energy(
            frame, frame[self.mctree].get_primaries()[0]
        )
        track_length = -1

        MuonGun_tracks = np.array(
            MuonGun.Track.harvest(frame[self.mctree], frame[self.mmctracklist])
        )
        MCTracklist_tracks = np.array(frame[self.mmctracklist])

        energies = np.array([track.energy for track in MuonGun_tracks])

        min_e_mask = energies > min_e
        energies = energies[min_e_mask]
        MuonGun_tracks = MuonGun_tracks[min_e_mask]
        MCTracklist_tracks = MCTracklist_tracks[min_e_mask]
        track_particles = np.array(
            [track.GetI3Particle() for track in MCTracklist_tracks]
        )

        pos, direc, lengths = np.asarray(
            [
                [
                    np.array(p.pos),
                    np.array([p.dir.x, p.dir.y, p.dir.z]),
                    p.length,
                ]
                for p in track_particles
            ],
            dtype=object,
        ).T

        # check if the rays intersect with the sphere approximating the hull
        lengths = lengths.astype(float)
        # replace length nan with 0
        lengths[np.isnan(lengths)] = 0
        pos = np.stack(pos)
        direc = np.stack(direc)

        sphere_mask, t_pos, t_neg = (
            self.hull.rays_and_sphere_intersection_check(pos, direc, lengths)
        )
        # apply sphere mask
        energies = energies[sphere_mask]
        MuonGun_tracks = MuonGun_tracks[sphere_mask]
        MCTracklist_tracks = MCTracklist_tracks[sphere_mask]
        track_particles = track_particles[sphere_mask]
        lengths = lengths[sphere_mask]
        t_pos = t_pos[sphere_mask]
        t_neg = t_neg[sphere_mask]

        assert len(MuonGun_tracks) == len(
            MCTracklist_tracks
        ), "MuonGun and MCTracklist have different lengths"

        while len(energies) > 0:
            loc = np.argmax(energies)
            track = MCTracklist_tracks[loc]
            track_particle = track_particles[loc]
            length = lengths[loc]
            energies = np.delete(energies, loc)
            MCTracklist_tracks = np.delete(MCTracklist_tracks, loc)
            MGtrack = MuonGun_tracks[loc]
            MuonGun_tracks = np.delete(MuonGun_tracks, loc)
            lengths = np.delete(lengths, loc)
            if self.daughters:
                if (
                    dataclasses.I3MCTree.parent(
                        frame[self.mctree], track_particle.id
                    )
                    != primary
                ):
                    continue

            if track_particle.energy > EonEntrance:
                intersections = self.hull.surface.intersection(
                    track_particle.pos, track_particle.dir
                )
                if not np.isnan(intersections.first) & (
                    intersections.first < length
                ):
                    if MGtrack.get_energy(intersections.first) > EonEntrance:
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
                        EonEntrance = MGtrack.get_energy(intersections.first)
                        track_length = (
                            intersections.second - intersections.first
                        )
                        e_mask = energies > EonEntrance
                        energies = energies[e_mask]
                        MCTracklist_tracks = MCTracklist_tracks[e_mask]
                        MuonGun_tracks = MuonGun_tracks[e_mask]
                        containment = track_containment(
                            intersections.first, intersections.second, length
                        )
        return particle, EonEntrance, distance, track_length, containment

    def highest_energy_cascade(
        self, frame: "icetray.I3Frame", min_e: float = 0
    ) -> Tuple["dataclasses.I3Particle", float, float, int]:
        """Get the highest energy cascade in the event.

        Args:
        frame: I3Frame object
        min_e: minimum energy for a particle to be considered
        """
        EonEntrance = 0
        HEparticle = dataclasses.I3Particle()
        distance = -1
        containment = -1
        if self.daughters:
            particles = dataclasses.I3MCTree.get_daughters(
                frame[self.mctree],
                self.check_primary_energy(
                    frame, frame[self.mctree].get_primaries()[0]
                ),
            )
        else:
            particles = frame[self.mctree]

        e_p = np.array(
            [
                np.array([p.energy, p])
                for p in particles
                if ((p.is_cascade) & (not p.is_neutrino)) & (p.energy > min_e)
            ]
        ).T
        if len(e_p) == 0:
            return HEparticle, EonEntrance, distance, containment
        else:
            energies = e_p[0]
            particles = e_p[1]
        # effectively get the pos dir and length of the particle in one pass
        pos, direc, length = np.asarray(
            [[np.array(p.pos), p.dir, p.length] for p in particles],
            dtype=object,
        ).T
        length = length.astype(float)

        # replace length nan with 0
        length[np.isnan(length)] = 0
        pos = pos + direc * length
        pos = np.stack(pos)
        in_volume = self.hull.point_in_hull(pos)
        particles = particles[in_volume]
        energies = energies[in_volume]

        if len(particles) == 0:
            return HEparticle, EonEntrance, distance, containment

        HE_loc = np.argmax(energies)
        HEparticle = particles[HE_loc]
        EonEntrance = HEparticle.energy
        distance = np.sqrt((pos[HE_loc] ** 2).sum())

        cascade_daughters = dataclasses.I3MCTree.get_daughters(
            frame[self.mctree], HEparticle
        )
        pos, direc, length = np.asarray(
            [[np.array(p.pos), p.dir, p.length] for p in cascade_daughters],
            dtype=object,
        ).T
        pos = pos + direc * length
        pos = np.stack(pos)
        in_volume = self.hull.point_in_hull(pos)
        if not np.all(in_volume):
            containment = partly_contained_cascade
        else:
            containment = containments_types.contained.value

        return HEparticle, EonEntrance, distance, containment
