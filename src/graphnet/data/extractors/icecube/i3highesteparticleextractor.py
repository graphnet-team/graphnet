"""Extract the highest energy particle in the event."""

from typing import Dict, Any, TYPE_CHECKING, Tuple, Union, List


from .i3extractor import I3Extractor
from .utilities.gcd_hull import GCD_hull
from .utilities.containments import track_containment

import numpy as np

from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube.utilities.containments import (
    GN_containment_types,
)

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
        MuonGun,
        simclasses,
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
        min_e: float = 5,
        **kwargs: Any,
    ):
        """Initialize the extractor.

        Args:
            hull: GCD_hull object
            mctree: Name of the MCTree object.
            mmctracklist: Name of the MMCTrackList object.
            extractor_name: Name of the extractor.
            daughters: forces the extractor to only consider daughters
                       of the primary particle.
            min_e: minimum energy for a particle to be considered,
                     default is 5.
            **kwargs: Additional keyword arguments for I3Extractors.
        """
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        self.min_e = min_e
        # Base class constructor
        super().__init__(extractor_name=extractor_name, **kwargs)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract the highest energy particle in the event."""
        output = {}
        if self.frame_contains_info(frame):
            HEParticle = dataclasses.I3Particle()
            HEParticle.energy = 0
            primary_energy = sum(
                prim.energy
                for prim in self.get_primaries(frame, self.daughters)
            )
            distance = -1.0
            EonEntrance = 0.0
            trackness = -1.0
            visible_length = -1.0
            # this part handles track particles
            if self._is_corsika:
                (
                    HEParticle,
                    EonEntrance,
                    distance,
                    visible_length,
                    containment,
                ) = self.highest_energy_bundle(frame, self.min_e)
                trackness = 1.0

            else:

                (
                    HEParticleT,
                    EonEntranceT,
                    distanceT,
                    visible_lengthT,
                    containmentT,
                ) = self.highest_energy_track(frame, self.min_e)
                (
                    HEParticleC,
                    EonEntranceC,
                    distanceC,
                    containmentC,
                    visible_lengthC,
                    tracknessC,
                ) = self.highest_energy_starting(
                    frame, min_e=max(EonEntranceT, self.min_e)
                )

                if EonEntranceT >= EonEntranceC:
                    HEParticle = HEParticleT
                    EonEntrance = EonEntranceT
                    distance = distanceT
                    trackness = 1.0
                    containment = containmentT
                    visible_length = visible_lengthT
                else:
                    HEParticle = HEParticleC
                    EonEntrance = EonEntranceC
                    distance = distanceC
                    trackness = tracknessC
                    containment = containmentC
                    visible_length = visible_lengthC

            try:
                parent_type = dataclasses.I3Particle.ParticleType(
                    dataclasses.I3MCTree.parent(
                        frame[self.mctree], HEParticle.id
                    ).type
                )
            except IndexError:
                parent_type = 0

            if primary_energy > 0:
                primary_fraction = EonEntrance / primary_energy
            else:
                primary_fraction = -1
            output.update(
                {
                    "e_fraction_" + self._extractor_name: primary_fraction,
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
                    "length_" + self._extractor_name: HEParticle.length,
                    "visible_length_" + self._extractor_name: visible_length,
                    "trackness_" + self._extractor_name: trackness,
                    "interaction_shape_"
                    + self._extractor_name: HEParticle.shape,
                    "particle_type_" + self._extractor_name: HEParticle.type,
                    "containment_" + self._extractor_name: containment,
                    "parent_type_" + self._extractor_name: parent_type,
                }
            )

            # convert missing values padded with -1 to None
            for key, value in output.items():
                if "type" in key:
                    continue
                if value == -1:
                    output[key] = None
        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the MCTree."""
        return all(
            [self.mctree in frame.keys(), self.mmctracklist in frame.keys()]
        )

    def get_tracks(
        self, frame: "icetray.I3Frame"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the tracks from the frame.

        Args:
        frame: I3Frame object
        """
        primaries = self.get_primaries(frame, self.daughters)
        primaries = [self.check_primary_energy(frame, p) for p in primaries]

        MMCTrackList = frame[self.mmctracklist]
        if self.daughters:
            temp_MMCTrackList = []
            for track in MMCTrackList:
                for p in primaries:
                    if frame[self.mctree].is_in_subtree(
                        p.id, track.GetI3Particle().id
                    ):
                        temp_MMCTrackList.append(track)
                        break
            MMCTrackList = simclasses.I3MMCTrackList(temp_MMCTrackList)

        MuonGun_tracks = np.array(
            MuonGun.Track.harvest(frame[self.mctree], MMCTrackList)
        )
        MMCTrackList = np.array(MMCTrackList)

        return (
            MuonGun_tracks,
            MMCTrackList,
        )

    def get_pos_dir_length(
        self, particles: "dataclasses.ListI3Particle"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the position, direction and length of the particles.

        Args:
        particles: List of I3Particles
        """
        pos, direc, lengths = np.asarray(
            [
                [np.array(p.pos), np.array(p.dir * 1), p.length]
                for p in particles
            ],
            dtype=object,
        ).T
        lengths = lengths.astype(float)

        # replace length nan with 0
        lengths[np.isnan(lengths)] = 0
        return pos, direc, lengths

    def get_bundle_HEP(self, particles: np.array) -> "dataclasses.I3Particle":
        """Get the energy averaged particle of a list of particles.

        Args:
        particles: List of I3Particles
        """
        if len(particles) == 0:
            return dataclasses.I3Particle(), np.array([]), True

        energies, lengths = np.array(
            [[p.energy, p.length] for p in particles]
        ).T
        loc_max = np.argmax(energies)
        # Inherit from the highest energy particle
        bundle = particles[loc_max]

        intersections = self.hull.surface.intersection(bundle.pos, bundle.dir)

        if np.isnan(intersections.first):
            # check if the particle does not intersect the hull,
            # if so return an empty particle
            # This check might be redundant.
            return dataclasses.I3Particle(), np.array([]), True

        length_mask = lengths > intersections.first

        return bundle, length_mask, False

    def highest_energy_track(
        self, frame: "icetray.I3Frame", min_e: float = 0
    ) -> Tuple["dataclasses.I3Particle", float, float, float, int]:
        """Get the highest energy track in the event.

        Args:
        frame: I3Frame object
        checked_id: dict of already checked particle ids
        min_e: minimum energy for a particle to be considered
        """
        particle = dataclasses.I3Particle()
        EonEntrance = 0.0
        distance = -1.0
        visible_length = -1.0
        containment = -1

        MuonGun_tracks, MMCTrackList = self.get_tracks(frame)

        energies = np.array([track.energy for track in MuonGun_tracks])

        min_e_mask = energies > min_e
        energies = energies[min_e_mask]
        if len(energies) == 0:
            return particle, EonEntrance, distance, visible_length, containment
        MuonGun_tracks = MuonGun_tracks[min_e_mask]
        MMCTrackList = MMCTrackList[min_e_mask]
        track_particles = np.array(
            [track.GetI3Particle() for track in MMCTrackList]
        )

        pos, direc, lengths = self.get_pos_dir_length(track_particles)

        pos = np.stack(pos)
        direc = np.stack(direc)

        sphere_mask, t_pos, t_neg = (
            self.hull.rays_and_sphere_intersection_check(pos, direc, lengths)
        )
        # apply sphere mask
        energies = energies[sphere_mask]
        MuonGun_tracks = MuonGun_tracks[sphere_mask]
        MMCTrackList = MMCTrackList[sphere_mask]
        track_particles = track_particles[sphere_mask]
        lengths = lengths[sphere_mask]
        t_pos = t_pos[sphere_mask]
        t_neg = t_neg[sphere_mask]

        assert len(MuonGun_tracks) == len(
            MMCTrackList
        ), "MuonGun and MCTracklist have different lengths"

        while len(energies) > 0:
            loc = np.argmax(energies)
            track = MMCTrackList[loc]
            track_particle = track_particles[loc]
            length = lengths[loc]
            energies = np.delete(energies, loc)
            MMCTrackList = np.delete(MMCTrackList, loc)
            MGtrack = MuonGun_tracks[loc]
            MuonGun_tracks = np.delete(MuonGun_tracks, loc)
            lengths = np.delete(lengths, loc)

            if track_particle.energy > EonEntrance:
                intersections = self.hull.surface.intersection(
                    track_particle.pos, track_particle.dir
                )
                if not np.isnan(intersections.first) & (
                    intersections.first < length
                ):
                    if MGtrack.get_energy(intersections.first) > EonEntrance:
                        particle = track_particle

                        closest_pos = np.array(
                            [
                                track.GetXc(),
                                track.GetYc(),
                                track.GetZc(),
                            ]
                        )

                        EonEntrance = MGtrack.get_energy(
                            max(intersections.first, 0)
                        )

                        # a skimming track can be outside the hull
                        # therefore it can have 0 visible length
                        visible_length = max(
                            0,
                            intersections.second - max(intersections.first, 0),
                        )
                        e_mask = energies > EonEntrance
                        energies = energies[e_mask]
                        MMCTrackList = MMCTrackList[e_mask]
                        MuonGun_tracks = MuonGun_tracks[e_mask]
                        containment = track_containment(
                            intersections.first, intersections.second, length
                        )
                        if containment in [
                            GN_containment_types.contained.value,
                            GN_containment_types.starting.value,
                        ]:
                            # If the track is contained or starting
                            # pos is the starting position.
                            starting_pos = np.array(
                                [track.GetXi(), track.GetYi(), track.GetZi()]
                            )
                            distance = np.sqrt((starting_pos**2).sum())
                            particle.pos = dataclasses.I3Position(
                                starting_pos[0],
                                starting_pos[1],
                                starting_pos[2],
                            )
                            particle.time = track.GetTi()
                        else:
                            # If the track is stopping or throughgoing,
                            # pos is point closest to detector center.
                            distance = np.sqrt((closest_pos**2).sum())
                            particle.pos = dataclasses.I3Position(
                                closest_pos[0], closest_pos[1], closest_pos[2]
                            )
                            particle.time = track.GetTc()

        return particle, EonEntrance, distance, visible_length, containment

    def highest_energy_starting(
        self,
        frame: "icetray.I3Frame",
        min_e: float = 0,
    ) -> Tuple["dataclasses.I3Particle", float, float, int, float, float]:
        """Get the highest energy starting particle in the event.

        Args:
        frame: I3Frame object
        min_e: minimum energy for a particle to be considered
        """
        EonEntrance = 0.0
        dummy_particle = dataclasses.I3Particle()
        dummy_particle.energy = 0.0
        distance = -1.0
        containment = GN_containment_types.no_intersect.value
        visible_length = 0.0
        if self.daughters:
            primaries = self.get_primaries(frame, self.daughters)
            primaries = [
                self.check_primary_energy(frame, p) for p in primaries
            ]

            particles = self.get_descendants(frame, primaries)

            e_p = []
            for part in particles:
                if (part.energy > min_e) & (~part.is_track):
                    e_p.append(np.array([part.energy, part]))

            e_p = np.array(e_p).T

        else:

            particles = frame[self.mctree]

            e_p = np.array(
                [
                    np.array([p.energy, p])
                    for p in particles
                    if (p.energy > min_e) & (not p.is_track)
                ]
            ).T

        if len(e_p) == 0:
            return (
                dummy_particle,
                EonEntrance,
                distance,
                containment,
                visible_length,
                -1,
            )

        energies = e_p[0]
        particles = e_p[1]

        pos, direc, lengths = self.get_pos_dir_length(particles)
        pos = pos + direc * lengths
        pos = np.stack(pos)
        in_volume = self.hull.point_in_hull(pos)
        particles = particles[in_volume]
        energies = energies[in_volume]
        pos = pos[in_volume]

        if len(particles) == 0:
            return (
                dummy_particle,
                EonEntrance,
                distance,
                containment,
                visible_length,
                -1,
            )

        # Move the particle position to the interaction vertex.
        HE_loc = np.argmax(energies)
        entry_particle = particles[HE_loc]
        entry_particle.pos = dataclasses.I3Position(
            pos[HE_loc][0], pos[HE_loc][1], pos[HE_loc][2]
        )
        # For starting tracks the time we are interested in is the time
        # at the interaction point i.e. the end of the generating particle.
        entry_particle.time = entry_particle.time + (
            entry_particle.length / entry_particle.speed
        )
        # distance to the interaction vertex
        distance = np.sqrt((pos[HE_loc] ** 2).sum())
        # Get all the visible particles produced by the entry particle
        visible_particles = self.get_visible_produced_particles(
            frame, entry_particle
        )
        # split the visible particles into tracks and cascades
        tracks = np.array([p for p in visible_particles if p.is_track])
        cascades = np.array([p for p in visible_particles if p.is_cascade])
        # if the tracks start inside the detector we consider the energy to
        # be reconstructable and therefore added to the energy on entrance
        E_tracks = 0
        if len(tracks) > 0:
            tracks_in_volume = self.hull.point_in_hull(
                np.array([p.pos for p in tracks])
            )
            tracks = tracks[tracks_in_volume]
            E_tracks = np.sum([p.energy for p in tracks])
            EonEntrance += E_tracks
            # get the visible length of the track
            t_containments = []
            real_track = False
            for track in tracks:
                intersections = self.hull.surface.intersection(
                    track.pos, track.dir
                )
                visible_length = max(
                    visible_length,
                    intersections.second - max(intersections.first, 0),
                )
                # Check if we have a single topologically "real" track
                if not real_track:
                    if not dataclasses.I3MCTree.parent(
                        frame[self.mctree], track.id
                    ).is_cascade:
                        real_track = True
                # decide the containment of the track
                temp_containment = track_containment(
                    intersections.first, intersections.second, track.length
                )
                assert temp_containment in [
                    GN_containment_types.contained.value,
                    GN_containment_types.starting.value,
                ], "Invalid containment type"
                t_containments.append(temp_containment)
        # for the cascades we need to check that they are still in the detector
        # at the generation point we consider the energy to be reconstructable
        # if the cascade starts inside the detector
        E_cascades = 0
        if len(cascades) > 0:
            cascades_in_volume = self.hull.point_in_hull(
                np.array([p.pos for p in cascades])
            )
            cascades = cascades[cascades_in_volume]
            E_cascades = np.sum([p.energy for p in cascades])
            EonEntrance += E_cascades
            # get the visible length of the cascade
            c_containments = []
            for cascade in cascades:
                cascade_terminal_pos = np.array(
                    cascade.pos + cascade.dir * cascade.length
                )
                visible_length = max(
                    visible_length,
                    np.sqrt(
                        np.sum(
                            (entry_particle.pos - cascade_terminal_pos) ** 2
                        )
                    ),
                )
                terminal_in_hull = self.hull.point_in_hull(
                    cascade_terminal_pos
                )
                if terminal_in_hull:
                    c_containments.append(GN_containment_types.contained.value)
                else:
                    c_containments.append(
                        GN_containment_types.partly_contained.value
                    )

        if EonEntrance == 0:
            containment = GN_containment_types.no_intersect.value
            return (
                entry_particle,
                EonEntrance,
                distance,
                containment,
                visible_length,
                -1,
            )

        if len(tracks) > 0:
            if len(cascades) > 0:
                if (
                    all(
                        [
                            t == GN_containment_types.stopping.value
                            for t in t_containments
                        ]
                    )
                    & all(cascades_in_volume)
                    & all(
                        [
                            c == GN_containment_types.contained.value
                            for c in c_containments
                        ]
                    )
                ):
                    containment = GN_containment_types.contained.value
                else:
                    if real_track:
                        containment = GN_containment_types.starting.value
                    else:
                        containment = (
                            GN_containment_types.partly_contained.value
                        )
            else:
                if all(
                    [
                        t == GN_containment_types.stopping.value
                        for t in t_containments
                    ]
                ):
                    containment = GN_containment_types.contained.value
                else:
                    containment = GN_containment_types.starting.value
        else:
            if len(cascades) > 0:
                if all(cascades_in_volume) & all(
                    [
                        c == GN_containment_types.contained.value
                        for c in c_containments
                    ]
                ):
                    containment = GN_containment_types.contained.value
                else:
                    containment = GN_containment_types.partly_contained.value
        return (
            entry_particle,
            EonEntrance,
            distance,
            containment,
            visible_length,
            E_tracks / (E_tracks + E_cascades),
        )

    def highest_energy_bundle(
        self, frame: "icetray.I3Frame", min_e: float = 0
    ) -> Tuple["dataclasses.I3Particle", float, float, float, int]:
        """Get the highest energy bundle in the event.

        Args:
        frame: I3Frame object
        min_e: minimum energy for a particle to be considered
        """
        particle = dataclasses.I3Particle()
        EonEntrance = 0.0
        distance = -1.0
        containment = None
        visible_length = -1
        closest_time = None

        MuonGun_tracks, MMCTrackList = self.get_tracks(frame)
        energies = np.array([track.energy for track in MuonGun_tracks])

        min_e_mask = energies > min_e
        energies = energies[min_e_mask]
        if len(energies) == 0:
            return (
                particle,
                EonEntrance,
                distance,
                visible_length,
                GN_containment_types.no_intersect.value,
            )

        MuonGun_tracks = MuonGun_tracks[min_e_mask]
        MMCTrackList = MMCTrackList[min_e_mask]
        track_particles = np.array(
            [track.GetI3Particle() for track in MMCTrackList]
        )

        pos, direc, lengths = self.get_pos_dir_length(track_particles)

        pos = np.stack(pos)
        direc = np.stack(direc)

        sphere_mask, t_pos, t_neg = (
            self.hull.rays_and_sphere_intersection_check(pos, direc, lengths)
        )

        energies = energies[sphere_mask]
        MuonGun_tracks = MuonGun_tracks[sphere_mask]
        MMCTrackList = MMCTrackList[sphere_mask]
        track_particles = track_particles[sphere_mask]
        lengths = lengths[sphere_mask]

        assert len(MuonGun_tracks) == len(
            MMCTrackList
        ), "MuonGun and MCTracklist have different lengths"

        no_intersect = True
        bundle, length_mask, no_intersect = self.get_bundle_HEP(
            track_particles
        )

        if no_intersect:
            # If the particle does not intersect the hull,
            # return an empty particle
            return (
                particle,
                EonEntrance,
                distance,
                visible_length,
                GN_containment_types.no_intersect.value,
            )

        energies = energies[length_mask]
        MuonGun_tracks = MuonGun_tracks[length_mask]
        MMCTrackList = MMCTrackList[length_mask]
        track_particles = track_particles[length_mask]
        lengths = lengths[length_mask]

        containment = GN_containment_types.stopping_bundle.value
        closest_pos = []
        for track, MGtrack in zip(MMCTrackList, MuonGun_tracks):
            intersections = self.hull.surface.intersection(
                MGtrack.pos, MGtrack.dir
            )

            track_energy = MGtrack.get_energy(intersections.first)
            EonEntrance += track_energy

            closest_pos.append(
                np.array(
                    [
                        track.GetXc(),
                        track.GetYc(),
                        track.GetZc(),
                    ]
                )
                * track_energy
            )
            if closest_time is None:
                closest_time = track.GetTc()
            elif closest_time < track.GetTc():
                closest_time = track.GetTc()

            if intersections.second > 0:
                visible_length = max(
                    visible_length, intersections.second - intersections.first
                )
                if MGtrack.length > intersections.second:
                    containment = (
                        GN_containment_types.throughgoing_bundle.value
                    )

        closest_pos = np.sum(closest_pos, axis=0) / EonEntrance

        bundle.pos = dataclasses.I3Position(
            closest_pos[0], closest_pos[1], closest_pos[2]
        )
        bundle.time = closest_time
        distance = np.sqrt((np.array(closest_pos) ** 2).sum())

        return bundle, EonEntrance, distance, visible_length, containment

    def get_visible_produced_particles(
        self, frame: "icetray.I3Frame", particle: "dataclasses.I3Particle"
    ) -> "dataclasses.ListI3Particle":
        """Get the visible particles produced by a particle.

        Produces a list of particles that are produced by the input particle
        down to the final node particles

        Args:
        frame: I3Frame object
        particle: I3Particle object
        """
        daughters = dataclasses.I3MCTree.get_daughters(
            frame[self.mctree], particle
        )
        visible_particles = dataclasses.ListI3Particle()
        while len(daughters) > 0:
            daughter = daughters[0]
            daughters = daughters[1:]
            if daughter.is_neutrino:
                daughters.extend(
                    dataclasses.I3MCTree.get_daughters(
                        frame[self.mctree], daughter
                    )
                )
            if daughter.is_cascade & (
                daughter.shape == dataclasses.I3Particle.ParticleShape.Dark
            ):
                daughters.extend(
                    dataclasses.I3MCTree.get_daughters(
                        frame[self.mctree], daughter
                    )
                )
            else:
                visible_particles.append(daughter)
        return visible_particles

    def get_descendants(
        self,
        frame: "icetray.I3Frame",
        particle: Union[
            "dataclasses.I3Particle", List["dataclasses.I3Particle"]
        ],
    ) -> "dataclasses.ListI3Particle":
        """Get the descendants of a particle and the particle as a list.

        Args:
            frame: I3Frame object
            particle: I3Particle object
        """
        if isinstance(particle, list):
            ret = []
            for p in particle:
                ret.extend(self.get_descendants(frame, p))
            return ret
        else:
            daughters = frame[self.mctree].get_daughters(particle)
            if len(daughters) == 0:
                return [particle]
            else:
                ret = []
                ret.append(particle)
                for p in daughters:
                    ret.extend(self.get_descendants(frame, p))
                return ret
