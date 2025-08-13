"""Extract all the visible particles entering the volume."""

from typing import Dict, Any, TYPE_CHECKING, Tuple, List

from .utilities.gcd_hull import GCD_hull
from .i3extractor import I3Extractor

import numpy as np

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
        MuonGun,
    )  # pyright: reportMissingImports=false


class I3Calorimetry(I3Extractor):
    """Event level energy labeling for IceCube data.

    This class extracts cumulative energy information from all visible
    particles entering the detector volume, during the event.
    """

    def __init__(
        self,
        hull: GCD_hull,
        mctree: str = "I3MCTree",
        mmctracklist: str = "MMCTrackList",
        extractor_name: str = "I3Calorimetry",
        daughters: bool = False,
        highest_energy_primary: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a ConvexHull object from the GCD file.

        hull: GCD_hull object containing the convex hull
        of the detector volume.
        mctree: Name of the I3MCTree in the frame.
        mmctracklist: Name of the MMCTrackList in the frame.
        extractor_name: Name of the extractor.
        daughters: If True, only calculate energies for particles
            that are daughters of the primary.
        highest_energy_primary: If True, takes into account only the
            primary with the highest energy.
            NOTE: Only makes a difference if daughters is False
                and the event is not a Corsika event.

        Variable explanation:
        - e_entrance_track: Total energy of tracks entering the hull.
        - e_deposited_track: Total energy deposited by tracks in the hull.
        - e_cascade: Total energy of cascade particles in the hull.
        - e_visible: Total energy of particles entering the hull.
            NOTE: if daughters is True, this is the total visible energy
            of daughter particles of the primary particles. If this is 0
            that means that all the light in the detector comes from
            particles that are daughters of coincident primaries.
        - fraction_primary: Fraction of `e_visible` compared to
            the primary energy.
        - fraction_cascade: Fraction of the total energy that is
            deposited by cascade particles compared to the total energy.
        """
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        self.highest_energy_primary = highest_energy_primary
        # Base class constructor
        super().__init__(extractor_name=extractor_name, **kwargs)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract all the visible particles entering the volume."""
        output = {}
        if self.frame_contains_info(frame):

            primaries = self.get_primaries(
                frame,
                self.daughters,
                self.highest_energy_primary,
            )

            if not len(primaries) == 0:

                MMCTrackList = frame[self.mmctracklist]
                # Filter tracks that are not daughters of the desired
                if self.daughters:
                    temp_MMCTrackList = []
                    for track in MMCTrackList:
                        for p in primaries:
                            if frame[self.mctree].is_in_subtree(
                                p.id, track.GetI3Particle().id
                            ):
                                temp_MMCTrackList.append(track)
                                break
                    MMCTrackList = temp_MMCTrackList

                # Create a lookup dict for the tracks
                track_lookup = {}
                for track in MuonGun.Track.harvest(
                    frame[self.mctree], MMCTrackList
                ):
                    track_lookup[track.id] = track

                e_dep_cascade, e_dep_track, e_ent_track = self.get_energies(
                    frame, primaries, track_lookup
                )

                primary_energy = sum([p.energy for p in primaries])
            else:
                e_ent_track = np.nan
                e_dep_track = np.nan
                e_dep_cascade = np.nan
                primary_energy = np.nan

            e_total = e_ent_track + e_dep_cascade

            # In case all particles are considered and
            # there is no energy deposited in the hull,
            # we warn the user.
            if all(
                (
                    not self.daughters,
                    not self.highest_energy_primary,
                    e_total == 0,
                )
            ):
                self.warning(
                    "No energy deposited in the hull,"
                    "Think about in creasing the padding."
                    f"\nCurrent padding: {self.hull.padding}"
                    f"\nTotal energy: {e_total}"
                    f"\nTrack energy: {e_ent_track}"
                    f"\nCascade energy: {e_dep_cascade}"
                    f"\nEvent header: {frame['I3EventHeader']}"
                )

            # Check only in the case that there were primaries
            if not len(primaries) == 0 and (not np.isnan(e_total)):

                # total energy should always be less than the primary energy
                assert e_total <= (
                    primary_energy * (1 + 1e-6)
                ), "Total energy on entrance is greater than primary energy\
                    \nTotal energy: {}\
                    \nPrimary energy: {}\
                    \nTrack energy: {}\
                    \nCascade energy: {}\
                    {}".format(
                    e_total,
                    primary_energy,
                    e_ent_track,
                    e_dep_cascade,
                    frame["I3EventHeader"],
                )

                assert (
                    primary_energy > 0
                ), "Primary energy is 0, this should not happen.\
                    \nTotal energy: {}\
                    \nTrack energy: {}\
                    \nCascade energy: {}\
                    {}".format(
                    e_total,
                    e_ent_track,
                    e_dep_cascade,
                    frame["I3EventHeader"],
                )
            fraction_primary = e_total / primary_energy

            cascade_fraction = None
            if e_total > 0:
                cascade_fraction = e_dep_cascade / e_total

            output.update(
                {
                    "e_entrance_track_" + self._extractor_name: e_ent_track,
                    "e_deposited_track_" + self._extractor_name: e_dep_track,
                    "e_cascade_" + self._extractor_name: e_dep_cascade,
                    "e_visible_" + self._extractor_name: e_total,
                    "fraction_primary_"
                    + self._extractor_name: fraction_primary,
                    "fraction_cascade_"
                    + self._extractor_name: cascade_fraction,
                }
            )

        output = {k: v for k, v in output.items() if k not in self._exclude}
        return output

    def get_energies(
        self,
        frame: "icetray.I3Frame",
        particles: List["dataclasses.I3Particle"],
        track_lookup: Dict["icetray.I3ParticleID", "icetray.I3Particle"],
    ) -> Tuple[float, float, float]:
        """Get the total energy of cascade particles on entrance."""
        e_dep_cascade = 0
        e_dep_track = 0
        e_ent_track = 0

        if len(particles) == 0:
            return e_dep_cascade, e_dep_track, e_ent_track

        for particle in particles:
            length = particle.length
            if length != length:
                length = 0
            # If the particle is a track in the MMCTrackList take the
            # energy at the entrance and exit of the hull.
            # NOTE: We do not consider daughters of tracks,
            # because they are already included in the track energy.
            if particle.is_track & (particle.id in track_lookup):
                track = track_lookup[particle.id]

                # Find distance to entrance and exit from sampling volume
                intersections = self.hull.surface.intersection(
                    track.pos, track.dir
                )
                # Get the corresponding energies
                try:
                    e0 = track.get_energy(intersections.first)
                    e1 = track.get_energy(intersections.second)

                # Catch MuonGun errors
                except RuntimeError as e:
                    if (
                        "sum of losses is smaller than "
                        "energy at last checkpoint" in str(e)
                    ):
                        hdr = frame["I3EventHeader"]
                        e.add_note(f"Error in MuonGun track in event {hdr}")
                        self.warning(f"Skipping bad event {hdr}: {e}")
                        e0 = np.nan
                        e1 = np.nan
                        e_dep_cascade = np.nan
                        continue  # skip this frame
                    else:
                        raise  # re-raise unexpected errors

                e_dep_track += e0 - e1
                e_ent_track += e0
            # if the particle is not in the hull, but has daughters,
            # we add the energies of the daughters.
            elif not self.is_in_hull(particle):
                daughters = dataclasses.I3MCTree.get_daughters(
                    frame[self.mctree], particle
                )
                if len(daughters) == 0:
                    continue
                (
                    e_dep_cascade,
                    e_dep_track,
                    e_ent_track,
                ) = tuple(
                    np.add(
                        (e_dep_cascade, e_dep_track, e_ent_track),
                        self.get_energies(
                            frame,
                            daughters,
                            track_lookup,
                        ),
                    )
                )
            # If the particle is a cascade in the hull, we add its energy.
            elif particle.is_cascade:
                # Check wether the cascade is made up of smaller segments
                # in this case the shape is dark and we want to count
                # the energy of its daughters.
                if particle.shape != dataclasses.I3Particle.ParticleShape.Dark:
                    e_dep_cascade += particle.energy
                else:
                    (
                        e_dep_cascade,
                        e_dep_track,
                        e_ent_track,
                    ) = tuple(
                        np.add(
                            (e_dep_cascade, e_dep_track, e_ent_track),
                            self.get_energies(
                                frame,
                                dataclasses.I3MCTree.get_daughters(
                                    frame[self.mctree], particle
                                ),
                                track_lookup,
                            ),
                        )
                    )
            # The particle is in the hull and not a track in the MMCTrackList,
            # or a cascade, so we look at its daughters.
            # Could be a NuMu interacting within the hull.
            else:
                (
                    e_dep_cascade,
                    e_dep_track,
                    e_ent_track,
                ) = tuple(
                    np.add(
                        (e_dep_cascade, e_dep_track, e_ent_track),
                        self.get_energies(
                            frame,
                            dataclasses.I3MCTree.get_daughters(
                                frame[self.mctree], particle
                            ),
                            track_lookup,
                        ),
                    )
                )

        return e_dep_cascade, e_dep_track, e_ent_track

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the necessary information."""
        return self.mctree in frame and self.mmctracklist in frame

    def is_in_hull(self, particle: "dataclasses.I3Particle") -> bool:
        """Check if a particle is in the hull."""
        pos = np.array(particle.pos)
        direc = np.array([particle.dir.x, particle.dir.y, particle.dir.z])
        length = particle.length if particle.length is not None else 0
        pos = pos + direc * length

        return self.hull.point_in_hull(pos)
