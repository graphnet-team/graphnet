"""Extract all the visible particles entering the volume."""

from typing import Dict, Any, TYPE_CHECKING, Tuple

from .utilities.gcd_hull import GCD_hull
from .i3extractor import I3Extractor

import numpy as np

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
        MuonGun,
        simclasses,
    )  # pyright: reportMissingImports=false


class I3Calorimetry(I3Extractor):
    """Energy on appearance of all visible particles in the volume."""

    def __init__(
        self,
        hull: GCD_hull,
        mctree: str = "I3MCTree",
        mmctracklist: str = "MMCTrackList",
        extractor_name: str = "event_energies",
        daughters: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a ConvexHull object from the GCD file."""
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        # Base class constructor
        super().__init__(extractor_name=extractor_name, **kwargs)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract all the visible particles entering the volume."""
        output = {}
        if self.frame_contains_info(frame):

            e_entrance_track, e_deposited_track = self.total_track_energy(
                frame
            )

            e_deposited_cascade = self.total_cascade_energy(frame)

            primary_energy = sum(
                [
                    p.energy
                    for p in self.check_primary_energy(
                        frame, self.get_primaries(frame, self.daughters)
                    )
                ]
            )
            e_total = e_entrance_track + e_deposited_cascade

            if self.daughters:
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
                    e_entrance_track,
                    e_deposited_cascade,
                    frame["I3EventHeader"],
                )

            cascade_fraction = None
            if e_total > 0:
                cascade_fraction = e_deposited_cascade / e_total

            if primary_energy > 0:
                fraction_primary = e_total / primary_energy
            else:
                fraction_primary = None
            output.update(
                {
                    "e_entrance_track_"
                    + self._extractor_name: e_entrance_track,
                    "e_deposited_track_"
                    + self._extractor_name: e_deposited_track,
                    "e_cascade_" + self._extractor_name: e_deposited_cascade,
                    "e_visible_" + self._extractor_name: e_total,
                    "fraction_primary_"
                    + self._extractor_name: fraction_primary,
                    "fraction_cascade_"
                    + self._extractor_name: cascade_fraction,
                }
            )

        output = {k: v for k, v in output.items() if k not in self._exclude}
        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the necessary information."""
        return self.mctree in frame and self.mmctracklist in frame

    def total_track_energy(
        self, frame: "icetray.I3Frame"
    ) -> Tuple[float, float]:
        """Get the total energy of track particles on entrance."""
        e_entrance = 0
        e_deposited = 0
        primaries = self.get_primaries(frame, self.daughters)
        primaries = self.check_primary_energy(frame, primaries)

        MMCTrackList = frame[self.mmctracklist]
        if self.daughters:
            MMCTrackList = [
                track
                for track in MMCTrackList
                if frame[self.mctree].get_primary(track.GetI3Particle())
                in primaries
            ]
            MMCTrackList = simclasses.I3MMCTrackList(MMCTrackList)

        for track in MuonGun.Track.harvest(frame[self.mctree], MMCTrackList):
            assert track.is_track, "Track is not a track"

            # Find distance to entrance and exit from sampling volume
            intersections = self.hull.surface.intersection(
                track.pos, track.dir
            )
            # Get the corresponding energies

            e0 = track.get_energy(intersections.first)
            e1 = track.get_energy(intersections.second)

            # Accumulate
            e_deposited += e0 - e1
            e_entrance += e0
            if self.daughters:
                assert e_entrance <= sum(
                    [p.energy for p in primaries]
                ), "Energy on entrance is greater than primary energy"
                assert e_deposited <= sum(
                    [p.energy for p in primaries]
                ), "Energy deposited is greater than primary energy"
        return e_entrance, e_deposited

    def total_cascade_energy(
        self,
        frame: "icetray.I3Frame",
    ) -> float:
        """Get the total energy of cascade particles on entrance."""
        e_deposited = 0

        particles = self.get_primaries(frame, self.daughters)

        particles = np.array([p for p in particles if (not p.is_track)])

        if len(particles) == 0:
            return e_deposited

        pos, direc, length = np.asarray(
            [
                [
                    np.array(p.pos),
                    np.array([p.dir.x, p.dir.y, p.dir.z]),
                    p.length,
                ]
                for p in particles
            ],
            dtype=object,
        ).T

        length = length.astype(float)

        # replace length nan with 0
        length[np.isnan(length)] = 0
        pos = pos + direc * length
        pos = np.stack(pos)

        in_volume = self.hull.point_in_hull(pos)
        particles = particles[in_volume]

        for particle in particles:
            if particle.is_cascade:
                e_deposited += particle.energy
            else:
                daughters = dataclasses.I3MCTree.get_daughters(
                    frame[self.mctree], particle
                )

                pos, direc, length = np.asarray(
                    [
                        [
                            np.array(p.pos),
                            np.array([p.dir.x, p.dir.y, p.dir.z]),
                            p.length,
                        ]
                        for p in daughters
                    ],
                    dtype=object,
                ).T

                length = length.astype(float)

                # replace length nan with 0
                length[np.isnan(length)] = 0
                pos = pos + direc * length
                pos = np.stack(pos)

                in_volume = self.hull.point_in_hull(pos)
                daughters = np.array(daughters)[in_volume]

            while len(daughters) > 0:
                daughter = daughters[0]
                daughters = daughters[1:]
                length = daughter.length
                if daughter.is_track:
                    continue
                if length == np.nan:
                    length = 0
                if daughter.is_cascade and daughter.shape != "Dark":
                    e_deposited += daughter.energy
                else:
                    daughters = np.concatenate(
                        [
                            daughters,
                            dataclasses.I3MCTree.get_daughters(
                                frame[self.mctree], daughter
                            ),
                        ]
                    )
        return e_deposited
