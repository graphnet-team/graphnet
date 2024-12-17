"""Extract all the visible particles entering the volume."""

from typing import Dict, Any, List

from .utilities import GCD_hull
from icecube import dataclasses, MuonGun, icetray
from .i3extractor import I3Extractor

import numpy as np


class I3TotalEExtractor(I3Extractor):
    """Extract all the visible particles entering the volume."""

    def __init__(
        self,
        hull: GCD_hull,
        mctree: str = "I3MCTree",
        mmctracklist: str = " MMCTrackList",
        extractor_name: str = "TotalEonEntrance",
        daughters: bool = False,
    ):
        """Create a ConvexHull object from the GCD file."""
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.daughters = daughters
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract all the visible particles entering the volume."""
        output = {}
        if self.frame_contains_info(frame):

            e_entrance_track, e_deposited_track, checked_id_list = (
                self.total_track_energy(frame)
            )

            e_entrance_cascade, e_deposited_cascade, checked_id_list = (
                self.total_cascade_energy(frame, checked_id_list)
            )

            if self.daughters:
                primary_energy = frame[self.mctree].get_primaries()[0].energy
            else:
                primary_energy = sum(
                    [
                        p.energy
                        for p in dataclasses.I3MCTree.get_primaries(
                            frame[self.mctree]
                        )
                    ]
                )

            output.update(
                {
                    "e_entrance_track_"
                    + self._extractor_name: e_entrance_track,
                    "e_deposited_track_"
                    + self._extractor_name: e_deposited_track,
                    "e_entrance_cascade_"
                    + self._extractor_name: e_entrance_cascade,
                    "e_deposited_cascade_"
                    + self._extractor_name: e_deposited_cascade,
                    "e_fraction_"
                    + self._extractor_name: (
                        e_entrance_track + e_entrance_cascade
                    )
                    / primary_energy,
                }
            )

        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the necessary information."""
        return self.mctree in frame and self.mmctracklist in frame

    def total_track_energy(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> tuple[int, int, Any]:
        """Get the total energy of track particles on entrance."""
        e_entrance = 0
        e_deposited = 0
        primary = frame[self.mctree].get_primaries()[0]
        for track in MuonGun.Track.harvest(
            frame[self.mctree], frame[self.mmctracklist]
        ):
            track_id = track.particle.id
            if self.daughters:
                if (
                    dataclasses.I3MCTree.parent(
                        frame[self.mctree], track.particle
                    )
                    != primary
                ):
                    continue

            if track_id in checked_id_list:
                continue
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
            checked_id_list.append(track_id)

        return e_entrance, e_deposited, checked_id_list

    def total_cascade_energy(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> tuple[int, int, Any]:
        """Get the total energy of cascade particles on entrance."""
        e_entrance = 0
        e_deposited = 0

        if self.daughters:
            particles = [
                dataclasses.I3MCTree.get_primaries(frame[self.mctree])[0]
            ]
        else:
            particles = dataclasses.I3MCTree.get_primaries(frame[self.mctree])

        for particle in particles:
            if (particle.id not in checked_id_list) & (not particle.is_track):
                checked_id_list.append(particle.id)

                decay_pos = particle.pos + particle.dir * particle.length

                if self.hull.point_in_hull(decay_pos):
                    e_entrance += particle.energy
                    daughters = dataclasses.I3MCTree.get_daughters(
                        frame[self.mctree], particle
                    )
                    while daughters:
                        daughter = daughters[0]
                        daughters = daughters[1:]
                        length = daughter.length
                        if length == np.nan:
                            length = 0
                        decay_pos = (
                            daughter.pos + daughter.dir * daughter.length
                        )
                        checked_id_list.append(daughter.id)
                        if self.hull.point_in_hull(decay_pos):
                            if daughter.is_cascade:
                                e_deposited += daughter.energy
                            else:
                                daughters.extend(
                                    dataclasses.I3MCTree.get_daughters(
                                        frame[self.mctree], daughter
                                    )
                                )
                        else:
                            daughters.extend(
                                dataclasses.I3MCTree.get_daughters(
                                    frame[self.mctree], daughter
                                )
                            )
        return e_entrance, e_deposited, checked_id_list
