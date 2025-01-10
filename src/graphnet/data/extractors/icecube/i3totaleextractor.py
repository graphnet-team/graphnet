"""Extract all the visible particles entering the volume."""

from typing import Dict, Any, List, TYPE_CHECKING, Tuple

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


class I3TotalEExtractor(I3Extractor):
    """Energy on appearance of all visible particles in the volume."""

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
        checked_id_list: List = []
        if self.frame_contains_info(frame):

            e_entrance_track, e_deposited_track, checked_id_list = (
                self.total_track_energy(frame, checked_id_list=checked_id_list)
            )

            e_deposited_cascade, checked_id_list = self.total_cascade_energy(
                frame, checked_id_list
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
            e_total = e_entrance_track + e_deposited_cascade

            if self.daughters:
                assert (
                    e_total <= primary_energy
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

            output.update(
                {
                    "e_entrance_track_"
                    + self._extractor_name: e_entrance_track,
                    "e_deposited_track_"
                    + self._extractor_name: e_deposited_track,
                    "e_cascade_" + self._extractor_name: e_deposited_cascade,
                    "e_fraction_"
                    + self._extractor_name: (e_total) / primary_energy,
                }
            )

        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the necessary information."""
        return self.mctree in frame and self.mmctracklist in frame

    def total_track_energy(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> Tuple[int, int, Any]:
        """Get the total energy of track particles on entrance."""
        e_entrance = 0
        e_deposited = 0
        primary = frame[self.mctree].get_primaries()[0]
        for track in MuonGun.Track.harvest(
            frame[self.mctree], frame[self.mmctracklist]
        ):
            if self.daughters:
                if (
                    dataclasses.I3MCTree.parent(frame[self.mctree], track.id)
                    != primary
                ):
                    continue

            if track.id in checked_id_list:
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
            if self.daughters:
                assert (
                    e_entrance <= primary.energy
                ), "Energy on entrance is greater than primary energy"
                assert (
                    e_deposited <= primary.energy
                ), "Energy deposited is greater than primary energy"
            checked_id_list.append(track.id)

        return e_entrance, e_deposited, checked_id_list

    def total_cascade_energy(
        self, frame: "icetray.I3Frame", checked_id_list: List = []
    ) -> Tuple[int, List]:
        """Get the total energy of cascade particles on entrance."""
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
                    if particle.is_cascade:
                        e_deposited += particle.energy
                    else:
                        daughters = dataclasses.I3MCTree.get_daughters(
                            frame[self.mctree], particle
                        )

                    while daughters:
                        daughter = daughters[0]
                        daughters = daughters[1:]
                        length = daughter.length
                        if (daughter.is_track) or (
                            daughter.id in checked_id_list
                        ):
                            continue
                        if length == np.nan:
                            length = 0
                        decay_pos = (
                            daughter.pos + daughter.dir * daughter.length
                        )
                        checked_id_list.append(daughter.id)
                        if daughter.is_cascade:
                            if self.hull.point_in_hull(decay_pos):
                                e_deposited += daughter.energy
                        else:
                            daughters.extend(
                                dataclasses.I3MCTree.get_daughters(
                                    frame[self.mctree], daughter
                                )
                            )
        return e_deposited, checked_id_list
