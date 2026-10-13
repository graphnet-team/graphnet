"""Extract all the visible particles entering the volume."""

from typing import Dict, Any, TYPE_CHECKING, Tuple, Union, List

from .utilities.gcd_hull import GCD_hull
from .i3extractor import I3Extractor

import numpy as np

from graphnet.utilities.imports import has_icecube_package
from copy import deepcopy
from collections import deque

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
        MuonGun,
        simclasses,
    )  # pyright: reportMissingImports=false

    DARK = dataclasses.I3Particle.ParticleShape.Dark


class I3Calorimetry(I3Extractor):
    """Event level energy labeling for IceCube data.

    This class extracts cumulative energy information from all visible
    particles entering the detector volume, during the event. The recorded energy is split into a "target" and "background" contribution, where the target contribution consists of all particles that downstream of neutrino primaries (or only the highest energy neutrino primary if the corresponding flag is set) and the background contribution consists of all other particles. The recorded energy is further split into a "track" and "cascade" contribution, where the track contribution consists of all energy deposited by particles that are classified as tracks (i.e. if a track is recorded in the MMCTrackList) and the cascade contribution consists of all energy deposited by particles that are not classified as tracks. The recorded energy varies depending on whether or not the entrance_energy flag is set. If the entrance_energy flag is set, the energy recorded is the energy of all visible particles entering the volume as they enter the volume. If the entrance_energy flag is not set, then the recorded energy is only the energy deposited inside the volume i.e. if a muon enters the volume with 100 GeV and leaves with 80 GeV, then the track energy recorded would be 20 GeV.

    Returns a dictionary with the following keys
    - e_track_target: Energy entering/deposited by target tracks.
    - e_cascade_target: Energy entering/deposited by target cascades.
    - e_target: Total energy entering/deposited by target particles.
    - e_track_bkg: Energy entering/deposited by background tracks.
    - e_cascade_bkg: Energy entering/deposited by background cascades.
    - e_bkg: Total energy entering/deposited by background particles.
    - fraction_target_total: Fraction of total recorded energy that is from target particles.
    - fraction_target_primary: Fraction of primar(y/ies) energy that is recorded as entering/deposited by target particles.
    - fraction_cascade_target: Fraction of target energy that is from cascades.
    - e_track_total: Total energy entering/deposited by tracks.
    - e_cascade_total: Total energy entering/deposited by cascades.
    - e_total: Total energy entering/deposited by all particles.
    - fraction_cascade_total: Fraction of total recorded energy that is from cascades.
    """

    def __init__(
        self,
        hull: GCD_hull,
        mctree: str = "I3MCTree",
        mmctracklist: str = "MMCTrackList",
        extractor_name: str = "I3Calorimetry",
        highest_energy_primary: bool = False,
        entrance_energy: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a ConvexHull object from the GCD file.

        hull: GCD_hull object containing the convex hull
        of the detector volume.
        mctree: Name of the I3MCTree in the frame.
        mmctracklist: Name of the MMCTrackList in the frame.
        extractor_name: Name of the extractor.
        highest_energy_primary: If True, only consider particles that are
            daughters of the highest energy primary.
        entrance_energy: If True, consider entrance energy.
        """
        # Member variable(s)
        self.hull = hull
        self.mctree = mctree
        self.mmctracklist = mmctracklist
        self.highest_energy_primary = highest_energy_primary
        self.entrance_energy = entrance_energy
        # Base class constructor
        super().__init__(extractor_name=extractor_name, **kwargs)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract all the visible particles entering the volume."""
        output = {}
        if self.frame_contains_info(frame):
            target_tree, bkg_tree = self.split_mc_tree(
                frame, highest_energy_primary=self.highest_energy_primary
            )
            if len(target_tree) + len(bkg_tree) != len(frame[self.mctree]):
                raise ValueError(
                    f"Split mctree has different number of particles than original mctree\nOriginal mctree: {len(frame[self.mctree])}\nTarget tree: {len(target_tree)}\nBkg tree: {len(bkg_tree)}\nHighest energy primary flag: {self.highest_energy_primary}\nEvent header: {frame['I3EventHeader']}"
                )

            # For the target we consider either all neutrino primary products or only the highest energy primary of the neutrino depending on the flag.
            target_primaries = self.get_primaries(
                target_tree,
                daughters=True,
                highest_energy_primary=self.highest_energy_primary,
            )
            target_primaries = self.check_primary_energy(
                target_tree, target_primaries
            )
            # For background we consider everything in the background tree.
            bkg_primaries = self.get_primaries(
                bkg_tree, daughters=False, highest_energy_primary=False
            )
            bkg_primaries = self.check_primary_energy(bkg_tree, bkg_primaries)

            target_primaries_energy = sum([p.energy for p in target_primaries])
            bkg_primaries_energy = sum([p.energy for p in bkg_primaries])

            if len(target_tree) > 0:
                e_track_target = self.total_track_energy(
                    frame,
                    target_tree,
                    entrance_energy=self.entrance_energy,
                )
            else:
                e_track_target = 0.0
            # Sanity check ensuring no double counting

            if not (e_track_target <= target_primaries_energy * (1 + 1e-6)):
                raise ValueError(
                    f"Energy deposited in target is greater than primary energy: {e_track_target} > {target_primaries_energy}\nEvent header: {frame['I3EventHeader']}"
                )
            if len(bkg_tree) > 0:
                e_track_bkg = self.total_track_energy(
                    frame,
                    bkg_tree,
                    entrance_energy=self.entrance_energy,
                )
            else:
                e_track_bkg = 0.0

            if len(target_tree) > 0:
                e_cascade_target = self.total_cascade_energy(
                    target_tree, target_primaries
                )
            else:
                e_cascade_target = 0.0
            if len(bkg_tree) > 0:
                e_cascade_bkg = self.total_cascade_energy(
                    bkg_tree, bkg_primaries
                )
            else:
                e_cascade_bkg = 0.0

            e_total_target = e_track_target + e_cascade_target
            e_total_bkg = e_track_bkg + e_cascade_bkg

            e_total = e_total_target + e_total_bkg

            if e_total == 0.0:
                self.warning(
                    "No energy deposited in the hull, "
                    "Think about increasing the padding."
                    f"\nCurrent padding: {self.hull.padding}"
                    f"\nEvent header: {frame['I3EventHeader']}"
                )

            if not (
                e_total_target <= (target_primaries_energy * (1 + 1e-6))
                or (e_total_target - target_primaries_energy < 0.5)
            ):
                raise ValueError(
                    "Total energy is greater than primary energy\n"
                    f"Total energy: {e_total_target}\n"
                    f"Primary energy: {target_primaries_energy}\n"
                    f"Track deposited energy: {e_track_target}\n"
                    f"Cascade deposited energy: {e_cascade_target}\n"
                    f"{frame['I3EventHeader']}"
                )

            if not (
                e_total_bkg <= (bkg_primaries_energy * (1 + 1e-6))
                or (e_total_bkg - bkg_primaries_energy < 0.5)
            ):
                raise ValueError(
                    "Total background energy is greater than primary energy\n"
                    f"Total background energy: {e_total_bkg}\n"
                    f"Background primary energy: {bkg_primaries_energy}\n"
                    f"Track deposited background energy: {e_track_bkg}\n"
                    f"Cascade deposited background energy: {e_cascade_bkg}\n"
                    f"{frame['I3EventHeader']}"
                )

            fraction_target_total = (
                e_total_target / e_total if e_total > 0 else 0.0
            )
            target_cascade_fraction = (
                e_cascade_target / e_total_target
                if e_total_target > 0
                else 0.0
            )

            if e_total > 0:
                cascade_fraction_tot = (
                    e_cascade_target + e_cascade_bkg
                ) / e_total

            if target_primaries_energy > 0:
                fraction_primary = e_total_target / target_primaries_energy
            else:
                fraction_primary = None
            output.update(
                {
                    "e_track_target_" + self._extractor_name: e_track_target,
                    "e_cascade_target_"
                    + self._extractor_name: e_cascade_target,
                    "e_target_" + self._extractor_name: e_total_target,
                    "e_track_bkg_" + self._extractor_name: e_track_bkg,
                    "e_cascade_bkg_" + self._extractor_name: e_cascade_bkg,
                    "e_bkg_" + self._extractor_name: e_total_bkg,
                    "fraction_target_total_"
                    + self._extractor_name: fraction_target_total,
                    "fraction_target_primary_"
                    + self._extractor_name: fraction_primary,
                    "fraction_cascade_target_"
                    + self._extractor_name: target_cascade_fraction,
                    "e_track_total_"
                    + self._extractor_name: e_track_target
                    + e_track_bkg,
                    "e_cascade_total_"
                    + self._extractor_name: e_cascade_target
                    + e_cascade_bkg,
                    "e_total_" + self._extractor_name: e_total,
                    "fraction_cascade_total_"
                    + self._extractor_name: (
                        cascade_fraction_tot if e_total > 0 else 0.0
                    ),
                }
            )

        output = {k: v for k, v in output.items() if k not in self._exclude}
        return output

    def frame_contains_info(self, frame: "icetray.I3Frame") -> bool:
        """Check if the frame contains the necessary information."""
        return self.mctree in frame and self.mmctracklist in frame

    def total_track_energy(
        self,
        frame: "icetray.I3Frame",
        mctree: "dataclasses.I3MCTree",
        entrance_energy: bool = False,
    ) -> float:
        """Get the total energy deposited by tracks entering the volume.

        If entrance_energy is True, return the total energy entering the
        volume as tracks instead of the energy deposited.
        """
        energy = 0

        if self._is_corsika:
            mmc_track_list = frame[self.mmctracklist]
        else:
            mmc_track_list = self.filter_track_list(
                mctree, frame[self.mmctracklist]
            )

        track_list = deque(MuonGun.Track.harvest(mctree, mmc_track_list))

        while len(track_list) > 0:
            track = track_list.popleft()

            try:
                particle = mctree.get_particle(track.id)
            except RuntimeError:
                # If the particle does not exist in the mctree, that means a particle further up was processed and therefore it should not be counted
                continue

            # Find distance to entrance and exit from sampling volume
            intersections = self.hull.surface.intersection(
                track.pos, track.dir
            )

            # Check if the track actually enters the volume. Values are NAN if the ray does not intersect the hull, negative if the intersection is behind the "origin". Uncertain if we can have negative intersections for both first and second intersection without it being converted to NAN (no intersection) but we check for both to be sure.
            if not (
                (
                    np.isfinite(intersections.first)
                    and (intersections.first < particle.length)
                )
                and (
                    np.isfinite(intersections.second)
                    and intersections.second > 0
                )
            ):
                continue

            # Get the corresponding energies
            try:
                e0 = track.get_energy(intersections.first)
                e1 = track.get_energy(intersections.second)

            except RuntimeError as e:
                if (
                    "sum of losses is smaller than "
                    "energy at last checkpoint" in str(e)
                ):
                    hdr = frame["I3EventHeader"]
                    e.add_note(f"Error in MuonGun track in event {hdr}")
                    self.warning(
                        f"Skipping bad track {hdr}: {e}"
                        f"\nTotal energy of offending particle: {particle.energy}"
                    )
                    continue
                else:
                    raise

            # Accumulate
            if entrance_energy:
                energy += e0
                # if we are looking at the entrance energy then all energy entering the volume as a track is considered "track energy" even if it is later deposited in a cascade, so we remove all descendants of the track from the mctree to avoid double counting
                mctree.erase(track.id)
            else:
                energy += e0 - e1
                # if we are looking at the deposited energy then we only want to remove the tracks that have either deposited all their energy in the volume or left the volume again thus descendants cannot produce cascades in the volume.
                if (e1 == 0) or (intersections.second < particle.length):
                    mctree.erase(track.id)
        return energy

    def total_cascade_energy(
        self,
        mctree: "dataclasses.I3MCTree",
        primaries: "dataclasses.ListI3Particle",
    ) -> float:
        """Get the total energy of cascade particles on entrance."""
        particles = deque(primaries)

        if len(particles) == 0:
            return 0.0

        pos_list, direc_list, length_list, cascade_bool, energies = (
            [],
            [],
            [],
            [],
            [],
        )

        while len(particles) > 0:
            p = particles.popleft()
            p_children = mctree.get_daughters(p)
            if len(p_children) > 0:
                particles.extend(p_children)
                continue
            if p.is_track or p.shape == DARK:
                continue

            pos_list.append([p.pos.x, p.pos.y, p.pos.z])
            direc_list.append([p.dir.x, p.dir.y, p.dir.z])
            length_list.append(p.length)
            cascade_bool.append(p.is_cascade)
            energies.append(p.energy)

        if len(energies) == 0:
            return 0.0

        length = np.array(length_list).astype(float)
        length[np.isnan(length)] = 0
        pos = np.asarray(pos_list)
        direc = np.asarray(direc_list)
        cascade_bool = np.array(cascade_bool)
        energies = np.array(energies)
        pos = (pos.T + direc.T * length).T
        in_hull = self.hull.point_in_hull(pos)

        return np.sum(energies[cascade_bool & in_hull])

    def filter_track_list(
        self,
        mctree: "dataclasses.I3MCTree",
        track_list: "simclasses.I3MMCTrackList",
    ) -> "simclasses.I3MMCTrackList":
        """Filter the track list based on the mctree provided.

        (This function is only meant to run on target/bkg split trees)
        """
        filtered_track_list = []
        for track in track_list:
            try:
                mctree.get_particle(track.particle.id)
                filtered_track_list.append(track)
            except RuntimeError as e:
                if "particleID not found" in str(e):
                    # if particle is not found in the mctree then it should not be included as it in the other tree.
                    continue
                else:
                    raise e
        return simclasses.I3MMCTrackList(filtered_track_list)
