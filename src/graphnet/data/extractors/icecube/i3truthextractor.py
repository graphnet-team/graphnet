"""I3Extractor class(es) for extracting truth-level information."""

import numpy as np
import matplotlib.path as mpath
from scipy.spatial import ConvexHull, Delaunay
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .i3extractor import I3Extractor
from .utilities.frames import (
    frame_is_montecarlo,
    frame_is_noise,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        dataclasses,
        icetray,
        phys_services,
        dataio,
        LeptonInjector,
    )  # pyright: reportMissingImports=false


class I3TruthExtractor(I3Extractor):
    """Class for extracting truth-level information."""

    def __init__(
        self,
        name: str = "truth",
        borders: Optional[List[np.ndarray]] = None,
        mctree: Optional[str] = "I3MCTree",
        extend_boundary: Optional[float] = 0.0,
    ):
        """Construct I3TruthExtractor.

        Args:
            name: Name of the `I3Extractor` instance.
            borders: Array of boundaries of the detector volume as ((x,y),z)-
                coordinates, for identifying, e.g., particles starting and
                stopping within the detector. Defaults to hard-coded boundary
                coordinates.
            mctree: Str of which MCTree to use for truth values.
            extend_boundary: Distance to extend the convex hull of the detector
                for defining starting events.
        """
        # Base class constructor
        super().__init__(name)

        if borders is None:
            border_xy = np.array(
                [
                    (-256.1400146484375, -521.0800170898438),
                    (-132.8000030517578, -501.45001220703125),
                    (-9.13000011444092, -481.739990234375),
                    (114.38999938964844, -461.989990234375),
                    (237.77999877929688, -442.4200134277344),
                    (361.0, -422.8299865722656),
                    (405.8299865722656, -306.3800048828125),
                    (443.6000061035156, -194.16000366210938),
                    (500.42999267578125, -58.45000076293945),
                    (544.0700073242188, 55.88999938964844),
                    (576.3699951171875, 170.9199981689453),
                    (505.2699890136719, 257.8800048828125),
                    (429.760009765625, 351.0199890136719),
                    (338.44000244140625, 463.7200012207031),
                    (224.5800018310547, 432.3500061035156),
                    (101.04000091552734, 412.7900085449219),
                    (22.11000061035156, 509.5),
                    (-101.05999755859375, 490.2200012207031),
                    (-224.08999633789062, 470.8599853515625),
                    (-347.8800048828125, 451.5199890136719),
                    (-392.3800048828125, 334.239990234375),
                    (-437.0400085449219, 217.8000030517578),
                    (-481.6000061035156, 101.38999938964844),
                    (-526.6300048828125, -15.60000038146973),
                    (-570.9000244140625, -125.13999938964844),
                    (-492.42999267578125, -230.16000366210938),
                    (-413.4599914550781, -327.2699890136719),
                    (-334.79998779296875, -424.5),
                ]
            )
            border_z = np.array([-512.82, 524.56])
            self._borders = [border_xy, border_z]
        else:
            self._borders = borders

        self._extend_boundary = extend_boundary
        self._mctree = mctree

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

        coordinates = []
        for omkey, g in self._gcd_dict.items():
            if g.position.z > 1200:
                continue  # We want to exclude icetop
            coordinates.append([g.position.x, g.position.y, g.position.z])
        coordinates = np.array(coordinates)

        if self._extend_boundary != 0.0:
            center = np.mean(coordinates, axis=0)
            d = coordinates - center
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            dn = d / norms
            coordinates = coordinates + dn * self._extend_boundary

        hull = ConvexHull(coordinates)

        self.hull = hull
        self.delaunay = Delaunay(coordinates[self.hull.vertices])

    def __call__(
        self, frame: "icetray.I3Frame", padding_value: Any = -1
    ) -> Dict[str, Any]:
        """Extract truth-level information."""
        is_mc = frame_is_montecarlo(frame, self._mctree)
        is_noise = frame_is_noise(frame, self._mctree)
        sim_type = self._find_data_type(is_mc, self._i3_file, frame)

        output = {
            "energy": padding_value,
            "position_x": padding_value,
            "position_y": padding_value,
            "position_z": padding_value,
            "azimuth": padding_value,
            "zenith": padding_value,
            "pid": padding_value,
            "event_time": frame["I3EventHeader"].start_time.utc_daq_time,
            "sim_type": sim_type,
            "interaction_type": padding_value,
            "elasticity": padding_value,
            "RunID": frame["I3EventHeader"].run_id,
            "SubrunID": frame["I3EventHeader"].sub_run_id,
            "EventID": frame["I3EventHeader"].event_id,
            "SubEventID": frame["I3EventHeader"].sub_event_id,
            "dbang_decay_length": padding_value,
            "track_length": padding_value,
            "stopped_muon": padding_value,
            "energy_track": padding_value,
            "energy_cascade": padding_value,
            "inelasticity": padding_value,
            "DeepCoreFilter_13": padding_value,
            "CascadeFilter_13": padding_value,
            "MuonFilter_13": padding_value,
            "OnlineL2Filter_17": padding_value,
            "L3_oscNext_bool": padding_value,
            "L4_oscNext_bool": padding_value,
            "L5_oscNext_bool": padding_value,
            "L6_oscNext_bool": padding_value,
            "L7_oscNext_bool": padding_value,
            "starting": padding_value,
        }

        # Only InIceSplit P frames contain ML appropriate I3RecoPulseSeriesMap etc.
        # At low levels i3files contain several other P frame splits (e.g NullSplit),
        # we remove those here.
        if frame["I3EventHeader"].sub_event_stream not in [
            "InIceSplit",
            "Final",
        ]:
            return output

        if "FilterMask" in frame:
            if "DeepCoreFilter_13" in frame["FilterMask"]:
                output["DeepCoreFilter_13"] = int(
                    bool(frame["FilterMask"]["DeepCoreFilter_13"])
                )
            if "CascadeFilter_13" in frame["FilterMask"]:
                output["CascadeFilter_13"] = int(
                    bool(frame["FilterMask"]["CascadeFilter_13"])
                )
            if "MuonFilter_13" in frame["FilterMask"]:
                output["MuonFilter_13"] = int(
                    bool(frame["FilterMask"]["MuonFilter_13"])
                )
            if "OnlineL2Filter_17" in frame["FilterMask"]:
                output["OnlineL2Filter_17"] = int(
                    bool(frame["FilterMask"]["OnlineL2Filter_17"])
                )

        elif "DeepCoreFilter_13" in frame:
            output["DeepCoreFilter_13"] = int(bool(frame["DeepCoreFilter_13"]))

        if "L3_oscNext_bool" in frame:
            output["L3_oscNext_bool"] = int(bool(frame["L3_oscNext_bool"]))

        if "L4_oscNext_bool" in frame:
            output["L4_oscNext_bool"] = int(bool(frame["L4_oscNext_bool"]))

        if "L5_oscNext_bool" in frame:
            output["L5_oscNext_bool"] = int(bool(frame["L5_oscNext_bool"]))

        if "L6_oscNext_bool" in frame:
            output["L6_oscNext_bool"] = int(bool(frame["L6_oscNext_bool"]))

        if "L7_oscNext_bool" in frame:
            output["L7_oscNext_bool"] = int(bool(frame["L7_oscNext_bool"]))

        if is_mc and (not is_noise):
            (
                MCInIcePrimary,
                interaction_type,
                elasticity,
            ) = self._get_primary_particle_interaction_type_and_elasticity(
                frame, sim_type
            )

            try:
                (
                    energy_track,
                    energy_cascade,
                    inelasticity,
                ) = self._get_primary_track_energy_and_inelasticity(frame)
            except RuntimeError:  # track energy fails on northeren tracks with ""Hadrons" has no mass implemented. Cannot get total energy."
                energy_track, energy_cascade, inelasticity = (
                    padding_value,
                    padding_value,
                    padding_value,
                )

            output.update(
                {
                    "energy": MCInIcePrimary.energy,
                    "position_x": MCInIcePrimary.pos.x,
                    "position_y": MCInIcePrimary.pos.y,
                    "position_z": MCInIcePrimary.pos.z,
                    "azimuth": MCInIcePrimary.dir.azimuth,
                    "zenith": MCInIcePrimary.dir.zenith,
                    "pid": MCInIcePrimary.pdg_encoding,
                    "interaction_type": interaction_type,
                    "elasticity": elasticity,
                    "dbang_decay_length": self._extract_dbang_decay_length(
                        frame, padding_value
                    ),
                    "energy_track": energy_track,
                    "energy_cascade": energy_cascade,
                    "inelasticity": inelasticity,
                }
            )
            if abs(output["pid"]) == 13:
                output.update(
                    {
                        "track_length": MCInIcePrimary.length,
                    }
                )
                muon_final = self._muon_stopped(output, self._borders)
                output.update(
                    {
                        "position_x": muon_final[
                            "x"
                        ],  # position_xyz has no meaning for muons. These will now be updated to muon final position, given track length/azimuth/zenith
                        "position_y": muon_final["y"],
                        "position_z": muon_final["z"],
                        "stopped_muon": muon_final["stopped"],
                    }
                )

            starting = self._contained_vertex(output)
            output.update(
                {
                    "starting": starting,
                }
            )

        return output

    def _extract_dbang_decay_length(
        self, frame: "icetray.I3Frame", padding_value: float = -1
    ) -> float:
        mctree = frame[self._mctree]
        try:
            p_true = mctree.primaries[0]
            p_daughters = mctree.get_daughters(p_true)
            if len(p_daughters) == 2:
                for p_daughter in p_daughters:
                    if p_daughter.type == dataclasses.I3Particle.Hadrons:
                        casc_0_true = p_daughter
                    else:
                        hnl_true = p_daughter
                hnl_daughters = mctree.get_daughters(hnl_true)
            else:
                decay_length = padding_value
                hnl_daughters = []

            if len(hnl_daughters) > 0:
                for count_hnl_daughters, hnl_daughter in enumerate(
                    hnl_daughters
                ):
                    if not count_hnl_daughters:
                        casc_1_true = hnl_daughter
                    else:
                        assert casc_1_true.pos == hnl_daughter.pos
                        casc_1_true.energy = (
                            casc_1_true.energy + hnl_daughter.energy
                        )
                decay_length = (
                    phys_services.I3Calculator.distance(
                        casc_0_true, casc_1_true
                    )
                    / icetray.I3Units.m
                )

            else:
                decay_length = padding_value
            return decay_length
        except:  # noqa: E722
            return padding_value

    def _muon_stopped(
        self,
        truth: Dict[str, Any],
        borders: List[np.ndarray],
        shrink_horizontally: float = 100.0,
        shrink_vertically: float = 100.0,
    ) -> Dict[str, Any]:
        """Calculate whether a simulated muon within the detector volume.

        IMPORTANT: The final position of the muon is saved in truth extractor/
        databases as position_x, position_y and position_z. This is analogouos
        to the neutrinos whose interaction vertex is saved under the same name.

        Args:
            truth: Dictionary of already extracted truth-level information.
            borders: The first entry are the (x,y) coordinates, the second
                entry is the z-axis min/max depths. See I3TruthExtractor
                constructor for hard-code example.
            shrink_horizontally: Shrink (x,y)-plane further with exclusion
                zone. Defaults to 100 meters. shrink_vertically: Further shrink
                detector depth with exclusion height. Defaults to 100 meters.

        Returns:
            Dictionary containing the (x,y,z)-coordinates of final the muon
                position as well as a boolean indicating whether the muon
                stopped within the chosen fiducial volume.
        """
        # @TODO: Remove hard-coded border coords and replace with GCD file
        # contents using string no's
        border = mpath.Path(borders[0])

        start_pos = np.array(
            [truth["position_x"], truth["position_y"], truth["position_z"]]
        )

        travel_vec = -1 * np.array(
            [
                truth["track_length"]
                * np.cos(truth["azimuth"])
                * np.sin(truth["zenith"]),
                truth["track_length"]
                * np.sin(truth["azimuth"])
                * np.sin(truth["zenith"]),
                truth["track_length"] * np.cos(truth["zenith"]),
            ]
        )

        end_pos = start_pos + travel_vec

        stopped_xy = border.contains_point(
            (end_pos[0], end_pos[1]), radius=-shrink_horizontally
        )
        stopped_z = (end_pos[2] > borders[1][0] + shrink_vertically) * (
            end_pos[2] < borders[1][1] - shrink_vertically
        )

        return {
            "x": end_pos[0],
            "y": end_pos[1],
            "z": end_pos[2],
            "stopped": (stopped_xy * stopped_z),
        }

    def _get_primary_particle_interaction_type_and_elasticity(
        self,
        frame: "icetray.I3Frame",
        sim_type: str,
        padding_value: float = -1.0,
    ) -> Tuple[Any, int, float]:
        """Return primary particle, interaction type, and elasticity.

        A case handler that does two things:
            1) Catches issues related to determining the primary MC particle.
            2) Error handles cases where interaction type and elasticity
                doesn't exist

        Args:
            frame: Physics frame containing MC record.
            sim_type: Simulation type.
            padding_value: The value used for padding.

        Returns
            A tuple containing the MCInIcePrimary, if it exists; the primary
                particle, encoded as 1 (charged current), 2 (neutral current),
                or 0 (neither); and the elasticity in the range ]0,1[.
        """
        if sim_type != "noise":
            try:
                MCInIcePrimary = frame["MCInIcePrimary"]
            except KeyError:
                MCInIcePrimary = frame[self._mctree][0]
            if (
                MCInIcePrimary.energy != MCInIcePrimary.energy
            ):  # This is a nan check. Only happens for some muons where second item in MCTree is primary. Weird!
                MCInIcePrimary = frame[self._mctree][
                    1
                ]  # For some strange reason the second entry is identical in all variables and has no nans (always muon)
        else:
            MCInIcePrimary = None

        if sim_type == "LeptonInjector":
            event_properties = frame["EventProperties"]
            final_state_1 = event_properties.finalType1
            if final_state_1 in [
                dataclasses.I3Particle.NuE,
                dataclasses.I3Particle.NuMu,
                dataclasses.I3Particle.NuTau,
                dataclasses.I3Particle.NuEBar,
                dataclasses.I3Particle.NuMuBar,
                dataclasses.I3Particle.NuTauBar,
            ]:
                interaction_type = 2  # NC
            else:
                interaction_type = 1  # CC

            elasticity = 1 - event_properties.finalStateY

        else:
            try:
                interaction_type = frame["I3MCWeightDict"]["InteractionType"]
            except KeyError:
                interaction_type = int(padding_value)

            try:
                elasticity = 1 - frame["I3MCWeightDict"]["BjorkenY"]
            except KeyError:
                elasticity = padding_value

        return MCInIcePrimary, interaction_type, elasticity

    def _get_primary_track_energy_and_inelasticity(
        self,
        frame: "icetray.I3Frame",
    ) -> Tuple[float, float, float]:
        """Get the total energy of tracks from primary, and inelasticity.

        Args:
            frame: Physics frame containing MC record.

        Returns:
            Tuple containing the energy of tracks from primary, and the
                corresponding inelasticity.
        """
        mc_tree = frame[self._mctree]
        primary = mc_tree.primaries[0]
        daughters = mc_tree.get_daughters(primary)
        tracks = []
        for daughter in daughters:
            if (
                str(daughter.shape) == "StartingTrack"
                or str(daughter.shape) == "Dark"
            ):
                tracks.append(daughter)

        energy_total = primary.total_energy
        energy_track = sum(track.total_energy for track in tracks)
        energy_cascade = energy_total - energy_track
        inelasticity = 1.0 - energy_track / energy_total

        return energy_track, energy_cascade, inelasticity

    # Utility methods
    def _find_data_type(
        self, mc: bool, input_file: str, frame: "icetray.I3Frame"
    ) -> str:
        """Determine the data type.

        Args:
            mc: Whether `input_file` is Monte Carlo simulation.
            input_file: Path to I3-file.
            frame: Physics frame containing MC record

        Returns:
            The simulation/data type.
        """
        # @TODO: Rewrite to automatically infer `mc` from `input_file`?
        if not mc:
            sim_type = "data"
        elif "muon" in input_file:
            sim_type = "muongun"
        elif "corsika" in input_file:
            sim_type = "corsika"
        elif "genie" in input_file or "nu" in input_file.lower():
            sim_type = "genie"
        elif "noise" in input_file:
            sim_type = "noise"
        elif frame.Has("EventProprties") or frame.Has(
            "LeptonInjectorProperties"
        ):
            sim_type = "LeptonInjector"
        elif frame.Has("I3MCWeightDict"):
            sim_type = "NuGen"
        else:
            raise NotImplementedError("Could not determine data type.")
        return sim_type

    def _contained_vertex(self, truth: Dict[str, Any]) -> bool:
        """Determine if an event is starting based on vertex position.

        Args:
            truth: Dictionary of already extracted truth-level information.

        Returns:
            True/False if vertex is inside detector.
        """
        vertex = np.array(
            [truth["position_x"], truth["position_y"], truth["position_z"]]
        )
        return self.delaunay.find_simplex(vertex) >= 0
