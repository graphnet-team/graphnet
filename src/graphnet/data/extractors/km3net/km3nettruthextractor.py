"""Code to extract the truth event information from the KM3NeT ROOT file."""

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
from graphnet.utilities.imports import has_km3net_package

from .km3netextractor import KM3NeTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id_run_by_run,
    xyz_dir_to_zen_az,
    assert_no_uint_values,
    filter_None_NaN,
)
if has_km3net_package() or TYPE_CHECKING:
    import km3io as ki

class KM3NeTTruthExtractor(KM3NeTExtractor):
    """Class for extracting the truth information from a file."""

    def __init__(
        self,
        name: str,
        add_hnl_info: bool = False,
        add_reco_info: bool = False,
    ):
        """Initialize the class to extract the truth information."""
        super().__init__(name)
        self.add_hnl_info = add_hnl_info
        self.add_reco_info = add_reco_info

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract truth event information as a dataframe."""
        truth_df = self._extract_truth_dataframe(file)
        truth_df = assert_no_uint_values(truth_df)  # asserts the data format

        return truth_df

    def _extract_particle_attributes(
        self,
        particle: Any,
        attributes: Any,
        padding_value: int,
        prefix: str,
        file_type: str,
    ) -> Dict[str, Any]:
        if (file_type != "data") and (file_type != "noise"):
            dict_particle = {}
            for attr in attributes:
                dict_particle[prefix + "_" + attr] = filter_None_NaN(
                    getattr(particle, attr), padding_value
                )

            # also add the zenith and the azimuth computed from the direction
            if (
                "dir_x" in attributes
                and "dir_y" in attributes
                and "dir_z" in attributes
            ):
                zen, az = xyz_dir_to_zen_az(
                    filter_None_NaN(getattr(particle, "dir_x"), padding_value),
                    filter_None_NaN(getattr(particle, "dir_y"), padding_value),
                    filter_None_NaN(getattr(particle, "dir_z"), padding_value),
                    padding_value,
                )
                dict_particle[prefix + "_zenith"] = zen
                dict_particle[prefix + "_azimuth"] = az
        elif (
            (file_type != "hnl")
            and (file_type != "neutrino")
            and (file_type != "muon")
        ):
            dict_particle = {}
            for attr in attributes:
                dict_particle[prefix + "_" + attr] = padding_value * np.ones(
                    len(particle.pos_x)
                )

            # also add the zenith and the azimuth computed from the direction
            if (
                "dir_x" in attributes
                and "dir_y" in attributes
                and "dir_z" in attributes
            ):
                dict_particle[prefix + "_zenith"] = padding_value * np.ones(
                    len(particle.pos_x)
                )
                dict_particle[prefix + "_azimuth"] = padding_value * np.ones(
                    len(particle.pos_x)
                )
        else:
            raise ValueError("File type not recognized")

        return dict_particle

    def _extract_event_attributes(
        self,
        file: Any,
        primaries: Any,
        primaries_jshower: Any,
        padding_value: int,
        file_type: str,
    ) -> Dict[str, Any]:
        """Return dictionary with event information."""
        evt_id, run_id, frame_index, trigger_counter = (
            np.array(file.id),
            np.array(file.run_id),
            np.array(file.frame_index),
            np.array(file.trigger_counter),
        )

        n_hits = np.array(file.n_hits)

        if file_type == "data":
            is_cc_flag = padding_value * np.ones(len(primaries_jshower.E))
            tau_topologies = padding_value * np.ones(len(primaries_jshower.E))
        elif file_type == "noise":
            is_cc_flag = padding_value * np.ones(len(primaries_jshower.E))
            tau_topologies = padding_value * np.ones(len(primaries_jshower.E))
        elif file_type == "hnl":
            is_cc_flag = np.ones(len(primaries.pos_x))
            tau_topologies = padding_value * np.ones(len(primaries.pos_x))
        elif file_type == "neutrino":
            is_cc_flag = np.array(file.w2list[:, 10] == 2)
            tau_topologies = [
                (
                    2
                    if 16 in np.abs(primaries.pdgid)
                    and 13 in np.abs(file.mc_trks.pdgid[i])
                    else 1 if 16 in np.abs(primaries.pdgid) else 3
                )
                for i in range(len(primaries.pdgid))
            ]
        elif file_type == "muon":
            is_cc_flag = padding_value * np.ones(len(primaries.pos_x))
            tau_topologies = padding_value * np.ones(len(primaries.pos_x))
        if file_type == "hnl":
            try:
                model_hnl = file.header.model.interaction
            except Exception:
                model_hnl = "none"
        elif (
            (file_type == "neutrino")
            or (file_type == "muon")
            or (file_type == "data")
            or (file_type == "noise")
        ):
            model_hnl = "none"
        else:
            raise ValueError("File type not recognized")

        unique_id = create_unique_id_run_by_run(
                    file_type=file_type,
                    run_id=np.array(file.run_id),
                    evt_id=np.array(file.id),
                    hnl_model=model_hnl,
                )

        return {
            "run_id": run_id,
            "evt_id": evt_id,
            "frame_index": frame_index,
            "trigger_counter": trigger_counter,
            "n_hits": n_hits,
            "event_no": np.array(unique_id).astype(int),
            "is_cc_flag": is_cc_flag,
            "tau_topology": tau_topologies,
        }

    def _extract_hnl_attributes(
        self,
        file: Any,
        primaries: Any,
        primaries_jshower: Any,
        padding_value: int,
        file_type: str,
    ) -> Dict[str, Any]:
        """Return dictionary with specific information of the HNL event."""
        if (file_type == "data") or (file_type == "noise"):
            return {
                "zenith_hnl": padding_value
                * np.ones(len(primaries_jshower.E)),
                "azimuth_hnl": padding_value
                * np.ones(len(primaries_jshower.E)),
                "angle_between_showers": padding_value
                * np.ones(len(primaries_jshower.E)),
                "Energy_hnl": padding_value
                * np.ones(len(primaries_jshower.E)),
                "Energy_second_shower": padding_value
                * np.ones(len(primaries_jshower.E)),
                "Energy_imbalance": padding_value
                * np.ones(len(primaries_jshower.E)),
                "distance": padding_value * np.ones(len(primaries_jshower.E)),
                "is_hnl": padding_value * np.ones(len(primaries_jshower.E)),
            }
        elif file_type == "hnl":
            if len(file.mc_trks.E[0]) > 3.5:
                hnl = file.mc_trks[:, 1]
                first_shower = file.mc_trks[:, 2]
                second_shower = file.mc_trks[:, 4]
            elif len(file.mc_trks.E[0]) < 3.5:
                hnl = file.mc_trks[:, 0]
                first_shower = file.mc_trks[:, 1]
                second_shower = file.mc_trks[:, 2]

            energy_imbalance = (
                np.array(first_shower.E) - np.array(second_shower.E)
            ) / (np.array(first_shower.E) + np.array(second_shower.E))
            distance = np.sqrt(
                (np.array(primaries.pos_x) - np.array(second_shower.pos_x))
                ** 2
                + (np.array(primaries.pos_y) - np.array(second_shower.pos_y))
                ** 2
                + (np.array(primaries.pos_z) - np.array(second_shower.pos_z))
                ** 2
            )

            # compute the angle between the two showers
            angle = np.arccos(
                np.array(first_shower.dir_x) * np.array(second_shower.dir_x)
                + np.array(first_shower.dir_y) * np.array(second_shower.dir_y)
                + np.array(first_shower.dir_z) * np.array(second_shower.dir_z)
            )

            # the zenith and azimuth of the heavy neutrino
            zen_hnl, az_hnl = xyz_dir_to_zen_az(
                np.array(hnl.dir_x),
                np.array(hnl.dir_y),
                np.array(hnl.dir_z),
                padding_value,
            )

            return {
                "zenith_hnl": zen_hnl,
                "azimuth_hnl": az_hnl,
                "angle_between_showers": angle,
                "Energy_hnl": np.array(hnl.E),
                "Energy_second_shower": np.array(second_shower.E),
                "Energy_imbalance": energy_imbalance,
                "distance": distance,
                "is_hnl": np.ones(len(primaries.pos_x)),
            }
        elif (file_type == "neutrino") or (file_type == "muon"):
            return {
                "zenith_hnl": padding_value * np.ones(len(primaries.pos_x)),
                "azimuth_hnl": padding_value * np.ones(len(primaries.pos_x)),
                "angle_between_showers": np.zeros(len(primaries.pos_x)),
                "Energy_hnl": padding_value * np.ones(len(primaries.pos_x)),
                "Energy_second_shower": padding_value
                * np.ones(len(primaries.pos_x)),
                "Energy_imbalance": padding_value
                * np.ones(len(primaries.pos_x)),
                "distance": np.zeros(len(primaries.pos_x)),
                "is_hnl": np.zeros(len(primaries.pos_x)),
            }
        else:
            raise ValueError("File type not recognized")

    def _construct_truth_dictionary(
        self,
        primaries: Any,
        primaries_jshower: Any,
        primaries_jmuon: Any,
        file: Any,
        padding_value: int,
        file_type: str,
    ) -> Dict[str, Any]:
        true_attrs = [
            "pdgid",
            "E",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
        ]
        dict_truth = self._extract_particle_attributes(
            primaries,
            true_attrs,
            padding_value,
            prefix="true",
            file_type=file_type,
        )

        primaries_jshower, primaries_jmuon = ki.tools.best_jshower(
            file.trks
        ), ki.tools.best_jmuon(file.trks)
        jshower_attrs = [
            "E",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
        ]
        jmuon_attrs = [
            "E",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
        ]
        if self.add_reco_info:
            jshower_data = self._extract_particle_attributes(
                primaries_jshower,
                jshower_attrs,
                padding_value,
                prefix="jshower",
                file_type=file_type,
            )
            jmuon_data = self._extract_particle_attributes(
                primaries_jmuon,
                jmuon_attrs,
                padding_value,
                prefix="jmuon",
                file_type=file_type,
            )
            dict_truth.update(jshower_data)
            dict_truth.update(jmuon_data)

        evt_data = self._extract_event_attributes(
            file,
            primaries,
            primaries_jshower,
            padding_value,
            file_type=file_type,
        )
        dict_truth.update(evt_data)

        if self.add_hnl_info:
            hnl_data = self._extract_hnl_attributes(
                file,
                primaries,
                primaries_jshower,
                padding_value,
                file_type=file_type,
            )
            dict_truth.update(hnl_data)

        return dict_truth

    def _extract_truth_dataframe(self, file: Any) -> Any:
        """Extract truth information from a file and returns a dataframe.

        Args:
            file (Any): The file from which to extract truth information.

        Returns:
            pd.DataFrame: A dataframe containing truth information.
        """
        nus_flavor = [12, 14, 16]
        padding_value = int(99999999)
        if len(file.mc_trks.E[0]) > 0:
            primaries = file.mc_trks[:, 0]
            ##############################################################
            # MUON-FILE####################################################
            ##############################################################
            if abs(np.array(primaries.pdgid)[0]) not in nus_flavor:
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)
                dict_truth = self._construct_truth_dictionary(
                    primaries,
                    primaries_jshower,
                    primaries_jmuon,
                    file,
                    padding_value,
                    file_type="muon",
                )

            ###############################################################
            # HNL-FILE######################################################
            ###############################################################
            elif 5914 in file.mc_trks.pdgid[0]:
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)
                dict_truth = self._construct_truth_dictionary(
                    primaries,
                    primaries_jshower,
                    primaries_jmuon,
                    file,
                    padding_value,
                    file_type="hnl",
                )

            elif (abs(np.array(primaries.pdgid)[0]) in nus_flavor) and (
                5914 not in file.mc_trks.pdgid[0]
            ):
                ####################################################
                # NEUTRINO-FILE######################################
                ####################################################
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)
                dict_truth = self._construct_truth_dictionary(
                    primaries,
                    primaries_jshower,
                    primaries_jmuon,
                    file,
                    padding_value,
                    file_type="neutrino",
                )
            else:
                ValueError("Not a neutrino, muon, hnl, noise or data file.")
        elif len(file.mc_trks.E[0]) == 0:
            if file.header["calibration"] == "dynamical":
                ####################################################
                # DATA-FILE##########################################
                ####################################################
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)
                dict_truth = self._construct_truth_dictionary(
                    primaries_jshower,
                    primaries_jshower,
                    primaries_jmuon,
                    file,
                    padding_value,
                    file_type="data",
                )

            elif file.header["calibration"] == "statical":
                ####################################################
                # NOISE-FILE##########################################
                ####################################################
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)
                dict_truth = self._construct_truth_dictionary(
                    primaries_jshower,
                    primaries_jshower,
                    primaries_jmuon,
                    file,
                    padding_value,
                    file_type="noise",
                )
            else:
                ValueError("Not a neutrino, muon, hnl, noise or data file.")
        else:
            ValueError("Not a neutrino, muon, hnl, noise or data file.")

        # Ensure all arrays in dict_truth are 1-dimensional
        for key, value in dict_truth.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                print(key)
                dict_truth[key] = value.flatten()
        truth_df = pd.DataFrame(dict_truth)

        return truth_df


class KM3NeTRegularTruthExtractor(KM3NeTTruthExtractor):
    """Class for extracting the truth regular true information from file."""

    def __init__(self, name: str = "truth"):
        """Initialize the class to extract the truth regular information."""
        super().__init__(name, add_hnl_info=False)


class KM3NeTHNLTruthExtractor(KM3NeTTruthExtractor):
    """Class for extracting the truth hnl true information from file."""

    def __init__(self, name: str = "truth_hnl"):
        """Initialize the class to extract the truth hnl information."""
        super().__init__(name, add_hnl_info=True)


class KM3NeTRegularRecoExtractor(KM3NeTTruthExtractor):
    """Class for extracting the truth regular reco information from file."""

    def __init__(self, name: str = "reco"):
        """Initialize the class."""
        super().__init__(name, add_hnl_info=False, add_reco_info=True)


class KM3NeTHNLRecoExtractor(KM3NeTTruthExtractor):
    """Class for extracting the truth hnl reco information from file."""

    def __init__(self, name: str = "reco_hnl"):
        """Initialize the class to extract the truth hnl reco information."""
        super().__init__(name, add_hnl_info=True, add_reco_info=True)
