"""Module for extracting pulse information from a KM3NeT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import awkward as ak

from .km3netextractor import KM3NeTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id_run_by_run,
    assert_no_uint_values,
    creating_time_zero,
)


class KM3NeTPulseExtractor(KM3NeTExtractor):
    """Base class for extracting pulse information from a file."""

    def __init__(self, name: str, filter_triggered_pulses: bool = False):
        """Initialize the base extractor with optional pulse filtering."""
        super().__init__(name)
        self.filter_triggered_pulses = filter_triggered_pulses

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract pulse map information and return a dataframe.

        Args:
            file (Any): The file from which to extract pulse map information.

        Returns:
            Dict[str, Any]: A dictionary containing pulse map information.
        """
        pulsemap_df = self._extract_pulse_map(file)
        pulsemap_df = assert_no_uint_values(pulsemap_df)
        return pulsemap_df

    def _extract_pulse_map(self, file: Any) -> pd.DataFrame:
        """Extract the pulse information and assign unique IDs.

        Args:
            file (Any): The file from which to extract pulse information.

        Returns:
            pd.DataFrame: A dataframe containing pulse information.
        """
        # Process Monte Carlo or events/noise files
        unique_id = self._determine_unique_id(file)

        # Extract hits data
        hits = file.hits
        keys_to_extract = [
            "t",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            "tot",
            "trig",
        ]
        hits_array = hits.arrays(keys_to_extract, library="ak")
        hits_array["entry"] = ak.local_index(hits_array, axis=0)
        df = ak.to_dataframe(hits_array).reset_index(drop=True)
        
        # Add unique event ID
        unique_extended = [
            int(unique_id[index]) for index in df["entry"].values
        ]
        df["event_no"] = unique_extended

        # Optionally filter triggered pulses
        if self.filter_triggered_pulses:
            df = df[df["trig"] != 0]

        # Final processing
        df = df.drop(["entry", "subentry"], axis=1) if "subentry" in df.columns else df.drop(["entry"], axis=1)
        df = creating_time_zero(df)
        df = df.reset_index(drop=True)

        return df

    def _determine_unique_id(self, file: Any) -> np.ndarray:
        """Determine the unique ID for events."""
        padding_value = 99999999.0
        if len(file.mc_trks.E[0]) > 0:  # Monte Carlo
            primaries = file.mc_trks[:, 0]
            nus_flavor = [12, 14, 16]

            if (abs(np.array(primaries.pdgid)[0]) not in nus_flavor) and (abs(np.array(primaries.pdgid)[0]) > 0.1):  # Muon
                return create_unique_id_run_by_run(
                    file_type="muon",
                    run_id=np.array(file.run_id),
                    evt_id=np.array(file.id),
                    hnl_model="none",
                )
            elif (5914 in file.mc_trks.pdgid[0]) or (file.mc_trks.pdgid[0][0] == 0):  # HNL
                try:
                    model_hnl = file.header.model.interaction
                except Exception:
                    model_hnl = "none"
                return create_unique_id_run_by_run(
                        file_type="hnl",
                        run_id=np.array(file.run_id),
                        evt_id=np.array(file.id),
                        hnl_model=model_hnl,
                    )
            elif (abs(np.array(primaries.pdgid)[0]) in nus_flavor) and (
                5914 not in file.mc_trks.pdgid[0]
            ):  # Neutrino
                return create_unique_id_run_by_run(
                        file_type="neutrino",
                        run_id=np.array(file.run_id),
                        evt_id=np.array(file.id),
                        hnl_model="none",
                    )
        elif len(file.mc_trks.E[0]) == 0:  # Events or Noise
            if file.header["calibration"] == "dynamical":  # Data
                return create_unique_id_run_by_run(
                        file_type="data",
                        run_id=np.array(file.run_id),
                        evt_id=np.array(file.id),
                        hnl_model="none",
                    )
            elif file.header["calibration"] == "statical":  # Noise
                return create_unique_id_run_by_run(
                    file_type="noise",
                    run_id=np.array(file.run_id),
                    evt_id=np.array(file.id),
                    hnl_model="none",
                )
        raise ValueError("File type not recognized or corrupted.")


class KM3NeTTriggPulseExtractor(KM3NeTPulseExtractor):
    """Extractor for triggered pulses."""

    def __init__(self, name: str = "trigg_pulse_map"):
        """Initialize the extractor."""
        super().__init__(name, filter_triggered_pulses=True)


class KM3NeTFullPulseExtractor(KM3NeTPulseExtractor):
    """Extractor for all pulses (no filtering)."""

    def __init__(self, name: str = "full_pulse_map"):
        """Initialize the extractor."""
        super().__init__(name, filter_triggered_pulses=False)
