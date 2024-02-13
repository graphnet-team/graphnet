
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import awkward as ak

from graphnet.data.extractors.rootextractor import rootExtractor

class rootFeatureExtractor(rootExtractor):
    """Base class for extracting specific, reconstructed features."""

    def __init__(self, name: str, branch_key: str, feature_keys: List[str]):
        """Construct rootFeatureExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
        """
        # Member variable(s)
        self._branch_key = branch_key
        self._feature_keys = feature_keys
        self._index_coulumn: str = ""

        # Base class constructor
        super().__init__(name)

    def set_index_column(self, index_column: str):
        """Called by the dataconverter to set the name of the index column"""
        self._index_coulumn = index_column


class rootFeatureExtractorESSnuSB(rootFeatureExtractor):
    """Class for extracting reconstructed features for ESSnuSB."""

    def __init__(self, name: str, branch_key: str):

        self._var_dict = {None: None}

        feature_keys = [
            'fX', 
            'fY', 
            'fZ', 
            'fTime', 
            'fCharge',
        ]

        # Base class constructor
        super().__init__(name, branch_key, feature_keys)

    def __call__(self, events: "root.events", index: int) -> pd.DataFrame:
        """Extract reconstructed features from `frame`.

        Args:
            events: root events from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """

        hits = events[self._branch_key][self._branch_key+'.'+self._feature_keys[0]].array(library='ak')
        indexes = index + np.arange(len(hits))

        repeat_indexes = np.repeat(
            indexes, 
            ak.count(hits, axis=1)
        )

        hit_data = pd.DataFrame(repeat_indexes, columns=[self._index_coulumn])

        for feature_key in self._feature_keys:
            hit_data[self._var_dict.get(
                feature_key, feature_key
            )] = ak.flatten(
                events[self._branch_key][self._branch_key+'.'+feature_key].array(library='ak')
            )

        return hit_data
    
class rootTruthExtractorESSnuSB(rootFeatureExtractor):
    """Class for extracting reconstructed features for ESSnuSB."""

    def __init__(self, name: str, branch_key: str):

        self._var_dict = {
            'fType': 'interaction_type',
            'fNpdg': 'pid',
            'fLpdg': 'Lpid',
        }

        feature_keys = [
            'fType',
            'fSign',
            'fVx', 
            'fVy', 
            'fVz', 
            'fNpdg', 
            'fNE', 
            'fNdx', 
            'fNdy', 
            'fNdz', 
            'fLpdg',
            'fLE', 
            'fLdx', 
            'fLdy', 
            'fLdz', 
            'fNphotons', 
            'fNpions',
        ]

        # Base class constructor
        super().__init__(name, branch_key, feature_keys)

    def __call__(self, events: "root.events", index: int) -> pd.DataFrame:
        """Extract reconstructed features from `frame`.

        Args:
            events: root events from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """

        indexes = index + np.arange(len(
            events[self._branch_key][self._feature_keys[0]].array(library='ak')
        ))

        # Get truths
        truth_data = pd.DataFrame(indexes, columns=[self._index_coulumn])
        for feature_key in self._feature_keys:
            truth_data[self._var_dict.get(
                feature_key, feature_key
            )] = events[self._branch_key][feature_key].array(library='ak')

        return truth_data
     
class rootfiTQunExtractorESSnuSB(rootFeatureExtractor):
    """Class for extracting reconstructed features for ESSnuSB."""

    def __init__(self, name: str, branch_key: str):

        self._var_dict = {None: None}

        feature_keys = [
            'fqe_ekin',
            'fqmu_ekin',
            'fqpi_ekin',
            
            'fqe_x',
            'fqmu_x',
            'fqpi_x',
            'fqe_y',
            'fqmu_y',
            'fqpi_y',
            'fqe_z',
            'fqmu_z',
            'fqpi_z',

            'fqe_dx',
            'fqmu_dx',
            'fqpi_dx',
            'fqe_dy',
            'fqmu_dy',
            'fqpi_dy',
            'fqe_dz',
            'fqmu_dz',
            'fqpi_dz',

            'fqe_theta',
            'fqmu_theta',
            'fqpi_theta',

            'fqe_nll',
            'fqmu_nll',
            'fqpi_nll',

            'fq_q',
            'fqe_dw',
            'fqe_dwd',
            'fqmu_dw',
            'fqmu_dwd',
            'mc_e',
        ]

        # Base class constructor
        super().__init__(name, branch_key, feature_keys)

    def __call__(self, events: "root.events", index: int) -> pd.DataFrame:
        """Extract reconstructed features from `frame`.

        Args:
            events: root events from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """

        indexes = index + np.arange(len(
            events[self._branch_key][self._feature_keys[0]].array(library='ak')
        ))

        # Get truths
        fiTQun_data = pd.DataFrame(indexes, columns=[self._index_coulumn])
        for feature_key in self._feature_keys:
            fiTQun_data[self._var_dict.get(
                feature_key, feature_key
            )] = events[self._branch_key][feature_key].array(library='ak')

        return fiTQun_data