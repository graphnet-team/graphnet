"""IceCube I3InferenceModule.

Contains functionality for writing model predictions to i3 files.
"""

from typing import List, Union, TYPE_CHECKING, Dict, Any, Tuple

import numpy as np

from .inference_module import I3InferenceModule
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube import (
    I3FeatureExtractor,
    I3FeatureExtractorIceCubeUpgrade,
)

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3MapKeyVectorDouble,
    )  # pyright: reportMissingImports=false
    from icecube import dataclasses, dataio


class I3PulseCleanerModule(I3InferenceModule):
    """A specialized module for pulse cleaning.

    It is assumed that the model provided has been trained for this.
    """

    def __init__(
        self,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model_config: Union[ModelConfig, str],
        state_dict: str,
        model_name: str,
        *,
        gcd_file: str,
        threshold: float = 0.7,
        discard_empty_events: bool = False,
    ):
        """General class for inference on I3Frames (physics).

        Arguments:
            pulsemap: the pulsmap that the model is expecting as input
                     (the one that is being cleaned).
            features: the features of the pulsemap that the model is expecting.
            pulsemap_extractor: The extractor used to extract the pulsemap.
            model_config: The ModelConfig (or path to it) that summarizes the
                            model used for inference.
            state_dict: Path to state_dict containing the learned weights.
            model_name: The name used for the model. Will help define the named
                        entry in the I3Frame. E.g. "dynedge".
            gcd_file: path to associated gcd file.
            threshold: the threshold for being considered a positive case.
                        E.g., predictions >= threshold will be considered
                        to be signal, all else noise.
            discard_empty_events: When true, this flag will eliminate events
                            whose cleaned pulse series are empty. Can be used
                            to speed up processing especially for noise
                            simulation, since it will not do any writing or
                            further calculations.
        """
        super().__init__(
            pulsemap=pulsemap,
            features=features,
            pulsemap_extractor=pulsemap_extractor,
            model_config=model_config,
            state_dict=state_dict,
            model_name=model_name,
            gcd_file=gcd_file,
        )
        self._threshold = threshold
        self._predictions_key = f"{pulsemap}_{model_name}_Predictions"
        self._total_pulsemap_name = f"{pulsemap}_{model_name}_Pulses"
        self._discard_empty_events = discard_empty_events

    def __call__(self, frame: I3Frame) -> bool:
        """Add a cleaned pulsemap to frame."""
        # inference
        gcd_file = self._gcd_file
        data = self._create_data_representation(frame)
        if data is None:  # If there is no pulses to clean
            return False
        predictions = self._inference(data)[0]

        if self._discard_empty_events:
            if sum(predictions > self._threshold) == 0:
                return False

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        assert predictions.shape[1] == 1

        del data  # memory
        # Build Dictionary of predictions
        data_dict = {}

        predictions_map = self._construct_prediction_map(
            frame=frame, predictions=predictions
        )

        # Adds the raw predictions to dictionary
        if self._predictions_key not in frame.keys():
            data_dict[self._predictions_key] = predictions_map

        # Create a pulse map mask, indicating the pulses that are over
        # threshold (e.g. identified as signal) and therefore should be kept
        # Using a lambda function to evaluate which pulses to keep by
        # checking the prediction for each pulse
        # (Adds the actual pulsemap to dictionary)
        if self._total_pulsemap_name not in frame.keys():
            data_dict[self._total_pulsemap_name] = (
                dataclasses.I3RecoPulseSeriesMapMask(
                    frame,
                    self._pulsemap,
                    lambda om_key, index, pulse: predictions_map[om_key][index]
                    >= self._threshold,
                )
            )

        # Submit predictions and general pulsemap
        frame = self._add_to_frame(frame=frame, data=data_dict)
        data = {}
        # Adds an additional pulsemap for each DOM type
        if isinstance(
            self._i3_extractors[0], I3FeatureExtractorIceCubeUpgrade
        ):
            mDOMMap, DEggMap, IceCubeMap = self._split_pulsemap_in_dom_types(
                frame=frame, gcd_file=gcd_file
            )

            if f"{self._total_pulsemap_name}_mDOMs_Only" not in frame.keys():
                data[f"{self._total_pulsemap_name}_mDOMs_Only"] = (
                    dataclasses.I3RecoPulseSeriesMap(mDOMMap)
                )

            if f"{self._total_pulsemap_name}_dEggs_Only" not in frame.keys():
                data[f"{self._total_pulsemap_name}_dEggs_Only"] = (
                    dataclasses.I3RecoPulseSeriesMap(DEggMap)
                )

            if f"{self._total_pulsemap_name}_pDOMs_Only" not in frame.keys():
                data[f"{self._total_pulsemap_name}_pDOMs_Only"] = (
                    dataclasses.I3RecoPulseSeriesMap(IceCubeMap)
                )

        # Submits the additional pulsemaps to the frame
        frame = self._add_to_frame(frame=frame, data=data)

        return True

    def _split_pulsemap_in_dom_types(
        self, frame: I3Frame, gcd_file: Any
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
        """Will split the cleaned pulsemap into multiple pulsemaps.

        Arguments:
            frame: I3Frame (physics)
            gcd_file: path to associated gcd file

        Returns:
            mDOMMap, DeGGMap, IceCubeMap
        """
        g = dataio.I3File(gcd_file)
        gFrame = g.pop_frame()
        while "I3Geometry" not in gFrame.keys():
            gFrame = g.pop_frame()
        omGeoMap = gFrame["I3Geometry"].omgeo

        mDOMMap, DEggMap, IceCubeMap = {}, {}, {}
        pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, self._total_pulsemap_name
        )
        for P in pulses:
            om = omGeoMap[P[0]]
            if om.omtype == 130:  # "mDOM"
                mDOMMap[P[0]] = P[1]
            elif om.omtype == 120:  # "DEgg"
                DEggMap[P[0]] = P[1]
            elif om.omtype == 20:  # "IceCube / pDOM"
                IceCubeMap[P[0]] = P[1]
        return mDOMMap, DEggMap, IceCubeMap

    def _construct_prediction_map(
        self, frame: I3Frame, predictions: np.ndarray
    ) -> I3MapKeyVectorDouble:
        """Make a pulsemap from predictions (for all OM types).

        Arguments:
            frame: I3Frame (physics)
            predictions: predictions from Model.

        Returns:
            predictions_map: a pulsemap from predictions
        """
        pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, self._pulsemap
        )

        idx = 0
        predictions = predictions.squeeze(1)
        predictions_map = dataclasses.I3MapKeyVectorDouble()
        for om_key, pulses in pulsemap.items():
            num_pulses = len(pulses)
            predictions_map[om_key] = predictions[
                idx : idx + num_pulses  # noqa: E203
            ].tolist()
            idx += num_pulses

        # Checks
        assert idx == len(
            predictions
        ), """Not all predictions were mapped to pulses,\n
            validation of predictions have failed."""

        assert (
            pulsemap.keys() == predictions_map.keys()
        ), """Input pulse map and predictions map do \n
              not contain exactly the same OMs"""
        return predictions_map
