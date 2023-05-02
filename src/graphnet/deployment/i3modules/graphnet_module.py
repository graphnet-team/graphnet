"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Union, Dict, Tuple

import dill
import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.data.extractors import (
    I3FeatureExtractor,
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.models import Model, StandardModel
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Module,
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Double,
        I3MapKeyVectorDouble,
    )  # pyright: reportMissingImports=false
    from icecube import dataclasses, dataio, icetray


class GraphNeTI3Module:
    """Base I3 Module for GraphNeT.

    Contains methods for extracting pulsemaps, producing graphs and writing to
    frames.
    """

    def __init__(
        self,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        gcd_file: str,
    ):
        """I3Module Constructor.

        Arguments:
            pulsemap: the pulse map on which the module functions
            features: the features that is used from the pulse map.
                      E.g. [dom_x, dom_y, dom_z, charge]
            pulsemap_extractor: The I3FeatureExtractor used to extract the
                                pulsemap from the I3Frames
            gcd_file: Path to the associated gcd-file.
        """
        self._pulsemap = pulsemap
        self._features = features
        assert isinstance(gcd_file, str), "gcd_file must be string"
        self._gcd_file = gcd_file
        if isinstance(pulsemap_extractor, list):
            self._i3_extractors = pulsemap_extractor
        else:
            self._i3_extractors = [pulsemap_extractor]

        for i3_extractor in self._i3_extractors:
            i3_extractor.set_files(i3_file="", gcd_file=self._gcd_file)

    @abstractmethod
    def __call__(self, frame: I3Frame) -> bool:
        """Define here how the module acts on the frame.

        Must return True if successful.

        Return True # SUPER IMPORTANT
        """

    def _make_graph(
        self, frame: I3Frame
    ) -> Data:  # py-l-i-n-t-:- -d-i-s-able=invalid-name
        """Process Physics I3Frame into graph."""
        # Extract features
        features = self._extract_feature_array_from_frame(frame)

        # Prepare graph data
        n_pulses = torch.tensor([features.shape[0]], dtype=torch.int32)
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            batch=torch.zeros(
                features.shape[0], dtype=torch.int64
            ),  # @TODO: Necessary?
            features=self._features,
        )
        # @TODO: This sort of hard-coding is not ideal; all features should be
        #        captured by `FEATURES` and included in the output of
        #        `I3FeatureExtractor`.
        data.n_pulses = n_pulses
        return data

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        """Apply the I3FeatureExtractors to the I3Frame.

        Arguments:
            frame: Physics I3Frame (PFrame)

        Returns:
            array with pulsemap
        """
        features = None
        for i3extractor in self._i3_extractors:
            feature_dict = i3extractor(frame)
            features_pulsemap = np.array(
                [feature_dict[key] for key in self._features]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _add_to_frame(self, frame: I3Frame, data: Dict[str, Any]) -> I3Frame:
        """Add every field in data to I3Frame.

        Arguments:
            frame: I3Frame (physics)
            data: Dictionary containing content that will be written to frame.

        Returns:
            frame: Same I3Frame as input, but with the new entries
        """
        assert isinstance(
            data, dict
        ), f"data must be of type dict. Got {type(data)}"
        for key in data.keys():
            if key not in frame:
                frame.Put(key, data[key])
        return frame


class I3InferenceModule(GraphNeTI3Module):
    """General class for inference on i3 frames."""

    def __init__(
        self,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model: Union[Model, StandardModel, str],
        model_name: str,
        prediction_columns: Union[List[str], str],
        gcd_file: str,
    ):
        """General class for inference on I3Frames (physics).

        Arguments:
            pulsemap: the pulsmap that the model is expecting as input.
            features: the features of the pulsemap that the model is expecting.
            pulsemap_extractor: The extractor used to extract the pulsemap.
            model: The model (or path to pickled model) that will be
                    used for inference.
            model_name: The name used for the model. Will help define the
                        named entry in the I3Frame. E.g. "dynedge".
            prediction_columns: column names for the predictions of the model.
                               Will help define the named entry in the I3Frame.
                                E.g. ['energy_reco'].
            gcd_file: path to associated gcd file.
        """
        super().__init__(
            pulsemap=pulsemap,
            features=features,
            pulsemap_extractor=pulsemap_extractor,
            gcd_file=gcd_file,
        )

        if isinstance(model, str):
            self.model = torch.load(
                model, pickle_module=dill, map_location="cpu"
            )
        else:
            self.model = model
        self.model.inference()

        self.model.to("cpu")

        if isinstance(prediction_columns, str):
            self.prediction_columns = [prediction_columns]
        else:
            self.prediction_columns = prediction_columns

        self.model_name = model_name

    def __call__(self, frame: I3Frame) -> bool:
        """Write predictions from model to frame."""
        # inference
        graph = self._make_graph(frame)
        if len(graph.x) > 0:
            predictions = self._inference(graph)
        else:
            predictions = np.repeat(
                [np.nan], len(self.prediction_columns)
            ).reshape(-1, len(self.prediction_columns))

        # Check dimensions of predictions and prediction columns
        if len(predictions.shape) > 1:
            dim = predictions.shape[1]
        else:
            dim = len(predictions)
        assert dim == len(
            self.prediction_columns
        ), f"""predictions have shape {dim} but \n
            prediction columns have [{self.prediction_columns}]"""

        # Build Dictionary of predictions
        data = {}
        assert predictions.shape[0] == 1
        for i in range(dim if isinstance(dim, int) else len(dim)):
            # print(predictions)
            try:
                assert len(predictions[:, i]) == 1
                data[
                    self.model_name + "_" + self.prediction_columns[i]
                ] = I3Double(float(predictions[:, i][0]))
            except IndexError:
                data[
                    self.model_name + "_" + self.prediction_columns[i]
                ] = I3Double(predictions[0])

        # Submission methods
        frame = self._add_to_frame(frame=frame, data=data)
        return True

    def _inference(self, data: Data) -> np.ndarray:
        # Perform inference
        task_predictions = self.model(data)
        assert (
            len(task_predictions) == 1
        ), f"""This method assumes a single task. \n
               Got {len(task_predictions)} tasks."""
        return self.model(data)[0].detach().numpy()


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
        model: Union[Model, StandardModel, str],
        model_name: str,
        prediction_columns: Union[List[str], str] = "",
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
            model: The model (or path to pickled model) that will be
                    used for inference.
            model_name: The name used for the model. Will help define the named
                        entry in the I3Frame. E.g. "dynedge".
            prediction_columns: column names for the predictions of the model.
                            Will help define the named entry in the I3Frame.
                            E.g. ['energy_reco'].
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
            model=model,
            model_name=model_name,
            prediction_columns=prediction_columns,
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
        graph = self._make_graph(frame)
        predictions = self._inference(graph)

        if self._discard_empty_events:
            if sum(predictions > self._threshold) == 0:
                return False

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        assert predictions.shape[1] == 1

        # Build Dictionary of predictions
        data = {}

        predictions_map = self._construct_prediction_map(
            frame=frame, predictions=predictions
        )

        # Adds the raw predictions to dictionary
        if self._predictions_key not in frame.keys():
            data[self._predictions_key] = predictions_map

        # Create a pulse map mask, indicating the pulses that are over
        # threshold (e.g. identified as signal) and therefore should be kept
        # Using a lambda function to evaluate which pulses to keep by
        # checking the prediction for each pulse
        # (Adds the actual pulsemap to dictionary)
        if self._total_pulsemap_name not in frame.keys():
            data[
                self._total_pulsemap_name
            ] = dataclasses.I3RecoPulseSeriesMapMask(
                frame,
                self._pulsemap,
                lambda om_key, index, pulse: predictions_map[om_key][index]
                >= self._threshold,
            )

        # Submit predictions and general pulsemap
        frame = self._add_to_frame(frame=frame, data=data)
        data = {}
        # Adds an additional pulsemap for each DOM type
        if isinstance(
            self._i3_extractors[0], I3FeatureExtractorIceCubeUpgrade
        ):
            mDOMMap, DEggMap, IceCubeMap = self._split_pulsemap_in_dom_types(
                frame=frame, gcd_file=gcd_file
            )

            if f"{self._total_pulsemap_name}_mDOMs_Only" not in frame.keys():
                data[
                    f"{self._total_pulsemap_name}_mDOMs_Only"
                ] = dataclasses.I3RecoPulseSeriesMap(mDOMMap)

            if f"{self._total_pulsemap_name}_dEggs_Only" not in frame.keys():
                data[
                    f"{self._total_pulsemap_name}_dEggs_Only"
                ] = dataclasses.I3RecoPulseSeriesMap(DEggMap)

            if f"{self._total_pulsemap_name}_pDOMs_Only" not in frame.keys():
                data[
                    f"{self._total_pulsemap_name}_pDOMs_Only"
                ] = dataclasses.I3RecoPulseSeriesMap(IceCubeMap)

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
            predictions: predictions from GNN

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
                idx : idx + num_pulses
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
