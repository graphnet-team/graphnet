"""Base model for defining data representations."""

from typing import Any, List, Optional, Dict, Callable, Union
import torch
from torch_geometric.data import Data
import numpy as np
from numpy.random import default_rng, Generator

from graphnet.models.detector import Detector
from graphnet.models import Model
from abc import abstractmethod


class DataRepresentation(Model):
    """An Abstract class to create data representations from."""

    def __init__(
        self,
        detector: Detector,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        add_inactive_sensors: bool = False,
        sensor_mask: Optional[List[int]] = None,
        string_mask: Optional[List[int]] = None,
        repeat_labels: bool = False,
    ):
        """Construct´DataRepresentation´. The ´detector´ holds.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            input_feature_names: Names of each column in expected input data
                that will be built into a the data. If not provided,
                it is automatically assumed that all features in `Detector` is
                used.
            dtype: data type used for features. e.g. ´torch.float´
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                  Defaults to None.
            add_inactive_sensors: If True, inactive sensors will be appended
                to the data with padded pulse information. Defaults to False.
            sensor_mask: A list of sensor id's to be masked from the data. Any
                sensor listed here will be removed from the data.
                    Defaults to None.
            string_mask: A list of string id's to be masked from the data.
                Defaults to None.
            repeat_labels: If True, labels will be repeated to match the
                the number of rows in the output of the GraphDefinition.
                Defaults to False.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member Variables
        self._detector = detector
        self._perturbation_dict = perturbation_dict
        self._sensor_mask = sensor_mask
        self._string_mask = string_mask
        self._add_inactive_sensors = add_inactive_sensors
        self._repeat_labels = repeat_labels

        self._resolve_masks()

        if input_feature_names is None:
            # Assume all features in Detector is used.
            input_feature_names = list(
                self._detector.feature_map().keys()
            )  # noqa: E501 # type: ignore
        self._input_feature_names = input_feature_names

        # Set data type
        self.to(dtype)

        # Set perturbation_cols if needed
        if isinstance(self._perturbation_dict, dict):
            self._perturbation_cols = [
                self._input_feature_names.index(key)
                for key in self._perturbation_dict.keys()
            ]
        if seed is not None:
            if isinstance(seed, int):
                self.rng = default_rng(seed)
            elif isinstance(seed, Generator):
                self.rng = seed
            else:
                raise ValueError(
                    "Invalid seed. Must be an int or a numpy Generator."
                )
        else:
            self.rng = default_rng()

    @property
    def nb_inputs(self) -> int:
        """Return the number of input features."""
        return len(self._input_feature_names)

    @property
    def nb_outputs(self) -> int:
        """Return the number of output features."""
        return len(self.output_feature_names)

    @property
    def output_feature_names(self) -> List[str]:
        """Initialize / return the names of output features."""
        if not hasattr(self, "_output_feature_names"):
            self._output_feature_names = self._set_output_feature_names(
                self._input_feature_names
            )
        return self._output_feature_names

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        """Construct data as ´Data´ object.

        Args:
            input_features: Input features for data construction.
                Shape ´[num_rows, d]´
            input_feature_names: name of each column. Shape ´[,d]´.
            truth_dicts: Dictionary containing truth labels.
            custom_label_functions: Custom label functions.
            loss_weight_column: Name of column that holds loss weight.
                                Defaults to None.
            loss_weight: Loss weight associated with event. Defaults to None.
            loss_weight_default_value: default value for loss weight.
                    Used in instances where some events have
                    no pre-defined loss weight. Defaults to None.
            data_path: Path to dataset data files. Defaults to None.

        Returns:
            data
        """
        # Checks
        self._validate_input(
            input_features=input_features,
            input_feature_names=input_feature_names,
        )

        # Add inactive sensors if `add_inactive_sensors = True`
        if self._add_inactive_sensors:
            input_features = self._attach_inactive_sensors(
                input_features, input_feature_names
            )

        # Mask out sensors if `sensor_mask` is given
        if self._sensor_mask is not None:
            input_features = self._mask_sensors(
                input_features, input_feature_names
            )

        # Gaussian perturbation of each column if perturbation dict is given
        input_features = self._perturb_input(input_features)

        # Transform to pytorch tensor
        input_features = torch.tensor(input_features, dtype=self.dtype)

        # Standardize / Scale final data features
        input_features = self._detector(input_features, input_feature_names)

        # Create data & get new final data feature names
        data, data_feature_names = self._create_data(
            input_features=input_features
        )

        # Attach number of pulses as static attribute.
        data.n_pulses = torch.tensor(len(input_features), dtype=torch.int32)

        # Attach data path - useful for Ensemble datasets.
        if data_path is not None:
            data["dataset_path"] = data_path

        # Attach loss weights if they exist
        data = self._add_loss_weights(
            data=data,
            loss_weight=loss_weight,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
        )

        # Attach default truth labels and node truths
        if truth_dicts is not None:
            data = self._add_truth(data=data, truth_dicts=truth_dicts)

        # Attach custom truth labels
        if custom_label_functions is not None:
            data = self._add_custom_labels(
                data=data, custom_label_functions=custom_label_functions
            )

        # DEPRECATION STAMP GRAPH_DEFINITION: REMOVE AT 2.0 LAUNCH
        # See https://github.com/graphnet-team/graphnet/issues/647
        data["graph_definition"] = self.__class__.__name__

        # Add data representation Stamp
        data["data_representation"] = self.__class__.__name__
        return data

    def _resolve_masks(self) -> None:
        """Handle cases with sensor/string masks."""
        if self._sensor_mask is not None:
            if self._string_mask is not None:
                assert (
                    1 == 2
                ), "Please specify only one of `sensor_mask`and `string_mask`."

        if (self._sensor_mask is None) & (self._string_mask is not None):
            self._sensor_mask = self._convert_string_to_sensor_mask()

        return

    def _convert_string_to_sensor_mask(self) -> List[int]:
        """Convert a string mask to a sensor mask."""
        string_id_column = self._detector.string_id_column
        sensor_id_column = self._detector.sensor_id_column
        geometry_table = self._detector.geometry_table
        idx = geometry_table[string_id_column].isin(self._string_mask)
        return np.asarray(geometry_table.loc[idx, sensor_id_column]).tolist()

    def _attach_inactive_sensors(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Attach inactive sensors to `input_features`.

        This function will query the detector geometry table and add any sensor
        in the geometry table that is not already present in queried data per
        event.
        """
        lookup = self._geometry_table_lookup(
            input_features, input_feature_names
        )
        geometry_table = self._detector.geometry_table
        unique_sensors = geometry_table.reset_index(drop=True)

        # multiple lines to avoid long line:
        inactive_idx = ~geometry_table.index.isin(lookup)
        inactive_sensors = unique_sensors.loc[
            inactive_idx, input_feature_names
        ]
        input_features = np.concatenate(
            [input_features, inactive_sensors.to_numpy()], axis=0
        )
        return input_features

    def _mask_sensors(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Mask sensors according to `sensor_mask`."""
        sensor_id_column = self._detector.sensor_index_name
        geometry_table = self._detector.geometry_table

        lookup = self._geometry_table_lookup(
            input_features=input_features,
            input_feature_names=input_feature_names,
        )
        mask = ~geometry_table.loc[lookup, sensor_id_column].isin(
            self._sensor_mask
        )

        return input_features[mask, :]

    def _geometry_table_lookup(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Convert xyz in `input_features` into a set of sensor ids."""
        lookup_columns = [
            input_feature_names.index(feature)
            for feature in self._detector.sensor_position_names
        ]
        idx = [*zip(*[tuple(input_features[:, k]) for k in lookup_columns])]
        return self._detector.geometry_table.loc[idx, :].index

    def _validate_input(
        self, input_features: np.array, input_feature_names: List[str]
    ) -> None:
        # raw data feature matrix dimension check
        assert input_features.shape[1] == len(input_feature_names)

        # check that provided features for input is the same that the
        # `DataRepresentation` was instantiated with.
        assert len(input_feature_names) == len(
            self._input_feature_names
        ), f"""Input features ({input_feature_names}) is not what 
               {self.__class__.__name__} was instatiated
               with ({self._input_feature_names})"""  # noqa
        for idx in range(len(input_feature_names)):
            assert (
                input_feature_names[idx] == self._input_feature_names[idx]
            ), f""" Order of representation features in data
                    are not the same as expected. Got {input_feature_names} 
                    vs. {self._input_feature_names}"""  # noqa

    def _perturb_input(self, input_features: np.ndarray) -> np.ndarray:
        if isinstance(self._perturbation_dict, dict):
            self.warning_once(
                f"""Will randomly perturb
                {list(self._perturbation_dict.keys())}
                using stds {self._perturbation_dict.values()}"""  # noqa
            )
            perturbed_features = self.rng.normal(
                loc=input_features[:, self._perturbation_cols],
                scale=np.array(
                    list(self._perturbation_dict.values()), dtype=float
                ),
            )
            input_features[:, self._perturbation_cols] = perturbed_features
        return input_features

    def _add_loss_weights(
        self,
        data: Data,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
    ) -> Data:
        """Attempt to store a loss weight in the data for use during training.

        I.e. `data[loss_weight_column] = loss_weight`

        Args:
            loss_weight: The non-negative weight to be stored.
            data: Data object representing the event.
            loss_weight_column: The name under which the weight is stored in
                                 the data.
            loss_weight_default_value: The default value used if
                                        none was retrieved.

        Returns:
            A data with loss weight added, if available.
        """
        # Add loss weight to data.
        if loss_weight is not None and loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current
            # event.
            if loss_weight < 0:
                if loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                data[loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self.dtype
                ).reshape(-1, 1)
            else:
                data[loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self.dtype
                ).reshape(-1, 1)
        return data

    def _add_custom_labels(
        self,
        data: Data,
        custom_label_functions: Dict[str, Callable[..., Any]],
    ) -> Data:
        # Add custom labels to the data
        for key, fn in custom_label_functions.items():
            label = fn(data)
            if self._repeat_labels:
                label = self._label_repeater(label, data)
            data[key] = label
        return data

    def _add_truth(
        self, data: Data, truth_dicts: List[Dict[str, Any]]
    ) -> Data:
        """Add truth labels from ´truth_dicts´ to ´data´.

        I.e. ´data[key] = truth_dict[key]´


        Args:
            data: data where the label will be stored
            truth_dicts: dictionary containing the labels

        Returns:
            data with labels
        """
        # Write attributes, either target labels, truth info or original
        # features.

        for truth_dict in truth_dicts:
            for key, value in truth_dict.items():
                try:
                    label = torch.tensor(value)
                    if self._repeat_labels:
                        label = self._label_repeater(label, data)
                    data[key] = label
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to data."
                        )
                    )
        return data

    @abstractmethod
    def _set_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Set the final data output feature names."""
        raise NotImplementedError

    @abstractmethod
    def _create_data(self, input_features: torch.Tensor) -> Data:
        """Create data from input features.

        Enforce the dtype of the feature tensor.
        E.g.: `data.x = data.x.type(self.dtype)`
        if the training data is stored in `data.x`.

        Should return:
            - data: torch_geometric.data.Data object representing the event.
            - data_feature_names: List of feature names in the data object.
        """
        raise NotImplementedError

    def _label_repeater(self, label: torch.Tensor, data: Data) -> torch.Tensor:
        """Handle the label repetition.

        Necessary only if the `repeat_labels` argument is being used
        E.g: for graphs: `label.repeat(data.x.shape[0], 1)`
        """
        raise NotImplementedError
