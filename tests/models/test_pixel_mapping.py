"""Unit tests for node definitions."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from copy import deepcopy
from graphnet.models.data_representation.images import IC86PixelMapping
from graphnet.constants import (
    TEST_IC86MAIN_IMAGE,
    IC86_CNN_MAPPING,
    TEST_IC86UPPERDC_IMAGE,
    TEST_IC86LOWERDC_IMAGE,
)
import pytest


def basic_checks_picture(picture: Data, dtype: torch.dtype) -> None:
    """Basic checks for the output of pixel mapping."""
    assert isinstance(
        picture, Data
    ), f"Output should be a Data object got {type(picture)}"
    assert isinstance(
        picture.x, list
    ), f"x should be a list of tensors got {type(picture.x)}"
    assert np.all(
        [isinstance(picture.x[i], torch.Tensor) for i in range(len(picture.x))]
    ), (
        "All tensors in x should be torch.Tensors",
        f"got {[type(picture.x[i]) for i in range(len(picture.x))]}",
    )
    assert np.all(
        [picture.x[i].dtype == dtype for i in range(len(picture.x))]
    ), (
        "All tensors in x should have the dtype specified in pixel_mapping",
        f"got {[picture.x[i].dtype for i in range(len(picture.x))]}",
    )


def test_pixel_mappings() -> None:
    """Test pixel mapping for IC86 DNN mapping."""
    # definitions
    dtype = torch.float32
    pixel_feature_names = ["string", "dom_number", "data1", "data2"]
    string_label = "string"
    dom_number_label = "dom_number"

    # Create dummy data
    dummy_data = Data(
        x=torch.tensor(
            [[1, 2, 5.8, 1e-4], [79, 46, 3.7, 1e-18], [84, 9, 6.87, 2e5]],
            dtype=dtype,
        ),
    )

    # Construct node definition
    # This defines each DOM as a cluster, and will summarize pulses seen by
    # DOMs using percentiles.
    pixel_mapping = IC86PixelMapping(
        dtype=dtype,
        pixel_feature_names=pixel_feature_names,
        string_label=string_label,
        dom_number_label=dom_number_label,
        include_lower_dc=True,
        include_upper_dc=True,
    )

    # Apply node definition to torch tensor with raw pulses
    picture = pixel_mapping(dummy_data, pixel_feature_names)
    new_features = pixel_mapping.image_feature_names
    n_features = len(new_features)

    # Check the output
    basic_checks_picture(picture, dtype)

    # More checks
    assert (
        len(pixel_mapping.shape) == 3
    ), f"Expected shape to be 3 got {len(pixel_mapping.shape)}"
    assert pixel_mapping.shape == [
        (n_features, 10, 10, 60),
        (n_features, 1, 8, 10),
        (n_features, 1, 8, 50),
    ], (
        f"Expected shape to be [({n_features},10,10,60), "
        f"({n_features},1,8,10), ({n_features},1,8,50)] got "
        f"{pixel_mapping.shape}"
    )
    assert isinstance(
        new_features, list
    ), f"Output should be a list of feature names got {type(new_features)}"
    assert new_features == [
        "data1",
        "data2",
    ], f"Expected feature to be ['data1', 'data2'] names got: {new_features}"
    assert len(picture.x) == 3, (
        "There should be three tensors in x ",
        f"got list with length {len(picture.x)}"
        "(main array, upper DeepCore, lower DeepCore)",
    )
    assert picture.x[0].size() == torch.Size(
        [1, 2, 10, 10, 60]
    ), f"Main array should have shape (1,2,10,10,60) got {picture.x[0].size()}"
    assert picture.x[1].size() == torch.Size(
        [1, 2, 8, 10]
    ), f"upper DeepCore should have shape (1,2,8,10) got {picture.x[1].size()}"
    assert picture.x[2].size() == torch.Size(
        [1, 2, 8, 50]
    ), f"lower DeepCore should have shape (1,2,8,50) got {picture.x[2].size()}"
    assert not torch.all(
        picture.x[0] == 0
    ), "Main array should not be all zeros, got all zeros."
    assert not torch.all(
        picture.x[1] == 0
    ), "Upper DeepCore should not be all zeros, got all zeros."
    assert not torch.all(
        picture.x[2] == 0
    ), "Lower DeepCore should not be all zeros, got all zeros."

    # Try string and dom_number that does not exist
    dummy_data = Data(
        x=torch.tensor(
            [
                [100, 5, 5.8, 1e-4],
                [54, 230, 3.7, 1e-18],
                [1294, 500, 6.87, 2e5],
            ],
            dtype=dtype,
        ),
    )

    # should raise KeyError since the string and dom_number
    # do not exist in the mapping
    with pytest.raises(KeyError):
        picture = pixel_mapping(dummy_data, pixel_feature_names)


def test_segments_mapping() -> None:
    """Test pixel mapping for IC86 main array."""
    # definitions
    dtype = torch.float32
    string_label = "string"
    dom_number_label = "dom_number"
    pixel_feature_names = [
        "string",
        "dom_number",
        "redundant_string",
        "redundant_dom_number",
    ]

    # Load the grid mapping
    # This is a mapping from string and dom_number to the pixel coordinates
    # in the main array, upper DeepCore and lower DeepCore.
    # Running the grid mapping through the pixel mapping will
    # create the full images for the main array, upper DeepCore
    # and lower DeepCore.
    grid = pd.read_parquet(IC86_CNN_MAPPING)
    grid = grid.loc[:, ["string", "dom_number"]]
    grid["redundant_string"] = grid["string"].copy()
    grid["redundant_dom_number"] = grid["dom_number"].copy()
    grid = Data(x=torch.tensor(grid.to_numpy(), dtype=dtype))

    # Test the pixel mapping for the main array, upper and lower DeepCore
    for image, inc_main, inc_upc, inc_lowdc, label in zip(
        [TEST_IC86MAIN_IMAGE, TEST_IC86UPPERDC_IMAGE, TEST_IC86LOWERDC_IMAGE],
        [True, False, False],
        [False, True, False],
        [False, False, True],
        ["main array", "upper deepcore", "lower deepcore"],
    ):
        tmp = deepcopy(grid)
        pixel_mapping = IC86PixelMapping(
            dtype=dtype,
            pixel_feature_names=pixel_feature_names,
            string_label=string_label,
            dom_number_label=dom_number_label,
            include_main_array=inc_main,
            include_lower_dc=inc_lowdc,
            include_upper_dc=inc_upc,
        )
        picture = pixel_mapping(tmp, pixel_feature_names)
        tensor_image: torch.tensor = torch.tensor(
            np.load(image), dtype=dtype
        ).unsqueeze(0)

        # Check the output
        basic_checks_picture(picture, dtype)

        # More checks
        assert len(picture.x) == 1, (
            "There should be one tensor in x ",
            f"got list with length {len(picture.x)}",
        )
        assert picture.x[0].size() == tensor_image.size(), (
            f"{label} should have shape {tensor_image.size()} "
            f"got {picture.x[0].size()}"
        )
        assert not torch.all(
            picture.x[0] == 0
        ), f"{label} should not be all zeros, got all zeros."
        # Check if the tensor matches the expected image
        assert torch.equal(tensor_image, picture.x[0]), (
            f"{label} should match the expected"
            " main array from IC86 DNN mapping."
        )
