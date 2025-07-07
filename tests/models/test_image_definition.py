from graphnet.models.data_representation import IC86Image
from graphnet.models.data_representation import NodesAsPulses
from graphnet.models.detector import IceCube86
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from graphnet.constants import (
    IC86_CNN_MAPPING,
    TEST_IC86MAIN_IMAGE,
    TEST_IC86UPPERDC_IMAGE,
    TEST_IC86LOWERDC_IMAGE,
)


def test_image_definition() -> None:
    """Test the ImageDefinition class for IC86 DNN data."""
    # Define input feature names

    grid = pd.read_parquet(IC86_CNN_MAPPING)
    grid = grid.loc[:, ["string", "dom_number"]]
    grid["redundant_string"] = grid["string"].copy()
    grid["redundant_dom_number"] = grid["dom_number"].copy()
    dtype = torch.float32

    # Create a NodeDefinition instance
    node_def = NodesAsPulses(
        input_feature_names=grid.columns.tolist(),
    )

    detector = IceCube86(replace_with_identity=grid.columns.tolist())

    # Create an instance of TestImageIC86
    image_definition = IC86Image(
        node_definition=node_def,
        input_feature_names=grid.columns.tolist(),
        include_lower_dc=True,
        include_upper_dc=True,
        string_label="string",
        dom_number_label="dom_number",
        dtype=dtype,
        detector=detector,
    )

    assert (
        image_definition.nb_outputs == 2
    ), "Expected 2 outputs, got {}".format(image_definition.nb_outputs)

    output_feature_names = grid.columns.tolist()
    output_feature_names.remove("string")
    output_feature_names.remove("dom_number")

    assert image_definition.output_feature_names == output_feature_names, (
        f"Output feature names do not match expected output: "
        f"{image_definition.output_feature_names} != {output_feature_names}"
    )

    image = image_definition(
        grid.values,
        input_feature_names=grid.columns.tolist(),
    )

    assert isinstance(
        image, Data
    ), "Expected output to be a torch_geometric.data.Data object"
    assert isinstance(image.x, list), "Expected image.x to be a list"
    assert np.all(
        [isinstance(x, torch.Tensor) for x in image.x]
    ), "Expected all elements in image.x to be torch.Tensor"
    assert (
        len(image.x) == 3
    ), "Expected image.x to have 3 elements, got {}".format(len(image.x))
    assert (
        "num_nodes" in image.keys()
    ), "Expected 'num_nodes' in image attributes"

    image_list = [
        TEST_IC86MAIN_IMAGE,
        TEST_IC86UPPERDC_IMAGE,
        TEST_IC86LOWERDC_IMAGE,
    ]
    for i, img in enumerate(image_list):
        expected_image = torch.tensor(np.load(img), dtype=dtype).unsqueeze(0)
        assert image.x[i].size() == expected_image.size(), (
            f"Image at index {i} size mismatch: "
            f"expected {torch.tensor(expected_image).size()},"
            f"got {image.x[i].size()}"
        )
        assert torch.equal(
            image.x[i], expected_image
        ), f"Image at index {i} does not match expected image"
