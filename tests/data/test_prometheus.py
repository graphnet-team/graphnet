"""Unit tests for the Prometheus truth schema and reader dispatch."""

import os

import pandas as pd
import pytest

from graphnet.constants import EXAMPLE_DATA_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.extractors.prometheus import (
    PrometheusFeatureExtractor,
    PrometheusTruthExtractor,
)
from graphnet.data.readers import PrometheusReader
from graphnet.exceptions.exceptions import ColumnMissingException
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.data_representation import KNNGraph
from graphnet.models.data_representation.graphs.nodes import NodesAsPulses

DB_PATH = f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db"


def _graph_definition() -> KNNGraph:
    """Return a minimal graph definition for the Prometheus detector."""
    return KNNGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=FEATURES.PROMETHEUS,
    )


def test_truth_prometheus_matches_truth_extractor() -> None:
    """`TRUTH.PROMETHEUS` must stay in lockstep with the truth extractor."""
    assert TRUTH.PROMETHEUS == PrometheusTruthExtractor()._columns


def test_dataset_raises_when_all_truth_variables_missing() -> None:
    """Requesting a truth schema absent from the file should not be silent.

    The bundled example database was produced by an old Prometheus
    version, so it contains none of the `TRUTH.PROMETHEUS` columns.
    """
    with pytest.raises(ColumnMissingException):
        SQLiteDataset(
            path=DB_PATH,
            pulsemaps="total",
            features=FEATURES.PROMETHEUS,
            truth=TRUTH.PROMETHEUS,
            truth_table="mc_truth",
            graph_definition=_graph_definition(),
        )


def test_dataset_accepts_legacy_truth_schema() -> None:
    """`TRUTH.PROMETHEUS_LEGACY` matches the bundled example database."""
    dataset = SQLiteDataset(
        path=DB_PATH,
        pulsemaps="total",
        features=FEATURES.PROMETHEUS,
        truth=TRUTH.PROMETHEUS_LEGACY,
        truth_table="mc_truth",
        graph_definition=_graph_definition(),
    )
    assert len(dataset) > 0
    # No truth variables should have been silently removed
    # (the index column is prepended by the Dataset itself)
    assert dataset._truth == ["event_no"] + TRUTH.PROMETHEUS_LEGACY


def test_reader_warns_when_extractor_table_not_in_file(
    tmp_path: "os.PathLike", caplog: pytest.LogCaptureFixture
) -> None:
    """A configured extractor matching no table should emit a warning."""
    # File written with a non-default `photon_field_name`
    file_path = os.path.join(tmp_path, "events.parquet")
    pd.DataFrame(
        {
            "photons_custom": [{"sensor_pos_x": [0.0], "sensor_pos_y": [0.0]}],
            "mc_truth": [{"interaction": 1}],
        }
    ).to_parquet(file_path)

    reader = PrometheusReader()
    reader.set_extractors(
        [PrometheusTruthExtractor(), PrometheusFeatureExtractor()]
    )
    with caplog.at_level("WARNING"):
        outputs = reader(file_path)

    assert "photons" in caplog.text
    assert file_path in caplog.text
    # Truth is still extracted; the missing table is simply absent
    assert all("photons" not in event for event in outputs)
    assert all("mc_truth" in event for event in outputs)
