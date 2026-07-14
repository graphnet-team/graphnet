"""Tests for the PrometheusReader."""

import os
from typing import List

import pytest

from graphnet.constants import TEST_DATA_DIR
from graphnet.data.constants import TRUTH
from graphnet.data.extractors.prometheus import (
    PrometheusExtractor,
    PrometheusFeatureExtractor,
    PrometheusTruthExtractor,
)
from graphnet.data.readers import PrometheusReader

FILE_PATH = os.path.join(
    TEST_DATA_DIR, "prometheus", "22980001_photons.parquet"
)


def test_prometheus_reader_extracts_configured_tables() -> None:
    """Reader extracts every configured table when all are present."""
    reader = PrometheusReader()
    reader.set_extractors(
        [PrometheusTruthExtractor(), PrometheusFeatureExtractor()]
    )
    events = reader(FILE_PATH)
    assert len(events) > 0
    assert set(events[0].keys()) == {"mc_truth", "photons"}
    assert set(TRUTH.PROMETHEUS) <= set(events[0]["mc_truth"].keys())


def test_prometheus_truth_extractor_columns_override() -> None:
    """Truth extractor extracts only the overridden columns."""
    reader = PrometheusReader()
    extractors: List[PrometheusExtractor] = [
        PrometheusTruthExtractor(columns=["initial_state_energy"])
    ]
    reader.set_extractors(extractors)
    events = reader(FILE_PATH)
    assert list(events[0]["mc_truth"].keys()) == ["initial_state_energy"]


def test_prometheus_reader_raises_on_missing_table() -> None:
    """Reader raises if an extractor's table is not in the file."""
    reader = PrometheusReader()
    reader.set_extractors(
        [
            PrometheusTruthExtractor(),
            PrometheusFeatureExtractor(table_name="not_a_table"),
        ]
    )
    with pytest.raises(ValueError, match="not_a_table"):
        reader(FILE_PATH)
