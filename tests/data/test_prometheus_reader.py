"""Tests for the PrometheusReader."""

import os

import pytest

from graphnet.constants import TEST_DATA_DIR
from graphnet.data.extractors.prometheus import (
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
