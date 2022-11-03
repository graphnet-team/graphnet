"""Unit tests for I3Extractor class."""

from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3RetroExtractor,
)


def test_featureextractor_constructor() -> None:
    """Test that the default constructor works."""
    extractor = I3FeatureExtractorIceCube86("pulsemap")
    assert extractor is not None


def test_truthextractor_constructor() -> None:
    """Test that the default constructor works."""
    extractor = I3TruthExtractor()
    assert extractor is not None


def test_retroextractor_constructor() -> None:
    """Test that the default constructor works."""
    extractor = I3RetroExtractor()
    assert extractor is not None
