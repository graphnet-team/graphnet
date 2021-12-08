"""Unit tests for I3Extractor class."""

from gnn_reco.data.i3extractor import I3FeatureExtractorIceCube86, I3TruthExtractor, I3RetroExtractor

# @TODO: Need to bundle the package with a dummy/test I3-file to allow for self-contained testing.

def test_featureextractor_constructor():
    """Test that the default constructor works"""
    extractor = I3FeatureExtractorIceCube86("pulsemap")
    assert extractor is not None

def test_truthextractor_constructor():
    """Test that the default constructor works"""
    extractor = I3TruthExtractor()
    assert extractor is not None

def test_retroextractor_constructor():
    """Test that the default constructor works"""
    extractor = I3RetroExtractor()
    assert extractor is not None
