"""Unit tests for I3Extractor class.
"""

from gnn_reco.data.i3extractor import I3Extractor

# @TODO: Need to bundle the package with a dummy/test I3-file to allow for self-contained testing.

def test_constructor():
    """Test that the default constructor works"""
    extractor = I3Extractor()
    assert extractor is not None
