"""Unit tests for I3Extractor class.
"""

from gnn_reco.data.utils import is_i3_file


def test_is_i3_file():
    assert is_i3_file("path/to/file.i3.bz2") is True
    assert is_i3_file("path/to/file.bz2") is False
    assert is_i3_file("path/to/GCD_file.i3.gz") is False
    assert is_i3_file("path/to/gcd_file.i3.zst") is False
    assert is_i3_file("path/to/GEO_file.i3.zst") is False
    assert is_i3_file("path/to/geo_file.i3.gz") is False
