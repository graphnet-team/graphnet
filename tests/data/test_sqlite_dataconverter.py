"""Unit tests for SQLiteDataConverter."""

from graphnet.data.sqlite_dataconverter import is_pulsemap_check


def test_is_pulsemap_check():
    assert is_pulsemap_check("SplitInIcePulses") is True
    assert is_pulsemap_check("SRTInIcePulses") is True
    assert is_pulsemap_check("InIceDSTPulses") is True
    assert is_pulsemap_check("RTTWOfflinePulses") is True
    assert is_pulsemap_check("truth") is False
    assert is_pulsemap_check("retro") is False
