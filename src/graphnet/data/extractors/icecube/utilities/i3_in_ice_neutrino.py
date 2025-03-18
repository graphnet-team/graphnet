"""Utility functions for extracting in-ice neutrino from ListI3Particle."""

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import (
        dataclasses,
    )  # pyright: reportMissingImports=false


def get_in_ice_neutrino(
    primaries: "dataclasses.ListI3Particle",
) -> "dataclasses.I3Particle":
    """Get the highest energy in ice neutrion.

    Args:
        primaries (dataclasses.ListI3Particle): List of primary particles
    """
    in_ice_neutrino = dataclasses.I3Particle()
    in_ice_neutrino.energy = 0
    for primary in primaries:
        if primary.is_neutrino and primary.location_type == 20:
            if primary.energy > in_ice_neutrino.energy:
                in_ice_neutrino = primary
    assert in_ice_neutrino.energy > 0, "No in-ice neutrino found"
    return in_ice_neutrino
