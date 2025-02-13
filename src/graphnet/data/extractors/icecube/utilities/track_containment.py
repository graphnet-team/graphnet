"""Function to determine the containment of a track."""

import numpy as np
from typing import TYPE_CHECKING
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.sim_services.label_events.enums import (
        containments_types,
    )  # pyright: reportMissingImports=false


def track_containment(first: float, second: float, length: float) -> int:
    """Determine the containment of a track.

    Inpsired by MCLabelerModule from IceCube.
    https://github.com/icecube/icetray/blob/main/sim-services/python/label_events/mc_labeler.py

    Args:
        first (float): First interaction with surface.
        second (float): Second interaction with surface.
        length (float): Length of the particle track.
    """
    if not np.isfinite(first):
        return containments_types.no_intersect.value

    if first <= 0 and second > 0:
        if length <= second:
            return containments_types.contained.value
        return containments_types.starting.value

    if first > 0 and second > 0:
        if length <= first:
            return containments_types.no_intersect.value
        if length > second:
            return containments_types.throughgoing.value
        else:
            return containments_types.stopping.value
    return containments_types.no_intersect.value
