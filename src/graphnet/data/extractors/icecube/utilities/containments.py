"""Utility function cataloging the types of containment of an icecube event."""

import numpy as np
from enum import Enum, auto


class GN_containment_types(Enum):
    """Containment types for an IceCube event.

    See
    https://github.com/icecube/icetray/blob/d68d0e0be86bf2cfeba8cab1190f631f4454e4c4/sim-services/python/label_events/enums.py#L50
    For the original IceCube enum.
    """

    no_intersect = auto()
    throughgoing = auto()  # For tracks only
    contained = auto()  # contained starting event
    tau_to_mu = (
        auto()
    )  # Special case for a contained tau that decays into a muon
    starting = auto()  # uncontained starting track
    stopping = auto()  # stopping track event
    decayed = auto()
    throughgoing_bundle = auto()  # background track bundle
    stopping_bundle = auto()  # stopping background track bundle
    partly_contained = auto()  # Partly contained starting event


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
        return GN_containment_types.no_intersect.value

    if first <= 0 and second > 0:
        if length <= second:
            return GN_containment_types.contained.value
        return GN_containment_types.starting.value

    if first > 0 and second > 0:
        if length <= first:
            return GN_containment_types.no_intersect.value
        if length > second:
            return GN_containment_types.throughgoing.value
        else:
            return GN_containment_types.stopping.value
    return GN_containment_types.no_intersect.value
