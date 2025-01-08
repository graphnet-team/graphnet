"""Convex hull object for IceCube geometry."""

from typing import TYPE_CHECKING

from scipy.spatial import ConvexHull

import numpy as np

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        MuonGun,
        dataclasses,
    )  # pyright: reportMissingImports=false


class GCD_hull(ConvexHull):
    """Convex hull object for IceCube geometry."""

    def __init__(self, gcd_file: str, padding: float = 0.0):
        """Initialize the ConvexHull object from the GCD file."""
        # Member variable(s)
        # IceCube surface object
        self.gcd_file = gcd_file
        self.padding = padding
        self.surface = MuonGun.ExtrudedPolygon.from_file(
            gcd_file, padding=padding
        )
        lower = np.array(
            [
                self.surface.x,
                self.surface.y,
                np.ones_like(self.surface.x) * self.surface.z[0],
            ]
        )
        upper = np.array(
            [
                self.surface.x,
                self.surface.y,
                np.ones_like(self.surface.x) * self.surface.z[1],
            ]
        )
        self.coords = np.concatenate([lower.T, upper.T])
        # Base class constructor
        super().__init__(self.coords)

    def point_in_hull(
        self, point: "dataclasses.I3Particle", tolerance: float = 1e-12
    ) -> bool:
        """Check if a point is inside the convex hull.

        Args:
        point: I3Position object
        tolerance: Tolerance for the dot product
        """
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in self.equations
        )

    def __getstate__(self) -> dict:
        """Return the state of the object for pickling."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if "surface" in state:
            del state["surface"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the object from the unpickled state."""
        self.__dict__.update(state)
        # Recreate the surface object if necessary.
        if "surface" not in self.__dict__:
            self.surface = MuonGun.ExtrudedPolygon.from_file(
                self.gcd_file, padding=self.padding
            )
