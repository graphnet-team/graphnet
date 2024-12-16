"""Convex hull object for IceCube geometry."""

from scipy.spatial import ConvexHull
from icecube import dataclasses, MuonGun

import numpy as np


class GCD_hull(ConvexHull):
    """Convex hull object for IceCube geometry."""

    def __init__(self, gcd_file: str, padding: float = 0.0):
        """Initialize the ConvexHull object from the GCD file."""
        # Member variable(s)
        # IceCube surface object
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
        self, point: dataclasses.I3Position, tolerance: float = 1e-12
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
