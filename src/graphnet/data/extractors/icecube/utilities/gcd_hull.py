"""Convex hull object for IceCube geometry."""

from typing import TYPE_CHECKING, Tuple

from scipy.spatial import ConvexHull

import numpy as np

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        MuonGun,
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
        self.furthest_distance = np.sqrt((self.points**2).sum(axis=1).max())

    def point_in_hull(
        self, points: np.array, tolerance: float = 1e-12
    ) -> bool:
        """Check if a point is inside the convex hull.

        Args:
        points: I3Position object
        tolerance: Tolerance for the dot product
        """
        return np.array(
            [
                (np.dot(eq[:-1], points.T) + eq[-1] <= tolerance)
                for eq in self.equations
            ]
        ).all(axis=0)

    def rays_and_sphere_intersection_check(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        length: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check if rays intersect with the sphere approximating the hull.

        Args:
            points: Points from which the rays originate.
            directions: Directions of the rays.
            length: Length of the rays.

        Returns:
            Intersection points of the rays with the sphere.
        """
        w = directions * length[:, np.newaxis]
        a = (w**2).sum(-1)
        b = 2 * (w * points).sum(-1)
        c = (points**2).sum(-1) - self.furthest_distance**2

        # Check if the discriminant is negative
        discriminant = b**2 - 4 * a * c

        with np.errstate(divide="ignore", invalid="ignore"):
            t_pos = (-b + np.sqrt(discriminant)) / (2 * a)
            t_neg = (-b - np.sqrt(discriminant)) / (2 * a)
        t_pos_copy = t_pos.copy()
        t_neg_copy = t_neg.copy()
        # check if t_pos are large than 1 or nan/imaginary
        t_pos = np.isnan(np.where(t_pos > 1, np.nan, t_pos))
        t_neg = np.isnan(np.where(t_neg > 1, np.nan, t_neg))

        t_mask = (t_pos & t_neg) == 0
        return (discriminant >= 0) & t_mask, t_pos_copy, t_neg_copy

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
