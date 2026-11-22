"""A plane, drawn as a square patch of it."""

from __future__ import annotations

import numpy as np

from ..transform import rotate_arbitrary_vector_matrix
from .primitives import Vector
from .shapes import Rectangle


def rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """The rotation matrix that turns the z axis into ``direction`` (a unit vector)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    sine, cosine = np.linalg.norm(axis), float(np.dot(z, direction))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])  # already z, or exactly -z
    return rotate_arbitrary_vector_matrix(axis, np.arctan2(sine, cosine), in_degrees=False)


class Plane(Rectangle):
    """
    A plane through ``point`` with the given ``normal``.

    Mathematically a plane has no edges; it is drawn as a square patch of ``size`` around
    ``point``, translucent by default. Its own axes x' and y' lie in the plane and z' is the
    normal, so anchors and ``rotate("z'", ...)`` work as for other shapes.

    Example:
        >>> floor = Plane([0, 0, 0], normal=[0, 0, 1], size=4)
        >>> floor.distance_to([1, 1, 2])
        2.0
    """

    def __init__(self, point, normal, *, size=2.0):
        """
        Args:
            point (array-like): A point of the plane; the patch is centered there.
            normal (array-like): A vector perpendicular to the plane (any length).
            size: Side length of the drawn patch, or ``(width, length)``.
        """
        normal = np.asarray(normal, dtype=float)
        length = np.linalg.norm(normal)
        if length == 0:
            raise ValueError("The normal of a plane must not be zero.")
        super().__init__(point, size=size)
        self.pose.matrix[:3, :3] = rotation_from_z(normal / length)

    @classmethod
    def through(cls, a, b, c, *, size=None) -> Plane:
        """
        The plane through three points, centered at their middle.

        Args:
            size: Side length of the drawn patch; default large enough to show all three points.
        """
        a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("The three points lie on one line; they do not define a plane.")
        middle = (a + b + c) / 3
        if size is None:
            size = 2.5 * max(np.linalg.norm(p - middle) for p in (a, b, c))
        return cls(middle, normal, size=size)

    @property
    def normal(self) -> np.ndarray:
        """The unit normal: the plane's own z' axis, or its opposite after mirroring."""
        return -self.axes[2] if self._mirrored else self.axes[2]

    @property
    def point(self) -> np.ndarray:
        """The point the plane goes through (the center of the drawn patch)."""
        return self.center

    def distance_to(self, point) -> float:
        """The signed distance of a point: positive on the side the normal points to."""
        return float(np.dot(np.asarray(point, dtype=float) - self.center, self.normal))

    def project(self, point) -> np.ndarray:
        """The foot of the perpendicular from ``point`` onto the plane."""
        point = np.asarray(point, dtype=float)
        return point - self.distance_to(point) * self.normal

    def contains(self, point, tolerance: float = 1e-9) -> bool:
        return abs(self.distance_to(point)) <= tolerance

    def intersect(self, line, *, segment: bool = False) -> np.ndarray:
        """
        Where a line meets the plane, as for every shape (:meth:`Shape.intersect`), but with
        the whole infinite plane, not only the drawn patch.

        Args:
            line: A Line, Vector, Axis, Polyline, Curve, or a pair of points ``(start, end)``.
            segment: Only points between the line's start and end; otherwise the line is
                extended in both directions.

        Returns:
            An ``(n, 3)`` array: one point, or none if the line runs parallel to the plane
            (a polyline can cross it several times).

        Example:
            >>> hit = floor.intersect(ray)[0]
        """
        return super().intersect(line, segment=segment)

    def normal_vector(self, length: float = 1.0) -> Vector:
        """The normal as a :class:`Vector` from the plane's center, e.g. for drawing it."""
        return Vector(self.normal * length, origin=self.center)

    def __repr__(self) -> str:
        return f"Plane through {self.center} with normal {np.round(self.normal, 6)}"
