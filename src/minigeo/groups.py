from __future__ import annotations

import numpy as np

from .anchors import point_in_bounding_box, resolve_pivot
from .geometry import BaseGeometry
from .transform import Transform


class GeometryGroup:
    def __init__(self, geometries: list[BaseGeometry] | None = None):
        """
        Container for multiple BaseGeometry objects.

        Args:
            geometries (list[BaseGeometry] | None): Optional starting list of geometries.
        """
        self.geometries = list(geometries) if geometries is not None else []

    def add(self, geometry: BaseGeometry) -> None:
        """Add a geometry object to the group."""
        self.geometries.append(geometry)

    def remove(self, geometry: BaseGeometry) -> None:
        """Remove a geometry object from the group."""
        self.geometries.remove(geometry)

    @property
    def center(self) -> np.ndarray:
        """
        The middle of the group's bounding box, the same point as ``group.anchor("center")``.

        An empty group has its center at the origin.
        """
        if not self.geometries:
            return np.zeros(3)
        return self.anchor("center")

    @property
    def vertices(self) -> np.ndarray:
        """Return the vertices of the geometry."""
        return np.concatenate([g.vertices for g in self.geometries])

    @vertices.setter
    def vertices(self, value: np.ndarray):
        """Set the vertices of the geometry."""
        # TODO: This is not very elegant because we need to reassign the vertices to each geometry
        # but allows to use a flat array of vertices.
        start = 0
        for geometry in self.geometries:
            end = start + len(geometry.vertices)
            geometry.vertices = value[start:end]
            geometry.update_geometry()
            start = end

    def update_geometry(self) -> None:
        """Update the geometry of the group."""
        for geometry in self.geometries:
            geometry.update_geometry()

    def anchor(self, name: str | np.ndarray = "center") -> np.ndarray:
        """
        Return a named point on the group's bounding box, in world coordinates.

        The names are the same as for shapes (``"top"``, ``"bottom front left"``, ...), but
        since a group has no axes of its own they refer to the world axes:
        left/right is x, front/back is y and bottom/top is z.

        Example:
            >>> robot.place("bottom", at=[0, 0, 0])  # stand the whole group on the floor
        """
        if not self.geometries:
            raise ValueError("An empty group has no anchors.")
        return point_in_bounding_box(self.vertices, name)

    def place(self, anchor: str | np.ndarray, at: np.ndarray) -> GeometryGroup:
        """Move all members together (without turning) so that the group's ``anchor`` ends up at ``at``."""
        offset = np.asarray(at, dtype=float) - self.anchor(anchor)
        self.apply_transform(Transform().translate(offset))
        return self

    def apply_transform(self, transform: Transform) -> None:
        """Apply a transform to every member, so each keeps its own state (e.g. center) in sync."""
        for geometry in self.geometries:
            geometry.apply_transform(transform)

    def rotate(
        self,
        axis: str | np.ndarray,
        angle: float,
        in_degrees: bool = True,
        *,
        about: np.ndarray | str | None = None,
    ) -> None:
        """
        Rotate all members together, like :meth:`BaseGeometry.rotate`.

        ``about="center"`` uses the group's center, so the members keep their arrangement
        and the group spins in place.
        """
        pivot = resolve_pivot(about, self)
        self.apply_transform(Transform().rotate(axis, angle, in_degrees, about=pivot))
