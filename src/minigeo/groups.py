import numpy as np

from .geometry import BaseGeometry, Transform, resolve_pivot


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
        Compute the group’s center as the average of the individual geometry centers.
        """
        if not self.geometries:
            return np.zeros(3)
        centers = np.array([g.center for g in self.geometries])
        return np.mean(centers, axis=0)

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
