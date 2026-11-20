"""The common base class of all geometries."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from ..anchors import resolve_pivot
from ..transform import Transform


class Geometry(ABC):
    """
    Base class for all geometry classes.
    Every geometry contains vertices (np.ndarrray) as underlying properties."""

    @property
    @abstractmethod
    def vertices(self) -> np.ndarray:
        """Return the vertices of the geometry."""

    @vertices.setter
    @abstractmethod
    def vertices(self, value: np.ndarray):
        """Set the vertices of the geometry."""

    @property
    def center(self) -> np.ndarray:
        """The mean of all vertices. Shapes override this with their defined center."""
        return self.vertices.mean(axis=0)

    def copy(self):
        """
        Return an independent copy: transforming one does not change the other.

        Example:
            >>> after = before.copy()
            >>> after.rotate("z", 60)
        """
        return copy.deepcopy(self)

    def apply_transform(self, transform: Transform) -> None:
        """Transform all vertices in place. Subclasses extend this to update derived state."""
        self.vertices = transform.apply_to_vertices(self.vertices)

    def rotate(
        self,
        axis: str | np.ndarray,
        angle: float,
        in_degrees: bool = True,
        *,
        about: np.ndarray | str | None = None,
    ) -> None:
        """
        Rotate the geometry in place.

        Args:
            axis (str or array-like): The axis direction ('x', 'y', 'z') or a direction vector.
            angle (float): The rotation angle.
            in_degrees (bool): If True, the angle is in degrees, else in radians.
            about (array-like or "center", optional): A point the rotation axis passes through.
                ``"center"`` uses the geometry's own center. Default: the origin.

        Example:
            >>> box.rotate("z", 45, about="center")  # spin the box in place
        """
        pivot = resolve_pivot(about, self)
        self.apply_transform(Transform().rotate(axis, angle, in_degrees, about=pivot))

    def anchor(self, name: str = "center") -> np.ndarray:
        """
        Return a named point of the geometry in world coordinates.

        Every geometry has the ``"center"`` anchor; lines and shapes add their own names.
        """
        if isinstance(name, str) and name == "center":
            return self.center
        raise ValueError(f"{type(self).__name__} has no anchor {name!r}; only 'center'.")

    def place(self, anchor: str | np.ndarray | float, at: np.ndarray) -> Geometry:
        """
        Move the geometry (without turning it) so that its ``anchor`` ends up at ``at``.

        Returns the geometry itself, so it can be used right after creating it.

        Example:
            >>> lamp.place("bottom", at=table.anchor("top"))  # put the lamp on the table
        """
        offset = np.asarray(at, dtype=float) - self.anchor(anchor)
        self.apply_transform(Transform().translate(offset))
        return self
