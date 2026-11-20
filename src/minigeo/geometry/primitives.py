"""Geometries without a body: points, line segments and axes."""

from __future__ import annotations

import numpy as np

from ..anchors import point_on_segment
from .base import BaseGeometry


class BasePoint(BaseGeometry):
    """
    Represents a single 3D point or vertex.
    """

    def __init__(self, position: np.ndarray, label: str | None = None):
        """
        Initialize a single point.

        Args:
            position (np.ndarray): The position of the point.
            label (Optional[str]): An optional label.
        """
        self.position = np.array(position)
        self.label = label if label is not None else ""

    @property
    def vertex(self) -> np.ndarray:
        """Return the position vertex."""
        return self.position

    @vertex.setter
    def vertex(self, value: np.ndarray):
        self.position = value

    @property
    def vertices(self) -> np.array:
        return np.reshape(self.position, (1, 3))

    @vertices.setter
    def vertices(self, value):
        self.position = value.flatten()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.position}"


class BaseLine(BaseGeometry):
    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        label: str | None = None,
    ):

        self.start = np.array(start)
        self.end = np.array(end)
        self.label = label if label is not None else ""
        self._vertices = np.array([self.start, self.end])

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = value
        self.start = value[0]
        self.end = value[1]

    @property
    def edges(self) -> np.ndarray:
        return self.vertices

    def anchor(self, name: str | float = "midway") -> np.ndarray:
        """
        Return a point along the line, named as in TikZ.

        Args:
            name: ``"start"``, ``"end"``, ``"midway"`` (or ``"center"``), ``"near start"``,
                ``"near end"``, ``"very near start"``, ``"very near end"``, or a number
                like TikZ's ``pos``: 0 is the start, 1 the end.

        Example:
            >>> line.anchor("near end")  # three quarters of the way along
        """
        return point_on_segment(self.start, self.end, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} from {self.start} to {self.end}"


class BaseAxis(BaseGeometry):
    """
    An axis: a segment from ``origin`` along a unit ``direction``, drawn as an arrow by default.

    Unlike a :class:`BaseLine`, it keeps its direction as a unit vector, which stays up to
    date when the axis is transformed.
    """

    def __init__(
        self,
        direction: np.ndarray,
        position: np.ndarray | None = None,
        *,
        length: float = 1.0,
        label: str | None = None,
    ):
        """
        Args:
            direction (array-like): The direction; only its direction matters, not its length.
            position (array-like, optional): Where the axis starts (default: the origin).
            length (float): How long the axis is drawn.
            label (str, optional): An optional name.

        Example:
            >>> BaseAxis([0, 0, 1], position=[1, 0, 0], length=2)  # from (1, 0, 0) up to (1, 0, 2)
        """
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm else direction
        self.origin = np.asarray(position, dtype=float) if position is not None else np.zeros(3)
        self.label = label if label is not None else ""
        self._vertices = np.array([self.origin, self.origin + self.direction * length])

    @property
    def length(self) -> float:
        """The length of the axis segment."""
        return float(np.linalg.norm(self._vertices[1] - self._vertices[0]))

    @property
    def vertices(self) -> np.ndarray:
        """Return the current endpoints (vertices) of the axis."""
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        """Set the endpoints; origin and direction are updated to match."""
        self._vertices = np.asarray(value)
        self.origin = self._vertices[0]
        vec = self._vertices[1] - self._vertices[0]
        norm = np.linalg.norm(vec)
        self.direction = vec / norm if norm else vec

    @property
    def edges(self) -> np.ndarray:
        """
        For an axis, the edge is defined by the line connecting its two endpoints.
        """
        return self.vertices

    def anchor(self, name: str | float = "midway") -> np.ndarray:
        """Return a point along the axis segment; the same names as :meth:`BaseLine.anchor`."""
        return point_on_segment(self.vertices[0], self.vertices[1], name)

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.origin} with direction {self.direction}"
