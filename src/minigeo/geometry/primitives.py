"""Geometries without a body: points, line segments and axes."""

from __future__ import annotations

import numpy as np

from ..anchors import point_on_segment
from .base import Geometry


class Point(Geometry):
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


class Line(Geometry):
    label_anchor = "midway"

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


class Axis(Geometry):
    """
    An axis: a segment from ``origin`` along a unit ``direction``, drawn as an arrow by default.

    Unlike a :class:`Line`, it keeps its direction as a unit vector, which stays up to
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
            >>> Axis([0, 0, 1], position=[1, 0, 0], length=2)  # from (1, 0, 0) up to (1, 0, 2)
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
        """Return a point along the axis segment; the same names as :meth:`Line.anchor`."""
        return point_on_segment(self.vertices[0], self.vertices[1], name)

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.origin} with direction {self.direction}"


def _components(value) -> np.ndarray:
    """The components of a Vector, or of anything array-like with three numbers."""
    if isinstance(value, Vector):
        return value.components
    components = np.asarray(value, dtype=float)
    if components.shape != (3,):
        raise ValueError(f"A vector needs three components, got {value!r}.")
    return components


class Vector(Line):
    """
    A vector: an arrow from ``origin`` with the given ``components``.

    Besides drawing, vectors do vector algebra. The results of ``+``, ``-``, scaling,
    :meth:`cross` and :meth:`projected_onto` start at this vector's origin, so they can be
    drawn right away. Transforms move both ends, like for a :class:`Line`.

    Example:
        >>> a = Vector([1, 0, 0])
        >>> b = Vector([0, 1, 0])
        >>> a.cross(b).components
        array([0., 0., 1.])
        >>> a.angle_to(b)
        90.0
    """

    #: Labels go at the arrow tip unless another anchor is given.
    label_anchor = "end"

    def __init__(self, components, origin=None):
        """
        Args:
            components (array-like): The three components.
            origin (array-like, optional): Where the arrow starts; default the origin.
        """
        origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        super().__init__(origin, origin + _components(components))

    @classmethod
    def between(cls, start, end) -> Vector:
        """The vector from point ``start`` to point ``end``."""
        start = np.asarray(start, dtype=float)
        return cls(np.asarray(end, dtype=float) - start, origin=start)

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.start, dtype=float)

    @property
    def components(self) -> np.ndarray:
        return np.asarray(self.end, dtype=float) - self.origin

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.components))

    @property
    def direction(self) -> np.ndarray:
        """The unit vector in the same direction (zero for a zero vector)."""
        length = self.length
        return self.components / length if length else self.components

    # ------------------------------------------------------------ algebra

    def dot(self, other) -> float:
        """The dot product with another vector (or three numbers)."""
        return float(np.dot(self.components, _components(other)))

    def cross(self, other) -> Vector:
        """The cross product, starting at this vector's origin."""
        return Vector(np.cross(self.components, _components(other)), origin=self.origin)

    def angle_to(self, other, in_degrees: bool = True) -> float:
        """The angle between the two directions, from 0 to 180 degrees (or 0 to pi)."""
        a, b = self.components, _components(other)
        lengths = np.linalg.norm(a) * np.linalg.norm(b)
        if lengths == 0:
            raise ValueError("A zero vector has no direction.")
        angle = float(np.arccos(np.clip(np.dot(a, b) / lengths, -1, 1)))
        return float(np.degrees(angle)) if in_degrees else angle

    def projected_onto(self, other) -> Vector:
        """The part of this vector along ``other``, starting at this vector's origin."""
        direction = _components(other)
        squared = float(np.dot(direction, direction))
        if squared == 0:
            raise ValueError("Cannot project onto a zero vector.")
        return Vector(np.dot(self.components, direction) / squared * direction, origin=self.origin)

    def __add__(self, other) -> Vector:
        return Vector(self.components + _components(other), origin=self.origin)

    def __sub__(self, other) -> Vector:
        return Vector(self.components - _components(other), origin=self.origin)

    def __mul__(self, factor: float) -> Vector:
        return Vector(self.components * float(factor), origin=self.origin)

    __rmul__ = __mul__

    def __truediv__(self, divisor: float) -> Vector:
        return Vector(self.components / float(divisor), origin=self.origin)

    def __neg__(self) -> Vector:
        return Vector(-self.components, origin=self.origin)

    def __repr__(self) -> str:
        return f"Vector {self.components} at {self.origin}"
