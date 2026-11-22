"""Geometries without a body: points, line segments and axes."""

from __future__ import annotations

import numpy as np

from ..anchors import point_on_segment
from .base import Geometry


class Point(Geometry):
    """A single point, drawn as a dot."""

    def __init__(self, position: np.ndarray):
        """
        Args:
            position (array-like): Where the point is.

        Example:
            >>> scene.add(Point([0, 0, 1]), label="P")
        """
        self.position = np.asarray(position, dtype=float).copy()

    @property
    def vertex(self) -> np.ndarray:
        """The position, the same as :attr:`position`."""
        return self.position

    @vertex.setter
    def vertex(self, value: np.ndarray):
        self.position = np.asarray(value, dtype=float).copy()

    @property
    def vertices(self) -> np.ndarray:
        return np.reshape(self.position, (1, 3))

    @vertices.setter
    def vertices(self, value):
        self.position = np.asarray(value, dtype=float).reshape(3).copy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.position}"


class Line(Geometry):
    """
    A straight line segment from ``start`` to ``end``.

    ``start`` and ``end`` are views of :attr:`vertices`, so changing one changes the other.

    Example:
        >>> scene.add(Line([0, 0, 0], [1, 1, 0]), dash="dashed", label="diagonal")
    """

    label_anchor = "midway"

    def __init__(self, start: np.ndarray, end: np.ndarray):
        """
        Args:
            start, end (array-like): The two ends.
        """
        self._vertices = np.array([start, end], dtype=float)

    @property
    def start(self) -> np.ndarray:
        """The first end (a view: changing it moves the line)."""
        return self._vertices[0]

    @start.setter
    def start(self, value):
        self._vertices[0] = value

    @property
    def end(self) -> np.ndarray:
        """The second end (a view: changing it moves the line)."""
        return self._vertices[1]

    @end.setter
    def end(self, value):
        self._vertices[1] = value

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = np.array(value, dtype=float).reshape(2, 3)

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

    Unlike a :class:`Line`, it is given by a direction and a length; ``origin`` and
    ``direction`` stay up to date when the axis is transformed.
    """

    def __init__(self, direction: np.ndarray, origin: np.ndarray | None = None, *, length: float = 1.0):
        """
        Args:
            direction (array-like): The direction; only its direction matters, not its length.
            origin (array-like, optional): Where the axis starts (default: the origin),
                as for a :class:`Vector`.
            length (float): How long the axis is drawn.

        Example:
            >>> Axis([0, 0, 1], origin=[1, 0, 0], length=2)  # from (1, 0, 0) up to (1, 0, 2)
        """
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        self._direction = direction / norm if norm else direction  # kept for zero lengths
        start = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        self._vertices = np.array([start, start + self._direction * length])

    @property
    def origin(self) -> np.ndarray:
        """Where the axis starts."""
        return self._vertices[0]

    @property
    def direction(self) -> np.ndarray:
        """The unit direction of the axis."""
        along = self._vertices[1] - self._vertices[0]
        norm = np.linalg.norm(along)
        return along / norm if norm else self._direction

    @property
    def length(self) -> float:
        """The length of the axis segment."""
        return float(np.linalg.norm(self._vertices[1] - self._vertices[0]))

    @property
    def vertices(self) -> np.ndarray:
        """The two ends of the axis segment."""
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = np.array(value, dtype=float).reshape(2, 3)
        self._direction = self.direction

    @property
    def edges(self) -> np.ndarray:
        """For an axis, the edge is the segment between its two ends."""
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
        """Where the arrow starts (the same as :attr:`start`)."""
        return self.start.copy()

    @property
    def components(self) -> np.ndarray:
        """The three components (from the origin to the tip)."""
        return self.end - self.start

    @property
    def length(self) -> float:
        """The length of the vector."""
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

    #: Lets ``numpy_array + vector`` use :meth:`__radd__` instead of an element-wise numpy sum.
    __array_ufunc__ = None

    def __add__(self, other) -> Vector:
        return Vector(self.components + _components(other), origin=self.origin)

    def __radd__(self, other) -> Vector:
        """``[1, 2, 3] + v``: the sum, starting at this vector's origin."""
        return self + other

    def __sub__(self, other) -> Vector:
        return Vector(self.components - _components(other), origin=self.origin)

    def __rsub__(self, other) -> Vector:
        """``[1, 2, 3] - v``: the difference, starting at this vector's origin."""
        return Vector(_components(other) - self.components, origin=self.origin)

    def __mul__(self, factor: float) -> Vector:
        return Vector(self.components * float(factor), origin=self.origin)

    __rmul__ = __mul__

    def __truediv__(self, divisor: float) -> Vector:
        return Vector(self.components / float(divisor), origin=self.origin)

    def __neg__(self) -> Vector:
        return Vector(-self.components, origin=self.origin)

    def __repr__(self) -> str:
        return f"Vector {self.components} at {self.origin}"
