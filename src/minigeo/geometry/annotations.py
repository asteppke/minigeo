"""
Annotations, as in technical drawings and TikZ: angle arcs and dimension lines.

They are geometries like any other, so transforms move them along with the rest, but they
are drawn on top of solids so that they stay readable.
"""

from __future__ import annotations

import numpy as np

from .base import Geometry
from .primitives import Line


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError("Cannot take the direction of a zero-length vector.")
    return vector / length


def _perpendicular(direction: np.ndarray) -> np.ndarray:
    """Some unit vector perpendicular to ``direction``."""
    helper = np.array([1.0, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(np.cross(direction, helper))


class Angle(Geometry):
    """
    The angle at ``vertex`` between the rays towards ``start`` and ``end``, drawn as an arc.

    A right angle is drawn with a small square instead (unless ``right_angle_mark=False``).
    Give it a ``fill`` style to shade the sector. The size in degrees is in :attr:`degrees`.

    Anchors: ``"vertex"``, ``"start"`` and ``"end"`` (ends of the arc), ``"midway"`` (middle
    of the arc) and ``"label"`` (just outside the middle, where labels go by default).

    Example:
        >>> angle = Angle([0, 0, 0], [1, 0, 0], [1, 1, 0], radius=0.5)
        >>> angle.degrees
        45.0
        >>> scene.add(angle, label="θ")
    """

    label_anchor = "label"
    label_align = "center"

    def __init__(
        self, vertex, start, end, *, radius: float = 0.3, right_angle_mark: bool = True, segments: int = 24
    ):
        """
        Args:
            vertex (array-like): The corner of the angle.
            start, end (array-like): A point on each of the two rays.
            radius: Radius of the arc.
            right_angle_mark: Draw a right angle as a square, as is usual in geometry.
            segments: Number of straight pieces the arc is drawn with.
        """
        self._vertices = np.array([vertex, start, end], dtype=float)
        _unit(self._vertices[1] - self._vertices[0])
        _unit(self._vertices[2] - self._vertices[0])  # both rays need a direction
        self.radius = float(radius)
        self.right_angle_mark = right_angle_mark
        self.segments = int(segments)

    @classmethod
    def between(cls, first: Line, second: Line, **options) -> Angle:
        """
        The angle between two lines or vectors, at the start of ``first``.

        ``second`` only contributes its direction, so it does not need to start at the same point.
        """
        vertex = np.asarray(first.start, dtype=float)
        start = vertex + _unit(np.asarray(first.end, dtype=float) - first.start)
        end = vertex + _unit(np.asarray(second.end, dtype=float) - second.start)
        return cls(vertex, start, end, **options)

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = np.asarray(value, dtype=float)

    @property
    def vertex(self) -> np.ndarray:
        return self._vertices[0]

    @property
    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """The unit directions from the vertex towards ``start`` and ``end``."""
        vertex, start, end = self._vertices
        return _unit(start - vertex), _unit(end - vertex)

    @property
    def radians(self) -> float:
        u, w = self.rays
        return float(np.arccos(np.clip(np.dot(u, w), -1, 1)))

    @property
    def degrees(self) -> float:
        return float(np.degrees(self.radians))

    @property
    def is_right(self) -> bool:
        return abs(self.degrees - 90) < 1e-6

    def _point_at(self, fraction: float, distance: float) -> np.ndarray:
        """The point at ``fraction`` of the way from the first ray to the second, at a distance."""
        u, w = self.rays
        across = w - np.dot(w, u) * u  # the part of w perpendicular to u
        across = _unit(across) if np.linalg.norm(across) > 1e-12 else _perpendicular(u)
        angle = fraction * self.radians
        return self.vertex + distance * (np.cos(angle) * u + np.sin(angle) * across)

    @property
    def arc_points(self) -> np.ndarray:
        """The arc as a polyline from the first ray to the second."""
        return np.array([self._point_at(t, self.radius) for t in np.linspace(0, 1, self.segments + 1)])

    @property
    def right_angle_points(self) -> np.ndarray:
        """The square mark of a right angle, as a polyline."""
        u, w = self.rays
        side = 0.7 * self.radius
        return self.vertex + side * np.array([u, u + w, w])

    @property
    def shows_square(self) -> bool:
        return self.right_angle_mark and self.is_right

    def anchor(self, name: str = "label") -> np.ndarray:
        points = {
            "vertex": lambda: self.vertex,
            "start": lambda: self._point_at(0, self.radius),
            "end": lambda: self._point_at(1, self.radius),
            "midway": lambda: self._point_at(0.5, self.radius),
            "center": lambda: self._point_at(0.5, self.radius),
            "label": lambda: self._point_at(0.5, 1.35 * self.radius),
        }
        if name not in points:
            raise ValueError(f"Unknown anchor {name!r} of an angle; use {', '.join(points)}.")
        return points[name]()

    def __repr__(self) -> str:
        return f"Angle of {self.degrees:.6g}° at {self.vertex}"


class Dimension(Geometry):
    """
    A dimension line, as in technical drawings: the distance between ``start`` and ``end``.

    The dimension line is drawn at ``offset`` from the measured points, with extension lines
    leading to it, arrows at both ends and the length as text in the middle.

    Anchors: ``"start"``, ``"end"``, ``"midway"`` (on the dimension line) and ``"label"``
    (where the text goes).

    Example:
        >>> Dimension(box.anchor("bottom front left"), box.anchor("bottom front right"),
        ...           offset=[0, -0.3, 0], unit=" m")
    """

    label_anchor = "label"

    def __init__(self, start, end, *, offset=None, text: str | None = None, digits: int = 2, unit: str = ""):
        """
        Args:
            start, end (array-like): The two points whose distance is shown.
            offset (array-like, optional): Where the dimension line goes, relative to the points,
                e.g. ``[0, -0.3, 0]`` to draw it in front of them. Default: through the points.
            text: The text to show; default the length with ``digits`` decimals and ``unit``.
            digits: Decimals of the length shown (trailing zeros are dropped).
            unit: Appended to the length, e.g. ``" m"`` or ``" cm"``.
        """
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if np.linalg.norm(end - start) < 1e-12:
            raise ValueError("A dimension needs two different points.")
        offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
        self._vertices = np.array([start, end, start + offset])
        self._text, self.digits, self.unit = text, digits, unit

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = np.asarray(value, dtype=float)

    @property
    def start(self) -> np.ndarray:
        return self._vertices[0]

    @property
    def end(self) -> np.ndarray:
        return self._vertices[1]

    @property
    def offset(self) -> np.ndarray:
        return self._vertices[2] - self._vertices[0]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def text(self) -> str:
        """The text shown: the given one, or the current length."""
        if self._text is not None:
            return self._text
        return f"{round(self.length, self.digits):g}{self.unit}"

    @property
    def line_points(self) -> tuple[np.ndarray, np.ndarray]:
        """The two ends of the dimension line."""
        return self.start + self.offset, self.end + self.offset

    @property
    def extension_lines(self) -> list[np.ndarray]:
        """Lines from the measured points to a little beyond the dimension line (none without offset)."""
        offset = self.offset
        distance = np.linalg.norm(offset)
        if distance < 1e-12:
            return []
        beyond = offset / distance * min(0.15 * distance, 0.05 * self.length)
        return [np.array([point, point + offset + beyond]) for point in (self.start, self.end)]

    def anchor(self, name: str = "label") -> np.ndarray:
        a, b = self.line_points
        middle = (a + b) / 2
        points = {
            "start": a,
            "end": b,
            "midway": middle,
            "center": middle,
            "label": middle + 0.5 * self.offset,  # beyond the dimension line, away from the object
        }
        if name not in points:
            raise ValueError(f"Unknown anchor {name!r} of a dimension; use {', '.join(points)}.")
        return points[name]

    def __repr__(self) -> str:
        return f"Dimension {self.text} from {self.start} to {self.end}"
