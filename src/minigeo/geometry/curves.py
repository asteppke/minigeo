"""Curves: polylines through given points, and curves given by a function."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..anchors import line_position
from ..transform import Transform
from .base import Geometry


class Polyline(Geometry):
    """
    A path of straight pieces through a list of points, open or closed.

    Anchors are positions along the path, measured by length, with the same names as for a
    :class:`Line` (``"start"``, ``"near end"``, ``"midway"``, ... or a number from 0 to 1).

    Example:
        >>> path = Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> path.length
        2.0
        >>> path.anchor("midway")
        array([1., 0., 0.])
    """

    label_anchor = "midway"

    def __init__(self, points, *, closed: bool = False):
        """
        Args:
            points (array-like): The points, ``(n, 3)``, at least two.
            closed: Whether the path returns from the last point to the first.
        """
        points = np.array(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError(
                f"A polyline needs at least two points with three coordinates each, got {points.shape}."
            )
        self._vertices = points
        self.closed = bool(closed)

    @property
    def vertices(self) -> np.ndarray:
        """The points, ``(n, 3)``."""
        return self._vertices

    @vertices.setter
    def vertices(self, value) -> None:
        self._vertices = np.asarray(value, dtype=float).reshape(-1, 3)

    @property
    def path(self) -> np.ndarray:
        """The points in drawing order; for a closed polyline the first point again at the end."""
        return np.vstack([self._vertices, self._vertices[:1]]) if self.closed else self._vertices

    @property
    def length(self) -> float:
        """The length of the path."""
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())

    def anchor(self, name: str | float = "midway") -> np.ndarray:
        """
        The point at a position along the path, measured by length.

        Args:
            name: ``"start"``, ``"end"``, ``"midway"`` (or ``"center"``), ``"near start"``, ...
                or a number: 0 is the start, 1 the end, 0.5 halfway along the path.
        """
        return _along(self.path, line_position(name))

    def __repr__(self) -> str:
        kind = "closed polyline" if self.closed else "polyline"
        return f"{kind.capitalize()} through {len(self._vertices)} points, length {self.length:.6g}"


class Curve(Polyline):
    """
    A curve given by a function of one parameter, ``t -> (x, y, z)``, drawn through samples.

    The function stays with the curve, so :meth:`at` and :meth:`tangent_at` are exact, also
    after the curve was moved or turned. The drawn path and its anchors use the samples.

    Example:
        >>> helix = Curve(lambda t: (np.cos(t), np.sin(t), 0.1 * t), t=(0, 4 * np.pi))
        >>> helix.at(np.pi)          # the point for t = pi: (-1, 0, 0.314...)
        >>> helix.tangent_at(0)      # the direction of the curve there
    """

    label_anchor = "end"

    def __init__(
        self,
        function: Callable[[float], tuple[float, float, float]],
        t: tuple[float, float] = (0.0, 1.0),
        *,
        samples: int = 100,
        closed: bool = False,
    ):
        """
        Args:
            function: Takes a number ``t`` and returns the point ``(x, y, z)`` for it.
            t: The range of the parameter, ``(first, last)``.
            samples: How many points the curve is drawn through (at least 2).
            closed: Whether the drawn path joins its last point to its first.
        """
        if samples < 2:
            raise ValueError("A curve needs at least 2 samples.")
        self.function = function
        self.t = (float(t[0]), float(t[1]))
        self._matrix = np.eye(4)  # the transforms applied since the curve was made
        parameters = np.linspace(*self.t, samples)
        super().__init__([self._evaluate(value) for value in parameters], closed=closed)

    def _evaluate(self, t: float) -> np.ndarray:
        point = np.asarray(self.function(t), dtype=float)
        if point.shape != (3,):
            raise ValueError(
                f"The function of a curve must return three coordinates, got {point.shape} for t={t}."
            )
        return point

    def at(self, t: float) -> np.ndarray:
        """The point of the curve for parameter ``t``, in world coordinates."""
        return self._matrix[:3, :3] @ self._evaluate(t) + self._matrix[:3, 3]

    def tangent_at(self, t: float) -> np.ndarray:
        """
        The unit direction of the curve at parameter ``t`` (towards growing ``t``), in world
        coordinates. Zero where the curve stands still.
        """
        step = 1e-6 * max(abs(self.t[1] - self.t[0]), 1e-12)
        derivative = self._matrix[:3, :3] @ (self._evaluate(t + step) - self._evaluate(t - step))
        length = np.linalg.norm(derivative)
        return derivative / length if length > 0 else derivative

    def apply_transform(self, transform: Transform) -> None:
        super().apply_transform(transform)
        self._matrix = transform.matrix @ self._matrix

    def __repr__(self) -> str:
        return f"Curve for t from {self.t[0]:g} to {self.t[1]:g}, {len(self._vertices)} samples, length {self.length:.6g}"


def _along(path: np.ndarray, fraction: float) -> np.ndarray:
    """The point a ``fraction`` of the way along a path of straight pieces (by length)."""
    pieces = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = pieces.sum()
    if total == 0:
        return path[0].copy()
    distance = np.clip(fraction, 0, 1) * total
    ends = np.cumsum(pieces)
    k = min(int(np.searchsorted(ends, distance)), len(pieces) - 1)
    begin = ends[k] - pieces[k]
    share = 0.0 if pieces[k] == 0 else (distance - begin) / pieces[k]
    return path[k] + share * (path[k + 1] - path[k])
