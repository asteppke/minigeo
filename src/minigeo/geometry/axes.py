"""
Coordinate axes with ticks and numbers, and grids like graph paper.

Both are ordinary geometries: add them to a scene like any other, style them, and
transform them. They draw in every backend.

Example:
    >>> axes = scene.add(Axes(low=[0, 0, 0], high=[3, 2, 1], step=0.5))
    >>> scene.add(axes.grid("xy"))
"""

from __future__ import annotations

import numpy as np

from .base import Geometry

#: The plane of a grid: which two world axes it spans.
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

#: Along which axis the ticks of each axis point: x ticks along y, y and z ticks along x.
TICK_DIRECTIONS = {0: 1, 1: 0, 2: 0}


def tick_values(low: float, high: float, step: float) -> np.ndarray:
    """The multiples of ``step`` from ``low`` to ``high`` (both included if they are multiples)."""
    first, last = np.ceil(low / step - 1e-9), np.floor(high / step + 1e-9)
    return np.round(np.arange(first, last + 1) * step, 12)


class Axes(Geometry):
    """
    Three coordinate axes with ticks at regular steps and the numbers next to them.

    The axes cross at ``cross`` (the origin, if it lies within the range; otherwise the
    nearest point of the range) and run from ``low`` to ``high`` with an arrow at the end.
    The ticks of the x axis point along y, those of the y and z axes along x. The tick at
    the crossing gets no number, as it would collide with the other axes.

    Not to be confused with :class:`Axis`, a single arrow.

    Attributes:
        ticks: For each tick: the axis (0, 1, 2), where it lies along the axis (0 at the
            start, 1 at the end) and its number as text.

    Example:
        >>> Axes(low=[-1, -1, 0], high=[2, 2, 1.5], step=0.5)
        >>> Axes(high=[4, 3, 0], step=1)  # only x and y: z goes from 0 to 0
    """

    def __init__(
        self,
        low=(0.0, 0.0, 0.0),
        high=(1.0, 1.0, 1.0),
        *,
        step=1.0,
        cross=(0.0, 0.0, 0.0),
        names=("x", "y", "z"),
        tick_length: float | None = None,
    ):
        """
        Args:
            low, high (array-like): Where each axis starts and ends. An axis with equal
                ends is left out, e.g. z for a 2D drawing.
            step: Distance between ticks: one number, or one per axis.
            cross (array-like): Where the axes cross, if it lies within the range.
            names: The names at the arrow tips; ``""`` for none.
            tick_length: Length of the ticks; default 3 % of the longest axis.
        """
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        steps = np.broadcast_to(np.asarray(step, dtype=float), (3,))
        if np.any(high < low):
            raise ValueError(f"Axes need low <= high on every axis, got {low} and {high}.")
        if np.any(steps <= 0):
            raise ValueError(f"The step must be positive, got {step!r}.")
        crossing = np.clip(np.asarray(cross, dtype=float), low, high)
        self.names = tuple(names)
        self.tick_length = tick_length if tick_length is not None else 0.03 * float(np.max(high - low) or 1)

        # the ends of the three axes, then the crossing
        ends = []
        for axis in range(3):
            for value in (low[axis], high[axis]):
                point = crossing.copy()
                point[axis] = value
                ends.append(point)
        self._vertices = np.array([*ends, crossing])

        # (axis, fraction along it, text) for each tick
        self.ticks: list[tuple[int, float, str]] = []
        for axis in range(3):
            if high[axis] == low[axis]:
                continue
            for value in tick_values(low[axis], high[axis], steps[axis]):
                if abs(value - crossing[axis]) > 1e-9 * steps[axis]:
                    fraction = (value - low[axis]) / (high[axis] - low[axis])
                    self.ticks.append((axis, float(fraction), f"{value:g}"))

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = np.asarray(value, dtype=float)

    def axis_ends(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Start and end of axis 0 (x), 1 (y) or 2 (z)."""
        return self._vertices[2 * axis], self._vertices[2 * axis + 1]

    @property
    def present(self) -> list[int]:
        """The axes that are drawn (those with different ends)."""
        return [axis for axis in range(3) if not np.allclose(*self.axis_ends(axis))]

    def tick_direction(self, axis: int) -> np.ndarray:
        """The unit direction in which the ticks of an axis point (along another axis)."""
        start, end = self.axis_ends(TICK_DIRECTIONS[axis])
        if np.allclose(start, end):  # that axis is left out: use the third one
            start, end = self.axis_ends(3 - axis - TICK_DIRECTIONS[axis])
        direction = end - start
        return direction / np.linalg.norm(direction) if np.linalg.norm(direction) else direction

    def grid(self, plane: str = "xy") -> Grid:
        """
        A :class:`Grid` that matches these axes: a line at every tick, in the plane of two
        axes through their crossing. (The axes themselves take the place of the lines through
        the crossing.)

        Example:
            >>> axes = scene.add(Axes(high=[3, 2, 1]))
            >>> scene.add(axes.grid("xy"))
        """
        if plane not in PLANES:
            raise ValueError(f"Unknown plane {plane!r}; use {', '.join(PLANES)}.")
        first, second = PLANES[plane]
        (a0, a1), (b0, b1) = self.axis_ends(first), self.axis_ends(second)
        crossing = self._vertices[6]
        corner = a0 + b0 - crossing
        ticks = [[fraction for a, fraction, _ in self.ticks if a == axis] for axis in (first, second)]
        return Grid.through(corner, a1 - a0, b1 - b0, *ticks, border=False)

    def __repr__(self) -> str:
        return f"Axes crossing at {self._vertices[6]} with {len(self.ticks)} ticks"


class Grid(Geometry):
    """
    Lines at regular distances in a parallelogram, like graph paper.

    Usually made by :meth:`Grid.flat` (in a plane of the world axes) or
    :meth:`Axes.grid` (matching a set of axes).
    """

    def __init__(self, corner, side1, side2, *, spacing=1.0, offsets=(0.0, 0.0)):
        """
        Args:
            corner (array-like): One corner of the grid.
            side1, side2 (array-like): The two sides from that corner, as vectors.
            spacing: Distance between the lines: one number, or one per side.
            offsets: Where the first line lies along each side, from the corner.
                The border is always drawn.
        """
        corner = np.asarray(corner, dtype=float)
        side1, side2 = np.asarray(side1, dtype=float), np.asarray(side2, dtype=float)
        spacings = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
        if np.any(spacings <= 0):
            raise ValueError(f"The spacing must be positive, got {spacing!r}.")
        self._vertices = np.array([corner, corner + side1, corner + side2])
        self.fractions = []  # for each side: where lines cross it, from 0 to 1
        for side, space, offset in zip((side1, side2), spacings, offsets, strict=True):
            length = float(np.linalg.norm(side))
            inner = np.arange(offset, length, space) / length if length else np.empty(0)
            self.fractions.append(_distinct([0.0, *inner, 1.0]))

    @classmethod
    def through(cls, corner, side1, side2, fractions1, fractions2, *, border: bool = True) -> Grid:
        """
        A grid with lines where given: as fractions (0 to 1) along each side, and the border
        unless ``border=False``.
        """
        grid = cls(corner, side1, side2)
        grid.fractions = [
            _distinct([0.0, *fractions, 1.0] if border else fractions)
            for fractions in (fractions1, fractions2)
        ]
        return grid

    @classmethod
    def flat(cls, low=(-1.0, -1.0), high=(1.0, 1.0), *, step=1.0, plane: str = "xy", at: float = 0.0) -> Grid:
        """
        A grid in a plane of the world axes.

        Args:
            low, high: Opposite corners, in the two coordinates of the plane.
            step: Distance between the lines (at multiples of it): one number, or one per
                direction.
            plane: ``"xy"``, ``"xz"`` or ``"yz"``.
            at: The third coordinate, e.g. the height of an ``"xy"`` grid.

        Example:
            >>> Grid.flat(low=(-2, -2), high=(2, 2), step=0.5)  # the floor around the origin
        """
        if plane not in PLANES:
            raise ValueError(f"Unknown plane {plane!r}; use {', '.join(PLANES)}.")
        first, second = PLANES[plane]
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        steps = np.broadcast_to(np.asarray(step, dtype=float), (2,))
        corner = np.zeros(3)
        corner[[first, second]] = low
        corner[3 - first - second] = at
        side1, side2 = np.zeros(3), np.zeros(3)
        side1[first], side2[second] = high[0] - low[0], high[1] - low[1]
        offsets = [(-value) % space for value, space in zip(low, steps, strict=True)]
        return cls(corner, side1, side2, spacing=steps, offsets=offsets)

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = np.asarray(value, dtype=float)

    @property
    def lines(self) -> list[np.ndarray]:
        """Every line of the grid as a ``(2, 3)`` array of its ends."""
        corner, end1, end2 = self._vertices
        side1, side2 = end1 - corner, end2 - corner
        lines = [np.array([corner + f * side1, corner + f * side1 + side2]) for f in self.fractions[0]]
        lines += [np.array([corner + f * side2, corner + f * side2 + side1]) for f in self.fractions[1]]
        return lines

    @property
    def pieces(self) -> list[np.ndarray]:
        """The lines cut where they cross each other: the sides of the grid's cells."""
        corner, end1, end2 = self._vertices
        side1, side2 = end1 - corner, end2 - corner
        first, second = self.fractions
        pieces = []
        for across, along, side, other in ((first, second, side1, side2), (second, first, side2, side1)):
            for f in across:
                points = corner + f * side + np.outer(_distinct([0.0, *along, 1.0]), other)
                pieces += [points[k : k + 2] for k in range(len(points) - 1)]
        return pieces

    def __repr__(self) -> str:
        return f"Grid at {self._vertices[0]} with {len(self.lines)} lines"


def _distinct(fractions) -> np.ndarray:
    """The fractions in order, without (nearly) equal ones."""
    ordered = np.sort(np.asarray(fractions, dtype=float))
    return ordered[np.concatenate([[True], np.diff(ordered) > 1e-9])] if len(ordered) else ordered
