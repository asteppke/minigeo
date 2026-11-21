"""
Picking anchors and measuring between them, independent of any backend.

:func:`pickable_points` lists the points of a scene that can be picked: the named anchors
that lie on each shape (``"top"``, ``"top front left"``, ...), the ends and middle of lines,
and points. A :class:`Measurement` remembers up to two picks and reports their coordinates,
distance and difference. Interactive backends (``scene.measure()``) use both: they snap the
mouse to the nearest pickable point and show the measurement.

Example:
    >>> spots = pickable_points(scene)
    >>> measurement = Measurement()
    >>> measurement.pick(spots[0])
    >>> measurement.pick(spots[5])
    >>> measurement.distance
    1.25
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchors import point_in_bounding_box
from .geometry import Angle, Axis, Dimension, Geometry, Line, Point, Shape, face_normal
from .groups import Group

#: The words of the anchors, along x', y' and z', for -1, 0 and +1.
_WORDS = (("left", "", "right"), ("front", "", "back"), ("bottom", "", "top"))


@dataclass(frozen=True, eq=False)
class Spot:
    """
    A point that can be picked.

    Attributes:
        position: Where it is, in world coordinates.
        name: What it is, e.g. ``"cup: top"`` or ``"Box 2: bottom front left"``.
    """

    position: np.ndarray
    name: str

    def __repr__(self) -> str:
        return f"Spot {self.name!r} at {_coordinates(self.position, 3)}"


def pickable_points(scene) -> list[Spot]:
    """
    All points of a scene that can be picked, each named after its geometry and anchor.

    A geometry is named by its label if it has one (``scene.add(cup, label="cup")``),
    otherwise by its kind and number (``"Box 1"``, ``"Box 2"``). Shapes offer the anchors
    that lie on them: all 27 of a box, but for a cylinder not the corners of the box around
    it. Lines offer their start, middle and end; groups the points of their members.
    Annotations (angles, dimensions) offer none.
    """
    names = _geometry_names(scene)
    spots: list[Spot] = []
    for item in scene.items:
        for geometry in _members(item.geometry):
            spots += _spots_of(geometry, names[id(geometry)])
    return spots


def nearest(spots: list[Spot], point) -> Spot:
    """The spot nearest to a point in 3D."""
    if not spots:
        raise ValueError("There is nothing to pick.")
    positions = np.array([spot.position for spot in spots])
    return spots[int(np.argmin(np.linalg.norm(positions - np.asarray(point, dtype=float), axis=1)))]


class Measurement:
    """
    Up to two picked spots, A and B, and what follows from them.

    Picking a third time starts a new measurement with that spot as A.
    """

    def __init__(self, digits: int = 3):
        """
        Args:
            digits: Decimals of the coordinates and distances in :meth:`text`.
        """
        self.picks: list[Spot] = []
        self.digits = digits

    def pick(self, spot: Spot) -> None:
        """Add a pick: A, then B; after B, start again with a new A."""
        if len(self.picks) == 2:
            self.picks = []
        self.picks.append(spot)

    def clear(self) -> None:
        self.picks = []

    @property
    def difference(self) -> np.ndarray | None:
        """B minus A: how far to go along x, y and z from A to B (None until both are picked)."""
        if len(self.picks) < 2:
            return None
        return self.picks[1].position - self.picks[0].position

    @property
    def distance(self) -> float | None:
        """The straight distance from A to B (None until both are picked)."""
        difference = self.difference
        return None if difference is None else float(np.linalg.norm(difference))

    def text(self) -> str:
        """The measurement as a few lines of text, e.g. for showing it in a figure."""
        if not self.picks:
            return "click an anchor"
        lines = [
            f"{label}  {spot.name}  {_coordinates(spot.position, self.digits)}"
            for label, spot in zip("AB", self.picks, strict=False)
        ]
        if len(self.picks) == 1:
            lines.append("click a second anchor to measure")
        else:
            lines.append(
                f"|AB| = {self.distance:.{self.digits}f}   Δ = {_coordinates(self.difference, self.digits)}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.text()


# ---------------------------------------------------------------- helpers


def _members(geometry) -> list[Geometry]:
    if isinstance(geometry, Group):
        return [member for part in geometry.geometries for member in _members(part)]
    return [geometry]


def _geometry_names(scene) -> dict[int, str]:
    labels = {}
    for label in scene.labels:
        labels.setdefault(id(label.target), label.text)
    names, counts = {}, {}
    for item in scene.items:
        group_label = labels.get(id(item.geometry)) if isinstance(item.geometry, Group) else None
        for geometry in _members(item.geometry):
            kind = type(geometry).__name__
            counts[kind] = counts.get(kind, 0) + 1
            name = labels.get(id(geometry), f"{kind} {counts[kind]}")
            names[id(geometry)] = f"{group_label}/{name}" if group_label else name
    return names


def _spots_of(geometry: Geometry, name: str) -> list[Spot]:
    if isinstance(geometry, Angle | Dimension):
        return []
    if isinstance(geometry, Point):
        return [Spot(np.asarray(geometry.position, dtype=float), name)]
    if isinstance(geometry, Line | Axis):
        return [Spot(geometry.anchor(where), f"{name}: {where}") for where in ("start", "midway", "end")]
    if isinstance(geometry, Shape):
        return _shape_spots(geometry, name)
    return [Spot(np.asarray(geometry.center, dtype=float), f"{name}: center")]


def _shape_spots(shape: Shape, name: str) -> list[Spot]:
    """The anchors that lie on the shape, the simplest name first where several coincide."""
    local = shape.local_vertices
    size = float(np.ptp(local, axis=0).max()) or 1.0
    on_shape = _solid_test(shape, local, size) if shape.is_closed else _flat_test(shape, local, size)
    spots, seen = [], set()
    directions = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    for direction in sorted(directions, key=lambda d: sum(map(abs, d))):  # center, faces, edges, corners
        point = point_in_bounding_box(local, np.array(direction, dtype=float))
        key = tuple(np.round(point / size, 6))
        if key in seen or not on_shape(point):
            continue
        seen.add(key)
        words = [_WORDS[axis][direction[axis] + 1] for axis in (2, 1, 0)]  # e.g. "top front left"
        anchor = " ".join(word for word in words if word) or "center"
        spots.append(Spot(shape.pose.apply_to_vertex(point), f"{name}: {anchor}"))
    return spots


def _solid_test(shape: Shape, local: np.ndarray, size: float):
    """Inside or on a closed (convex) shape: behind every face's plane."""
    planes = [(local[face][0], face_normal(local[face])) for face in shape.face_indices]
    tolerance = 1e-6 * size
    return lambda point: all(float(np.dot(point - origin, normal)) <= tolerance for origin, normal in planes)


def _flat_test(shape: Shape, local: np.ndarray, size: float):
    """On a flat shape: in the plane of its faces and inside one of them."""
    tolerance = 1e-6 * size
    if np.ptp(local[:, 2]) > tolerance:
        return lambda point: True  # not flat: offer every anchor
    outlines = [local[face][:, :2] for face in shape.face_indices]
    return lambda point: any(_in_polygon(point[:2], outline, tolerance) for outline in outlines)


def _in_polygon(point: np.ndarray, outline: np.ndarray, tolerance: float) -> bool:
    """Whether a 2D point lies inside a polygon or on its edge."""
    x, y = point
    following = np.roll(outline, -1, axis=0)
    for a, b in zip(outline, following, strict=True):  # on an edge?
        edge = b - a
        length = float(edge @ edge)
        t = 0.0 if length == 0 else float(np.clip((point - a) @ edge / length, 0, 1))
        if np.linalg.norm(point - (a + t * edge)) <= tolerance:
            return True
    inside = False
    for (x1, y1), (x2, y2) in zip(outline, following, strict=True):  # crossing number
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _coordinates(vector, digits: int) -> str:
    return "(" + ", ".join(f"{value:.{digits}f}" for value in np.asarray(vector, dtype=float)) + ")"
