"""
Where lines meet shapes.

:func:`intersections` (also ``shape.intersect(line)``) returns the points where a line
enters and leaves a shape, sorted along the line. Spheres and cylinders are met exactly,
as the round shapes they describe (also when stretched, as ellipsoids), and planes as
infinite planes. All other shapes are met at their flat faces, exactly as they are drawn.

The line can be a :class:`~minigeo.Line`, :class:`~minigeo.Vector`, :class:`~minigeo.Axis`
or a pair of points; like :meth:`Plane.intersect <minigeo.Plane.intersect>`, it counts as
infinite unless ``segment=True``. A :class:`~minigeo.Polyline` or :class:`~minigeo.Curve`
is followed piece by piece and never extended.
"""

from __future__ import annotations

import numpy as np

from .exact import unit_map
from .geometry import Axis, Line, Plane, Polyline, Shape
from .geometry.mesh import triangulate

#: Hits closer than this (as a fraction of the line's length) count as one point.
SAME_POINT = 1e-9


def intersections(line, shape: Shape, *, segment: bool = False) -> np.ndarray:
    """
    The points where a line meets a shape, sorted along the line.

    Args:
        line: A Line, Vector, Axis, Polyline, Curve, or a pair of points ``(start, end)``.
        shape: Any shape, e.g. a :class:`~minigeo.Box` or a :class:`~minigeo.Mesh`.
        segment: Only points between the line's start and end; otherwise the line is
            extended in both directions. (Polylines and curves are never extended.)

    Returns:
        An ``(n, 3)`` array of points, empty if the line misses the shape. A line through a
        solid usually gives two points: where it enters and where it leaves.

    Example:
        >>> ball = Sphere([0, 0, 0], radius=1)
        >>> intersections(Line([-2, 0, 0], [2, 0, 0]), ball)
        array([[-1.,  0.,  0.],
               [ 1.,  0.,  0.]])
    """
    if isinstance(line, Polyline):
        points = line.path
        hits = [
            _along(start, end, shape, segment=True)
            for start, end in zip(points[:-1], points[1:], strict=True)
        ]
        hits = [hit for hit in hits if len(hit)]
        return _distinct(np.concatenate(hits), points) if hits else np.empty((0, 3))
    start, end = _ends(line)
    return _along(start, end, shape, segment)


def _ends(line) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(line, Line | Axis):
        start, end = line.vertices[0], line.vertices[-1]
    else:
        start, end = line
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if np.allclose(start, end):
        raise ValueError("A line needs two different points to meet a shape.")
    return start, end


def _along(start: np.ndarray, end: np.ndarray, shape: Shape, segment: bool) -> np.ndarray:
    """The points ``start + t * (end - start)`` on the shape, sorted by ``t``."""
    direction = end - start
    mapped = unit_map(shape)
    if isinstance(shape, Plane):  # the whole infinite plane
        t = _plane_t(start, direction, shape)
    elif mapped is None:
        t = _faces_t(start, direction, shape)
    else:  # the line seen from the unit sphere or cylinder; t stays the same
        kind, to_world = mapped
        from_world = np.linalg.inv(to_world)
        origin = from_world[:3, :3] @ start + from_world[:3, 3]
        unit_direction = from_world[:3, :3] @ direction
        t = _sphere_t(origin, unit_direction) if kind == "sphere" else _cylinder_t(origin, unit_direction)
    t = np.sort(t)
    if segment:
        t = t[(t >= -SAME_POINT) & (t <= 1 + SAME_POINT)]
    if len(t):
        t = t[np.concatenate([[True], np.diff(t) > SAME_POINT])]  # a point on a shared edge counts once
    return start + t[:, np.newaxis] * direction


def _plane_t(origin: np.ndarray, direction: np.ndarray, plane: Plane) -> np.ndarray:
    """Where the line crosses an infinite plane (nowhere if it runs parallel to it)."""
    along_normal = float(direction @ plane.normal)
    if abs(along_normal) < 1e-12 * np.linalg.norm(direction):
        return np.empty(0)
    return np.array([float((plane.center - origin) @ plane.normal) / along_normal])


def _faces_t(origin: np.ndarray, direction: np.ndarray, shape: Shape) -> np.ndarray:
    """Where the line crosses the shape's faces (cut into triangles), by Möller and Trumbore."""
    vertices = shape.vertices
    _, triangles = triangulate(shape.face_indices, vertices)
    corners = vertices[triangles]
    a, edge1, edge2 = corners[:, 0], corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    across = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, across)
    scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1) * np.linalg.norm(direction)
    crossing = np.abs(determinant) > 1e-12 * scale  # lines in the plane of a triangle don't cross it
    a, edge1, edge2, across, determinant = (x[crossing] for x in (a, edge1, edge2, across, determinant))

    offset = origin - a
    u = np.einsum("ij,ij->i", offset, across) / determinant
    turned = np.cross(offset, edge1)
    v = (turned @ direction) / determinant
    t = np.einsum("ij,ij->i", edge2, turned) / determinant
    tolerance = 1e-9
    inside = (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1 + tolerance)
    return t[inside]


def _sphere_t(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Where ``origin + t * direction`` meets the unit sphere."""
    return _quadratic(direction @ direction, 2 * origin @ direction, origin @ origin - 1)


def _cylinder_t(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Where ``origin + t * direction`` meets the unit cylinder: its side and its two caps."""
    flat_origin, flat_direction = origin[:2], direction[:2]
    side = _quadratic(
        flat_direction @ flat_direction, 2 * flat_origin @ flat_direction, flat_origin @ flat_origin - 1
    )
    side = side[np.abs(origin[2] + side * direction[2]) <= 0.5 + 1e-12]
    caps = []
    if abs(direction[2]) > 1e-15:
        for height in (-0.5, 0.5):
            t = (height - origin[2]) / direction[2]
            if np.sum((flat_origin + t * flat_direction) ** 2) <= 1 + 1e-12:
                caps.append(t)
    return np.concatenate([side, caps])


def _quadratic(a: float, b: float, c: float) -> np.ndarray:
    """
    The real solutions of ``a t² + b t + c = 0`` for a line and a unit sphere or cylinder
    (one if the line touches it, none if a is 0).
    """
    if a < 1e-30:
        return np.empty(0)
    discriminant = b * b - 4 * a * c
    # A touching line gives 0 only up to rounding, which can make it slightly negative, also
    # when b is 0. c is a squared distance minus 1 (the unit radius), so it rounds by ~1e-16.
    if discriminant < -1e-9 * (b * b + 4 * a * (abs(c) + 1)):
        return np.empty(0)
    root = np.sqrt(max(discriminant, 0.0))
    return np.unique([(-b - root) / (2 * a), (-b + root) / (2 * a)])


def _distinct(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Drop repeats where a polyline's corner lies on the shape (hit by both pieces)."""
    size = float(np.ptp(path, axis=0).max()) or 1.0
    keep = [0]
    for k in range(1, len(points)):
        if np.linalg.norm(points[k] - points[keep[-1]]) > SAME_POINT * size:
            keep.append(k)
    return points[keep]
