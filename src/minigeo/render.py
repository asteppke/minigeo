"""
The bridge between geometry and backends.

A scene is turned into a flat list of a few neutral *primitives*: faces, paths, markers and
labels, all in 3D world coordinates and each with a complete :class:`Style`. A backend only
needs to know how to draw these four things; it never needs to know what a cylinder is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from .geometry import BaseAxis, BaseGeometry, BaseLine, BasePoint, BaseShape
from .groups import GeometryGroup
from .style import Style


@dataclass(frozen=True, eq=False)  # holds arrays, so no automatic ==
class Face:
    """
    A flat polygon, e.g. one side of a box.

    Attributes:
        points: ``(k, 3)`` corners, counter-clockwise seen from the front (normal points outward).
        style: Its complete style; ``fill`` and ``opacity`` color the face.
        outline: The sides to draw with ``stroke``, as index pairs into ``points``. A box face
            outlines all its sides; a cylinder's side face only its top and bottom.
        closed: Whether the face belongs to a closed solid. Then its back side can never be
            seen, so backends may skip it when it faces away and the solid is opaque.
    """

    points: np.ndarray
    style: Style
    outline: tuple[tuple[int, int], ...] = ()
    closed: bool = False

    @property
    def fully_outlined(self) -> bool:
        return len(self.outline) == len(self.points)


@dataclass(frozen=True, eq=False)  # holds arrays, so no automatic ==
class Path:
    """A line through ``points`` (``(k, 3)``), drawn with ``stroke``; with an arrow tip if ``style.arrow``."""

    points: np.ndarray
    style: Style


@dataclass(frozen=True, eq=False)  # holds arrays, so no automatic ==
class Marker:
    """A point, drawn as a dot of ``style.marker_size`` in the ``fill`` color."""

    position: np.ndarray
    style: Style


@dataclass(frozen=True, eq=False)  # holds arrays, so no automatic ==
class Label:
    """A text at ``position``, in ``style.text_color`` and ``style.font_size``."""

    position: np.ndarray
    text: str
    style: Style


Primitive = Face | Path | Marker | Label


# ---------------------------------------------------------------- geometry -> primitives


@singledispatch
def to_primitives(geometry: BaseGeometry, style: Style) -> list[Primitive]:
    """Turn one geometry into primitives, all drawn with the given complete ``style``."""
    raise TypeError(f"Don't know how to draw {type(geometry).__name__}.")


@to_primitives.register
def _(shape: BaseShape, style: Style) -> list[Primitive]:
    outline_edges = {frozenset(edge) for edge in shape.edge_indices.tolist()}
    vertices = shape.vertices
    faces = []
    for face in shape.face_indices:
        sides = zip(range(len(face)), [*range(1, len(face)), 0], strict=True)
        outline = tuple((i, j) for i, j in sides if frozenset((face[i], face[j])) in outline_edges)
        faces.append(Face(vertices[face], style, outline, closed=shape.is_closed))
    return faces


@to_primitives.register
def _(line: BaseLine, style: Style) -> list[Primitive]:
    return [Path(line.vertices.copy(), style)]


@to_primitives.register
def _(axis: BaseAxis, style: Style) -> list[Primitive]:
    return [Path(axis.vertices.copy(), style)]


@to_primitives.register
def _(point: BasePoint, style: Style) -> list[Primitive]:
    return [Marker(point.position.astype(float), style)]


@to_primitives.register
def _(group: GeometryGroup, style: Style) -> list[Primitive]:
    return [primitive for member in group.geometries for primitive in to_primitives(member, style)]


def type_style(geometry: BaseGeometry | GeometryGroup) -> Style:
    """Defaults that depend on the kind of geometry, e.g. axes are drawn as arrows."""
    if isinstance(geometry, BaseAxis):
        return Style(arrow=True)
    return Style()


def bounds(primitives: list[Primitive]) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest corner of the box around all primitives."""
    points = [p.points for p in primitives if isinstance(p, Face | Path)]
    points += [p.position[np.newaxis] for p in primitives if isinstance(p, Marker | Label)]
    if not points:
        return np.zeros(3), np.zeros(3)
    stacked = np.vstack(points)
    return stacked.min(axis=0), stacked.max(axis=0)
