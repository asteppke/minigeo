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

from .geometry import Angle, Axis, Dimension, Geometry, Line, Plane, Point, Polyline, Shape, Vector
from .groups import Group
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
    """
    A line through ``points`` (``(k, 3)``), drawn with ``stroke``; with an arrow tip if ``style.arrow``.

    ``on_top`` paths are painted after all faces by the backends that paint from back to
    front (SVG, matplotlib), e.g. the axes of a coordinate frame starting inside a shape.
    """

    points: np.ndarray
    style: Style
    on_top: bool = False


@dataclass(frozen=True, eq=False)  # holds arrays, so no automatic ==
class Marker:
    """A point, drawn as a dot of ``style.marker_size`` in the ``fill`` color."""

    position: np.ndarray
    style: Style


@dataclass(frozen=True, eq=False)  # holds arrays, so no automatic ==
class Label:
    """
    A text at ``position``, in ``style.text_color`` and ``style.font_size``.

    ``align="left"`` puts the text to the upper right of the point (for labels next to
    things); ``"center"`` centers it on the point (e.g. for dimension texts).
    """

    position: np.ndarray
    text: str
    style: Style
    align: str = "left"


Primitive = Face | Path | Marker | Label


# ---------------------------------------------------------------- geometry -> primitives


@singledispatch
def to_primitives(geometry: Geometry, style: Style) -> list[Primitive]:
    """Turn one geometry into primitives, all drawn with the given complete ``style``."""
    raise TypeError(f"Don't know how to draw {type(geometry).__name__}.")


@to_primitives.register
def _(shape: Shape, style: Style) -> list[Primitive]:
    outline_edges = {frozenset(edge) for edge in shape.edge_indices.tolist()}
    vertices = shape.vertices
    faces = []
    for face in shape.face_indices:
        sides = zip(range(len(face)), [*range(1, len(face)), 0], strict=True)
        outline = tuple((i, j) for i, j in sides if frozenset((face[i], face[j])) in outline_edges)
        faces.append(Face(vertices[face], style, outline, closed=shape.is_closed))
    return faces


@to_primitives.register
def _(line: Line, style: Style) -> list[Primitive]:
    return [Path(line.vertices.copy(), style)]


@to_primitives.register
def _(polyline: Polyline, style: Style) -> list[Primitive]:
    return [Path(polyline.path.copy(), style)]


@to_primitives.register
def _(axis: Axis, style: Style) -> list[Primitive]:
    return [Path(axis.vertices.copy(), style)]


@to_primitives.register
def _(point: Point, style: Style) -> list[Primitive]:
    return [Marker(point.position.astype(float), style)]


@to_primitives.register
def _(angle: Angle, style: Style) -> list[Primitive]:
    primitives: list[Primitive] = []
    if style.fill != "none":  # a shaded sector
        sector = np.vstack([angle.vertex[np.newaxis], angle.arc_points])
        primitives.append(Face(sector, style))
    points = angle.right_angle_points if angle.shows_square else angle.arc_points
    primitives.append(Path(points, style.with_(arrow=False), on_top=True))
    return primitives


@to_primitives.register
def _(dimension: Dimension, style: Style) -> list[Primitive]:
    line = style.with_(arrow=False)
    primitives: list[Primitive] = [Path(points, line, on_top=True) for points in dimension.extension_lines]
    a, b = dimension.line_points
    middle = (a + b) / 2
    arrow = style.with_(arrow=True)
    primitives += [
        Path(np.array([middle, a]), arrow, on_top=True),
        Path(np.array([middle, b]), arrow, on_top=True),
    ]
    primitives.append(Label(dimension.anchor("label"), dimension.text, style, align="center"))
    return primitives


@to_primitives.register
def _(group: Group, style: Style) -> list[Primitive]:
    return [primitive for member in group.geometries for primitive in to_primitives(member, style)]


def type_style(geometry: Geometry | Group) -> Style:
    """Defaults that depend on the kind of geometry, e.g. axes are drawn as arrows."""
    if isinstance(geometry, Axis | Vector):
        return Style(arrow=True)
    if isinstance(geometry, Plane):
        return Style(opacity=0.4)
    if isinstance(geometry, Angle):
        return Style(fill="none", stroke="black")
    if isinstance(geometry, Dimension):
        return Style(stroke="black", line_width=0.8)
    return Style()


def bounds(primitives: list[Primitive]) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest corner of the box around all primitives."""
    points = [p.points for p in primitives if isinstance(p, Face | Path)]
    points += [p.position[np.newaxis] for p in primitives if isinstance(p, Marker | Label)]
    if not points:
        return np.zeros(3), np.zeros(3)
    stacked = np.vstack(points)
    return stacked.min(axis=0), stacked.max(axis=0)
