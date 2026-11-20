"""
Geometries: points, lines, axes and shapes.

The building blocks (transforms and anchors) are re-exported here as well, so that
``import minigeo.geometry as geo`` gives access to everything needed to build a scene.
The old names (``BaseBox``, ``BaseCylinder``, ...) still work but warn; see :mod:`.legacy`.
"""

import warnings

from ..anchors import (
    ANCHOR_WORDS,
    LINE_POSITIONS,
    anchor_direction,
    line_position,
    point_in_bounding_box,
    point_on_segment,
    resolve_pivot,
)
from ..transform import Transform, rotate_arbitrary_vector_matrix, split_rotation
from .annotations import Angle, Dimension
from .base import Geometry
from .plane import Plane
from .primitives import Axis, Line, Point, Vector
from .shape import Shape, face_normal, ring
from .shapes import Box, Circle, Cone, Cylinder, Polygon, Rectangle, Sphere

__all__ = [
    "ANCHOR_WORDS",
    "LINE_POSITIONS",
    "Angle",
    "Axis",
    "Box",
    "Circle",
    "Cone",
    "Cylinder",
    "Dimension",
    "Geometry",
    "Line",
    "Plane",
    "Point",
    "Polygon",
    "Rectangle",
    "Shape",
    "Sphere",
    "Transform",
    "Vector",
    "anchor_direction",
    "face_normal",
    "line_position",
    "point_in_bounding_box",
    "point_on_segment",
    "resolve_pivot",
    "ring",
    "rotate_arbitrary_vector_matrix",
    "split_rotation",
]

#: Old names: those whose arguments changed live in .legacy, the others are just renamed.
_OLD_NAMES = {
    "BaseGeometry": (Geometry, "Geometry"),
    "BaseShape": (Shape, "Shape"),
    "BasePoint": (Point, "Point"),
    "BaseLine": (Line, "Line"),
    "BaseAxis": (Axis, "Axis"),
    "BasePolygon": (Polygon, "Polygon"),
    "BaseBox": (None, "Box(position, size=(width, length, height))"),
    "BaseRectangle": (None, "Rectangle(position, size=(width, length))"),
    "BaseCircle": (None, "Circle(position, radius=...)"),
    "BaseCylinder": (None, "Cylinder(position, radius=..., height=...)"),
    "BaseCone": (None, "Cone(position, radius=..., height=..., top_radius=...)"),
}


_warned: set[str] = set()  # warn once per name; `from ... import` looks names up twice


def __getattr__(name: str):
    if name in _OLD_NAMES:
        new_class, replacement = _OLD_NAMES[name]
        if name not in _warned:
            _warned.add(name)
            warnings.warn(f"{name} is deprecated; use {replacement}.", DeprecationWarning, stacklevel=2)
        if new_class is None:
            from . import legacy

            return getattr(legacy, name)
        return new_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
