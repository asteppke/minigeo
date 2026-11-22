"""
Geometries: points, lines, axes and shapes.

The building blocks (transforms and anchors) are re-exported here as well, so that
``import minigeo.geometry as geo`` gives access to everything needed to build a scene.
"""

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
from .axes import Axes, Grid
from .base import Geometry
from .curves import Curve, Polyline
from .mesh import Mesh
from .plane import Plane
from .primitives import Axis, Line, Point, Vector
from .shape import Shape, face_normal, ring
from .shapes import Box, Circle, Cone, Cylinder, Polygon, Rectangle, Sphere

__all__ = [
    "ANCHOR_WORDS",
    "LINE_POSITIONS",
    "Angle",
    "Axes",
    "Axis",
    "Box",
    "Circle",
    "Cone",
    "Curve",
    "Cylinder",
    "Dimension",
    "Geometry",
    "Grid",
    "Line",
    "Mesh",
    "Plane",
    "Point",
    "Polygon",
    "Polyline",
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
