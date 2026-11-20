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
from .base import BaseGeometry
from .primitives import BaseAxis, BaseLine, BasePoint
from .shape import BaseShape, face_normal, ring
from .shapes import BaseBox, BaseCircle, BaseCone, BaseCylinder, BasePolygon, BaseRectangle

__all__ = [
    "ANCHOR_WORDS",
    "LINE_POSITIONS",
    "BaseAxis",
    "BaseBox",
    "BaseCircle",
    "BaseCone",
    "BaseCylinder",
    "BaseGeometry",
    "BaseLine",
    "BasePoint",
    "BasePolygon",
    "BaseRectangle",
    "BaseShape",
    "Transform",
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
