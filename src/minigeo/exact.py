"""
Exact round shapes for backends that can draw them: spheres and cylinders.

Most backends draw every shape from its flat faces. Renderers that hit surfaces with rays
(the ray tracer, and later e.g. Mitsuba) can draw spheres and cylinders exactly instead, so
they are perfectly round however few segments they have. :func:`exact_shape` decides whether
a geometry can be drawn that way and describes it as an affine image of a *unit shape*:

* the unit sphere: radius 1 around the origin,
* the unit cylinder: radius 1 around the z axis, from z = -1/2 to z = +1/2 (with caps).

``to_world`` maps the unit shape onto the geometry, including any scaling, shear or
mirroring, so a stretched sphere is an exact ellipsoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Cone, Cylinder, Sphere
from .render import Face, Primitive
from .style import Style

KINDS = ("sphere", "cylinder")


@dataclass(frozen=True, eq=False)
class ExactShape:
    """
    A sphere or cylinder to draw exactly.

    Attributes:
        kind: ``"sphere"`` or ``"cylinder"``.
        to_world: ``(4, 4)`` affine map from the unit shape to world coordinates.
        styles: The complete styles of its parts: ``(surface,)`` for a sphere,
            ``(side, bottom, top)`` for a cylinder.
    """

    kind: str
    to_world: np.ndarray
    styles: tuple[Style, ...]

    @property
    def from_world(self) -> np.ndarray:
        """The inverse map: world coordinates to the unit shape."""
        return np.linalg.inv(self.to_world)

    def box_corners(self) -> np.ndarray:
        """The 8 corners of the box around the shape (the unit shape's box, mapped), ``(8, 3)``."""
        height = 1.0 if self.kind == "sphere" else 0.5
        corners = np.array([[x, y, z, 1.0] for x in (-1, 1) for y in (-1, 1) for z in (-height, height)])
        return (corners @ self.to_world.T)[:, :3]

    def reach(self, direction: np.ndarray) -> float:
        """How far the shape extends from its center along a unit ``direction`` (its exact outline)."""
        unit = self.to_world[:3, :3].T @ np.asarray(direction, dtype=float)  # seen from the unit shape
        if self.kind == "sphere":
            return float(np.linalg.norm(unit))
        return float(np.linalg.norm(unit[:2]) + 0.5 * abs(unit[2]))

    @property
    def center(self) -> np.ndarray:
        return self.to_world[:3, 3].copy()

    def rim(self, z: float, points: int = 96) -> np.ndarray:
        """A closed circle of the unit cylinder at height ``z`` (±1/2 for the rims), in world coordinates."""
        angles = np.linspace(0, 2 * np.pi, points + 1)
        unit = np.column_stack(
            [np.cos(angles), np.sin(angles), np.full(len(angles), z), np.ones(len(angles))]
        )
        return (unit @ self.to_world.T)[:, :3]


def unit_map(geometry) -> tuple[str, np.ndarray] | None:
    """
    For a sphere or cylinder (not a cone): its kind and the ``(4, 4)`` affine map from the
    unit shape to world coordinates. None for other geometries, or if the shape was flattened.

    Example:
        >>> kind, to_world = unit_map(Sphere([1, 0, 0], radius=2))
        >>> kind, to_world[:3, 3]
        ('sphere', array([1., 0., 0.]))
    """
    if isinstance(geometry, Sphere):
        kind, unit = "sphere", Sphere([0, 0, 0], 1, segments=geometry.segments)
    elif isinstance(geometry, Cylinder) and not isinstance(geometry, Cone):
        kind, unit = "cylinder", Cylinder([0, 0, 0], 1, 1, segments=geometry.segments)
    else:
        return None
    if len(unit.local_vertices) != len(geometry.local_vertices):
        return None

    # Every transform is affine, so the shape is an affine image of the unit shape.
    unit_vertices = np.column_stack([unit.local_vertices, np.ones(len(unit.local_vertices))])
    solution, *_ = np.linalg.lstsq(unit_vertices, geometry.local_vertices, rcond=None)
    if not np.allclose(unit_vertices @ solution, geometry.local_vertices, atol=1e-9):
        return None
    to_local = np.eye(4)
    to_local[:3, :3] = solution[:3].T
    to_local[:3, 3] = solution[3]
    to_world = geometry.pose.matrix @ to_local
    if abs(np.linalg.det(to_world[:3, :3])) < 1e-15:
        return None  # flattened completely
    return kind, to_world


def exact_shape(geometry, primitives: list[Primitive]) -> ExactShape | None:
    """
    The exact shape for a geometry and its styled faces, or None to draw the faces instead.

    Spheres and cylinders (not cones) qualify, unless their faces are styled in a way an exact
    shape cannot show: side faces of a cylinder in different styles, or faces without fill.

    Args:
        geometry: The geometry, e.g. a :class:`~minigeo.Cylinder`.
        primitives: Its primitives with their complete styles, as the scene produces them
            (face styles of a cylinder: its side faces, then bottom, then top).
    """
    mapped = unit_map(geometry)
    faces = [p for p in primitives if isinstance(p, Face)]
    if mapped is None or len(faces) != len(primitives):
        return None
    kind, to_world = mapped
    if kind == "sphere":
        if len({face.style for face in faces}) != 1:
            return None
        styles = (faces[0].style,)
    else:
        sides = len(geometry.side_face_indices)
        if len({face.style for face in faces[:sides]}) != 1:
            return None  # differently styled side faces: draw the faces themselves
        styles = (faces[0].style, faces[sides].style, faces[sides + 1].style)
    if any(style.fill == "none" for style in styles):
        return None
    return ExactShape(kind, to_world, styles)
