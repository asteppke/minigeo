"""
The steps shared by the backends that draw flat pictures (SVG, TikZ).

1. Faces that can never be seen (the back faces of closed, opaque solids) are left out.
2. The rest is projected with the :class:`~minigeo.camera.Camera` and fitted into a picture of
   the requested width.
3. Everything is sorted from back to front (:func:`~minigeo.depth.painter_order`).
4. Faces get a *darkness*: how much they are shaded because they turn away from the light.

A backend then only writes each :class:`Placed` primitive in its own language.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .depth import is_back_face, painter_order
from .geometry import face_normal
from .render import Face, Label, Marker, Path, Primitive

#: How much faces turning away from the light are darkened (0: no shading, 1: up to black).
SHADING_STRENGTH = 0.45


@dataclass(frozen=True, eq=False)
class Placed:
    """
    A primitive with its position in the picture.

    Attributes:
        primitive: The face, path, marker or label.
        points: ``(k, 2)`` picture coordinates: the corners of a face or path, or the single
            position of a marker or label.
        darkness: For faces: from 0 (facing the light) to ``SHADING_STRENGTH`` (turned away).
    """

    primitive: Primitive
    points: np.ndarray
    darkness: float = 0.0


@dataclass
class FlatPicture:
    """A scene projected and sorted for drawing: ``items`` from back to front."""

    width: float
    height: float
    items: list[Placed]


def flatten(
    primitives: list[Primitive],
    camera: Camera,
    *,
    width: float,
    margin: float,
    y_down: bool,
    shading: bool = True,
) -> FlatPicture:
    """
    Project, fit, sort and shade the primitives of a scene.

    Args:
        primitives: Everything to draw, e.g. ``scene.primitives()``.
        camera: Where the viewer looks from.
        width: Width of the picture, in the backend's units (pixels, centimetres, ...).
        margin: Empty space around the drawing, in the same units.
        y_down: Whether the picture's y axis points down (as in SVG) rather than up (TikZ).
        shading: Whether faces get a darkness from the light.
    """
    visible = [p for p in primitives if not _hidden(p, camera)]
    canvas = _Canvas.fitting(visible, camera, width, margin, y_down)
    light = camera.light
    items = []
    for index in painter_order(visible, camera.toward_viewer):
        primitive = visible[index]
        darkness = 0.0
        if shading and isinstance(primitive, Face) and primitive.style.fill != "none":
            darkness = face_darkness(primitive, light)
        items.append(Placed(primitive, canvas.place(_points(primitive)), darkness))
    return FlatPicture(canvas.width, canvas.height, items)


def face_darkness(face: Face, light: np.ndarray) -> float:
    """0 for a face looking straight at the light, up to SHADING_STRENGTH for one turned away."""
    brightness = abs(float(np.dot(face_normal(face.points), light)))  # flat faces have two sides
    return SHADING_STRENGTH * (1 - brightness)


def _hidden(primitive: Primitive, camera: Camera) -> bool:
    """Back faces of closed, opaque solids can never be seen."""
    return (
        isinstance(primitive, Face)
        and primitive.closed
        and primitive.style.opacity == 1
        and is_back_face(primitive, camera.toward_viewer)
    )


def _points(primitive: Primitive) -> np.ndarray:
    if isinstance(primitive, Face | Path):
        return np.asarray(primitive.points, dtype=float)
    if isinstance(primitive, Marker | Label):
        return np.asarray(primitive.position, dtype=float)[np.newaxis]
    raise TypeError(f"Cannot draw {type(primitive).__name__}.")


@dataclass
class _Canvas:
    """Turns world coordinates into picture coordinates."""

    camera: Camera
    scale: float  # picture units per world unit
    low: np.ndarray  # lowest screen corner in world units
    high: np.ndarray
    margin: float
    y_down: bool

    @classmethod
    def fitting(cls, primitives, camera: Camera, width: float, margin: float, y_down: bool) -> _Canvas:
        points = [_points(p) for p in primitives]
        if points:
            xy, _ = camera.project(np.vstack(points))
            low, high = xy.min(axis=0), xy.max(axis=0)
        else:
            low, high = np.zeros(2), np.ones(2)
        size = np.maximum(high - low, 1e-9)
        scale = (width - 2 * margin) / max(size[0], 0.1 * size[1])  # very narrow scenes are not stretched
        return cls(camera, scale, low, high, margin, y_down)

    @property
    def width(self) -> float:
        return (self.high[0] - self.low[0]) * self.scale + 2 * self.margin

    @property
    def height(self) -> float:
        return (self.high[1] - self.low[1]) * self.scale + 2 * self.margin

    def place(self, points: np.ndarray) -> np.ndarray:
        xy, _ = self.camera.project(points)
        x = (xy[:, 0] - self.low[0]) * self.scale + self.margin
        if self.y_down:
            y = (self.high[1] - xy[:, 1]) * self.scale + self.margin
        else:
            y = (xy[:, 1] - self.low[1]) * self.scale + self.margin
        return np.column_stack([x, y])
