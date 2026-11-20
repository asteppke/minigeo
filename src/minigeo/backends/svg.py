"""
Draw a scene as an SVG image (a vector drawing, like a TikZ picture).

The scene is projected with an orthographic :class:`~minigeo.camera.Camera`, the back faces
of opaque solids are left out, and everything else is painted from back to front in the
order of :func:`minigeo.depth.painter_order`. SVG has no lighting, so faces are shaded by
laying a translucent black copy over them, darker the more they turn away from the light.

Requires ``pip install "minigeo[svg]"`` (the `svg.py <https://pypi.org/project/svg.py/>`_ package).
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path as FilePath

import numpy as np
import svg

from ..camera import Camera
from ..depth import is_back_face, painter_order
from ..geometry import face_normal
from ..render import Face, Label, Marker, Path, Primitive
from ..style import Style, color_value

DASH_PATTERNS = {"solid": None, "dashed": (6.0, 4.0), "dotted": (1.5, 3.0)}

#: How much faces turning away from the light are darkened (0: no shading, 1: up to black).
SHADING_STRENGTH = 0.45


@dataclass
class SvgDrawing:
    """The result of the SVG backend: shows itself in Jupyter and can be saved to a file."""

    element: svg.SVG

    def __str__(self) -> str:
        return str(self.element)

    def _repr_svg_(self) -> str:
        return str(self.element)

    def save(self, path: str | FilePath) -> None:
        """Write the drawing to an ``.svg`` file."""
        FilePath(path).write_text(str(self.element), encoding="utf-8")


def render(
    scene,
    *,
    width: float = 500,
    elev: float = 30.0,
    azim: float = -60.0,
    vertical_axis: str = "z",
    margin: float = 20,
    shading: bool = True,
) -> SvgDrawing:
    """
    Draw ``scene`` as SVG.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        width: Width of the image in pixels; the height follows from the scene's shape.
        elev, azim, vertical_axis: Where the viewer looks from, as in the matplotlib backend.
        margin: Empty space around the drawing, in pixels.
        shading: Darken faces that turn away from the light, for a 3D impression.
    """
    camera = Camera(elev, azim, vertical_axis)
    primitives = [p for p in scene.primitives() if not _hidden(p, camera)]
    canvas = _Canvas.fitting(primitives, camera, width, margin)
    painter = _Painter(canvas, camera, shading)

    for index in painter_order(primitives, camera.toward_viewer):
        painter.paint(primitives[index])

    elements = (
        [svg.Defs(elements=list(painter.arrow_markers.values()))] if painter.arrow_markers else []
    ) + painter.elements
    return SvgDrawing(
        svg.SVG(
            width=round(canvas.width),
            height=round(canvas.height),
            viewBox=svg.ViewBoxSpec(0, 0, round(canvas.width), round(canvas.height)),
            elements=elements,
        )
    )


def _hidden(primitive: Primitive, camera: Camera) -> bool:
    """Back faces of closed, opaque solids can never be seen."""
    return (
        isinstance(primitive, Face)
        and primitive.closed
        and primitive.style.opacity == 1
        and is_back_face(primitive, camera.toward_viewer)
    )


@dataclass
class _Canvas:
    """Turns world coordinates into pixel coordinates (y pointing down, as SVG wants)."""

    camera: Camera
    scale: float
    low: np.ndarray  # lowest screen corner in world units
    high: np.ndarray
    margin: float

    @classmethod
    def fitting(cls, primitives: list[Primitive], camera: Camera, width: float, margin: float) -> _Canvas:
        points = [p.points for p in primitives if isinstance(p, Face | Path)]
        points += [p.position[np.newaxis] for p in primitives if isinstance(p, Marker | Label)]
        if points:
            xy, _ = camera.project(np.vstack(points))
            low, high = xy.min(axis=0), xy.max(axis=0)
        else:
            low, high = np.zeros(2), np.ones(2)
        size = np.maximum(high - low, 1e-9)
        scale = (width - 2 * margin) / max(size[0], 0.1 * size[1])  # very narrow scenes are not stretched
        return cls(camera, scale, low, high, margin)

    @property
    def width(self) -> float:
        return (self.high[0] - self.low[0]) * self.scale + 2 * self.margin

    @property
    def height(self) -> float:
        return (self.high[1] - self.low[1]) * self.scale + 2 * self.margin

    def pixels(self, points: np.ndarray) -> np.ndarray:
        xy, _ = self.camera.project(points)
        x = (xy[:, 0] - self.low[0]) * self.scale + self.margin
        y = (self.high[1] - xy[:, 1]) * self.scale + self.margin
        return np.round(np.column_stack([x, y]), 2)


class _Painter:
    """Collects the SVG elements for the primitives, in the order they are painted."""

    def __init__(self, canvas: _Canvas, camera: Camera, shading: bool):
        self.canvas = canvas
        self.shading = shading
        self.light = camera.light
        self.elements: list[svg.Element] = []
        self.arrow_markers: dict[str, svg.Marker] = {}

    def paint(self, primitive: Primitive) -> None:
        if isinstance(primitive, Face):
            self.face(primitive)
        elif isinstance(primitive, Path):
            self.path(primitive)
        elif isinstance(primitive, Marker):
            self.marker(primitive)
        elif isinstance(primitive, Label):
            self.label(primitive)
        else:
            raise TypeError(f"Cannot draw {type(primitive).__name__}.")

    # ------------------------------------------------------------ primitives

    def face(self, face: Face) -> None:
        style = face.style
        corners = self.canvas.pixels(face.points)
        flat = corners.flatten().tolist()
        filled = style.fill != "none"
        outline = _stroke(style) if face.fully_outlined else _seam_cover(style.fill, style.opacity)
        self.elements.append(
            svg.Polygon(
                points=flat,
                fill=color_value(style.fill),
                fill_opacity=style.opacity if filled else None,
                stroke_linejoin="round",
                **outline,
            )
        )
        darkness = self._darkness(face) if filled and self.shading else 0
        if darkness > 0.01:
            shade = round(darkness * style.opacity, 3)
            self.elements.append(
                svg.Polygon(
                    points=flat,
                    fill="black",
                    fill_opacity=shade,
                    **_seam_cover("black", style.opacity, shade),
                )
            )
            if face.fully_outlined:  # keep the outline on top of the shading
                self.elements.append(
                    svg.Polygon(points=flat, fill="none", stroke_linejoin="round", **outline)
                )
        if face.outline and not face.fully_outlined:
            commands = []
            for i, j in face.outline:
                commands += [svg.M(*corners[i]), svg.L(*corners[j])]
            self.elements.append(svg.Path(d=commands, fill="none", stroke_linecap="round", **_stroke(style)))

    def path(self, path: Path) -> None:
        style = path.style
        marker = f"url(#{self._arrow_marker(style.stroke)})" if style.arrow else None
        self.elements.append(
            svg.Polyline(
                points=self.canvas.pixels(path.points).flatten().tolist(),
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
                marker_end=marker,
                **_stroke(style),
            )
        )

    def marker(self, marker: Marker) -> None:
        style = marker.style
        ((x, y),) = self.canvas.pixels(marker.position[np.newaxis])
        self.elements.append(
            svg.Circle(
                cx=x,
                cy=y,
                r=round(np.sqrt(style.marker_size) / 2, 2),
                fill=color_value(style.fill),
                fill_opacity=style.opacity,
            )
        )

    def label(self, label: Label) -> None:
        style = label.style
        ((x, y),) = self.canvas.pixels(label.position[np.newaxis])
        centered = label.align == "center"
        self.elements.append(
            svg.Text(
                x=x,
                y=y,
                dx=0 if centered else 3,
                dy=round(0.35 * style.font_size, 2) if centered else -3,  # vertically centered
                text_anchor="middle" if centered else None,
                text=html.escape(label.text),  # svg.py writes text as it is
                fill=color_value(style.text_color),
                font_size=style.font_size,
                font_family="sans-serif",
            )
        )

    # ------------------------------------------------------------ helpers

    def _darkness(self, face: Face) -> float:
        """0 for a face looking straight at the light, up to SHADING_STRENGTH for one turned away."""
        brightness = abs(float(np.dot(face_normal(face.points), self.light)))  # flat faces have two sides
        return SHADING_STRENGTH * (1 - brightness)

    def _arrow_marker(self, color: str) -> str:
        """An arrow tip in the given color, defined once per color."""
        marker_id = f"arrow-{len(self.arrow_markers)}"
        if color not in self.arrow_markers:
            self.arrow_markers[color] = svg.Marker(
                id=marker_id,
                viewBox=svg.ViewBoxSpec(0, 0, 10, 10),
                refX=9,
                refY=5,
                markerWidth=6,
                markerHeight=6,
                orient="auto-start-reverse",
                elements=[
                    svg.Path(d=[svg.M(0, 0), svg.L(10, 5), svg.L(0, 10), svg.Z()], fill=color_value(color))
                ],
            )
        return self.arrow_markers[color].id


def _seam_cover(color: str, opacity: float, stroke_opacity: float | None = None) -> dict:
    """
    A hairline in the face's own color around opaque faces without an outline.

    Without it, antialiasing leaves faint light seams where neighbouring faces meet, e.g.
    between the side faces of a cylinder. Transparent faces get none, as overlapping
    hairlines would show as darker lines.
    """
    if opacity < 1 or color == "none":
        return {"stroke": "none"}
    return {"stroke": color_value(color), "stroke_width": 0.6, "stroke_opacity": stroke_opacity}


def _stroke(style: Style) -> dict:
    """The SVG stroke attributes for a style."""
    dashes = DASH_PATTERNS[style.dash]
    return {
        "stroke": color_value(style.stroke),
        "stroke_width": style.line_width,
        "stroke_opacity": style.opacity if style.opacity < 1 else None,
        "stroke_dasharray": [d * style.line_width for d in dashes] if dashes else None,
    }
