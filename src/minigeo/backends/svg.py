"""
Draw a scene as an SVG image (a vector drawing, like a TikZ picture).

The scene is projected, sorted and shaded by :func:`minigeo.flat.flatten` (shared with the
TikZ backend). SVG has no lighting, so faces are shaded by laying a translucent black copy
over them, darker the more they turn away from the light.

Requires ``pip install "minigeo[svg]"`` (the `svg.py <https://pypi.org/project/svg.py/>`_ package).
Labels with math (``r"$\theta$"``) are drawn as outlines from matplotlib's mathtext, so they
also need ``minigeo[mpl]``.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path as FilePath

import numpy as np
import svg

from ..camera import Camera
from ..flat import Placed, flatten
from ..math_labels import has_math, matplotlib_available, outline, plain, warn_without_matplotlib
from ..render import Face, Label, Marker, Path
from ..style import Style, color_value

DASH_PATTERNS = {"solid": None, "dashed": (6.0, 4.0), "dotted": (1.5, 3.0)}


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
    picture = flatten(scene.primitives(), camera, width=width, margin=margin, y_down=True, shading=shading)
    painter = _Painter()
    for item in picture.items:
        painter.paint(item)

    elements = (
        [svg.Defs(elements=list(painter.arrow_markers.values()))] if painter.arrow_markers else []
    ) + painter.elements
    return SvgDrawing(
        svg.SVG(
            width=round(picture.width),
            height=round(picture.height),
            viewBox=svg.ViewBoxSpec(0, 0, round(picture.width), round(picture.height)),
            elements=elements,
        )
    )


class _Painter:
    """Collects the SVG elements for the primitives, in the order they are painted."""

    def __init__(self):
        self.elements: list[svg.Element] = []
        self.arrow_markers: dict[str, svg.Marker] = {}

    def paint(self, item: Placed) -> None:
        primitive, points = item.primitive, np.round(item.points, 2)
        if isinstance(primitive, Face):
            self.face(primitive, points, item.darkness)
        elif isinstance(primitive, Path):
            self.path(primitive, points)
        elif isinstance(primitive, Marker):
            self.marker(primitive, points[0])
        elif isinstance(primitive, Label):
            self.label(primitive, points[0])
        else:
            raise TypeError(f"Cannot draw {type(primitive).__name__}.")

    # ------------------------------------------------------------ primitives

    def face(self, face: Face, corners: np.ndarray, darkness: float) -> None:
        style = face.style
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

    def path(self, path: Path, points: np.ndarray) -> None:
        style = path.style
        marker = f"url(#{self._arrow_marker(style.stroke)})" if style.arrow else None
        self.elements.append(
            svg.Polyline(
                points=points.flatten().tolist(),
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
                marker_end=marker,
                **_stroke(style),
            )
        )

    def marker(self, marker: Marker, position: np.ndarray) -> None:
        style = marker.style
        x, y = position
        self.elements.append(
            svg.Circle(
                cx=x,
                cy=y,
                r=round(np.sqrt(style.marker_size) / 2, 2),
                fill=color_value(style.fill),
                fill_opacity=style.opacity,
            )
        )

    def label(self, label: Label, position: np.ndarray) -> None:
        style = label.style
        x, y = position
        centered = label.align == "center"
        if has_math(label.text):
            if matplotlib_available():
                self.math_label(label, position)
                return
            warn_without_matplotlib()
            label = Label(label.position, plain(label.text), style, label.align)
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

    def math_label(self, label: Label, position: np.ndarray) -> None:
        """A label with math, drawn as the outlines of its letters (see :mod:`minigeo.math_labels`)."""
        style = label.style
        path = outline(label.text, style.font_size)
        box = path.get_extents()
        x, y = position
        if label.align == "center":  # the middle of the text on the point
            x, y = x - (box.x0 + box.x1) / 2, y + (box.y0 + box.y1) / 2
        else:  # the baseline starts a little up and to the right, like plain labels
            x, y = x + 3, y - 3
        self.elements.append(
            svg.Path(d=_path_commands(path, x, y), fill=color_value(style.text_color), stroke="none")
        )

    # ------------------------------------------------------------ helpers

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


def _path_commands(path, x: float, y: float) -> list:
    """SVG path commands for a matplotlib path, whose y points up, with its origin at (x, y)."""
    kinds = {path.MOVETO: svg.M, path.LINETO: svg.L, path.CURVE3: svg.Q, path.CURVE4: svg.C}
    commands = []
    for vertices, code in path.iter_segments(simplify=False):
        if code == path.CLOSEPOLY:
            commands.append(svg.Z())
            continue
        points = vertices.reshape(-1, 2)
        numbers = [round(value, 2) for px, py in points for value in (x + px, y - py)]
        commands.append(kinds[code](*numbers))
    return commands
