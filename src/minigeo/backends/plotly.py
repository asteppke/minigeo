"""
Draw a scene with plotly, for interactive viewing in the browser and in Jupyter.

Plotly renders with WebGL, so hidden parts are hidden correctly from every angle and no
drawing order is needed. Faces become ``Mesh3d`` traces (one per opacity, as plotly sets
opacity per trace), outlines and lines become ``Scatter3d`` lines, and arrow tips cones.

Requires ``pip install "minigeo[plotly]"``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from ..camera import AXIS_INDEX, Camera
from ..render import Face, Label, Marker, Path, Primitive, bounds
from ..style import Style, color_value

DASHES = {"solid": "solid", "dashed": "dash", "dotted": "dot"}

#: Distance of the camera from the scene's center, in plotly's scene units.
CAMERA_DISTANCE = 1.8


def render(
    scene,
    *,
    elev: float = 30.0,
    azim: float = -60.0,
    vertical_axis: str = "z",
    show_axes: bool = True,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """
    Draw ``scene`` as an interactive plotly figure and return it.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        elev, azim, vertical_axis: The initial view, as in the other backends.
        show_axes: Show the axes, grid and background; set False for just the scene.
        height, width: Size of the figure in pixels (width None: as wide as available).

    Use ``figure.show()`` to display it, or ``figure.write_html("scene.html")`` to save it.
    """
    primitives = scene.primitives()
    low, high = bounds(primitives)
    arrow_size = 0.04 * max(float(np.max(high - low)), 1e-9)

    traces = (
        face_traces([p for p in primitives if isinstance(p, Face)])
        + outline_traces(primitives)
        + [t for p in primitives if isinstance(p, Path) for t in path_traces(p, arrow_size)]
        + [marker_trace(p) for p in primitives if isinstance(p, Marker)]
        + [label_trace(p) for p in primitives if isinstance(p, Label)]
    )
    figure = go.Figure(data=traces)
    figure.update_layout(
        scene=scene_layout(Camera(elev, azim, vertical_axis), show_axes),
        showlegend=False,
        height=height,
        width=width,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return figure


def scene_layout(camera: Camera, show_axes: bool) -> dict:
    up = np.zeros(3)
    up[AXIS_INDEX[camera.vertical_axis]] = 1
    eye = camera.toward_viewer * CAMERA_DISTANCE
    axis = {"visible": show_axes}
    return {
        "aspectmode": "data",  # equal scale on all axes
        "camera": {
            "eye": dict(zip("xyz", eye.tolist(), strict=True)),
            "up": dict(zip("xyz", up.tolist(), strict=True)),
            "projection": {"type": "orthographic"},  # like the SVG backend and TikZ
        },
        "xaxis": axis,
        "yaxis": axis,
        "zaxis": axis,
    }


# ---------------------------------------------------------------- traces


def face_traces(faces: list[Face]) -> list[go.Mesh3d]:
    """All faces as triangle meshes, one per opacity. Faces are split into triangle fans."""
    by_opacity: dict[float, list[Face]] = defaultdict(list)
    for face in faces:
        if face.style.fill != "none":
            by_opacity[face.style.opacity].append(face)

    traces = []
    for opacity, group in by_opacity.items():
        points, triangles, colors = [], [], []
        for face in group:
            first = sum(len(p) for p in points)
            points.append(face.points)
            for k in range(1, len(face.points) - 1):  # faces are convex, so a fan works
                triangles.append((first, first + k, first + k + 1))
                colors.append(color_value(face.style.fill))
        x, y, z = np.vstack(points).T
        i, j, k = np.array(triangles).T
        traces.append(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                facecolor=colors,
                opacity=opacity,
                flatshading=True,
                hoverinfo="skip",
            )
        )  # fmt: skip
    return traces


def outline_traces(primitives: list[Primitive]) -> list[go.Scatter3d]:
    """The outlines of all faces, one line trace per look; shared edges are drawn once."""
    segments: dict[tuple, dict[tuple, np.ndarray]] = defaultdict(dict)
    for face in primitives:
        if not isinstance(face, Face):
            continue
        sides = face.outline
        look = _line_look(face.style)
        for a, b in sides:
            start, end = face.points[a], face.points[b]
            key = tuple(sorted([tuple(np.round(start, 9)), tuple(np.round(end, 9))]))
            segments[look][key] = np.array([start, end])
    return [_lines_trace(list(group.values()), look) for look, group in segments.items()]


def path_traces(path: Path, arrow_size: float) -> list:
    """A line, plus a cone at its end if it has an arrow tip."""
    look = _line_look(path.style)
    traces = [_lines_trace([path.points], look)]
    if path.style.arrow:
        tip_start, tip_end = path.points[-2], path.points[-1]
        direction = tip_end - tip_start
        length = np.linalg.norm(direction)
        if length > 0:
            u, v, w = direction / length
            color = color_value(path.style.stroke)
            traces.append(
                go.Cone(
                    x=[tip_end[0]], y=[tip_end[1]], z=[tip_end[2]], u=[u], v=[v], w=[w],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=arrow_size,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=path.style.opacity,
                    hoverinfo="skip",
                )
            )  # fmt: skip
    return traces


def marker_trace(marker: Marker) -> go.Scatter3d:
    style = marker.style
    x, y, z = marker.position
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers",
        marker={"size": float(np.sqrt(style.marker_size)), "color": color_value(style.fill)},
        opacity=style.opacity,
        hoverinfo="skip",
    )  # fmt: skip


def label_trace(label: Label) -> go.Scatter3d:
    style = label.style
    x, y, z = label.position
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="text",
        text=[label.text],
        textposition="top right",
        textfont={"color": color_value(style.text_color), "size": style.font_size},
        hoverinfo="skip",
    )  # fmt: skip


def _line_look(style: Style) -> tuple:
    return (color_value(style.stroke), style.line_width, DASHES[style.dash], style.opacity)


def _lines_trace(polylines: list[np.ndarray], look: tuple) -> go.Scatter3d:
    """Several polylines in one trace, separated by gaps (None)."""
    color, width, dash, opacity = look
    coordinates: list[list] = [[], [], []]
    for line in polylines:
        for axis in range(3):
            coordinates[axis] += [*line[:, axis].tolist(), None]
    x, y, z = coordinates
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line={"color": color, "width": width * 2, "dash": dash},
        opacity=opacity,
        hoverinfo="skip",
    )  # fmt: skip
