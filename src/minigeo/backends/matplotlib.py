"""
Draw a scene with matplotlib's 3D axes.

Every face becomes its own artist. They are drawn in the order of
:func:`minigeo.depth.painter_order`, which is recomputed whenever the view is rotated
with the mouse. Back faces of opaque solids are hidden.

Requires ``pip install "minigeo[mpl]"``.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..depth import is_back_face, painter_order, view_direction
from ..mpl_3d_arrow import Arrow3D
from ..render import Face, Label, Marker, Path, Primitive, bounds
from ..style import color_value

LINE_STYLES = {"solid": "-", "dashed": "--", "dotted": ":"}


def render(scene, ax=None, *, show_axes: bool = True, elev: float | None = None, azim: float | None = None):
    """
    Draw ``scene`` into a 3D matplotlib axes and return the axes.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        ax: A 3D axes (``fig.add_subplot(projection="3d")``). If None, a new figure is made.
        show_axes: Show the x, y, z axes and grid; set False for a clean, TikZ-like drawing.
        elev, azim: Viewing angles in degrees, as in ``ax.view_init``.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    if getattr(ax, "name", None) != "3d":
        raise ValueError('The matplotlib backend needs 3D axes: fig.add_subplot(projection="3d").')

    ax.computed_zorder = False  # we decide the drawing order ourselves, see DepthSorter
    primitives = scene.primitives()
    artists = [draw(ax, primitive) for primitive in primitives]

    fit_view(ax, *bounds(primitives))
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)
    if show_axes:
        ax.set(xlabel="x", ylabel="y", zlabel="z")
    else:
        ax.set_axis_off()

    sorter = DepthSorter(ax, primitives, artists)
    sorter.sort()
    sorter.connect()
    return ax


class DepthSorter:
    """Keeps the artists of a scene in back-to-front order for the current view."""

    def __init__(self, ax, primitives: list[Primitive], artists: list[list]):
        self.ax = ax
        self.primitives = primitives
        self.artists = artists

    def sort(self) -> None:
        """Set the drawing order and hide opaque back faces, for the axes' current angles."""
        toward_viewer = view_direction(self.ax.elev, self.ax.azim)
        for rank, index in enumerate(painter_order(self.primitives, toward_viewer)):
            primitive = self.primitives[index]
            hidden = (
                isinstance(primitive, Face)
                and primitive.closed
                and primitive.style.opacity == 1
                and is_back_face(primitive, toward_viewer)
            )
            for artist in self.artists[index]:
                artist.set_zorder(rank + 1)
                artist.set_visible(not hidden)

    def connect(self) -> None:
        """Re-sort after the user rotates the view with the mouse."""
        canvas = self.ax.figure.canvas

        def on_release(event):
            if event.inaxes is self.ax:
                self.sort()
                canvas.draw_idle()

        canvas.mpl_connect("button_release_event", on_release)


def to_png(scene, *, size: float = 5, dpi: int = 100, **options) -> bytes:
    """Render ``scene`` into PNG bytes, without pyplot (used for display in Jupyter)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(size, size))
    FigureCanvasAgg(figure)
    render(scene, figure.add_subplot(projection="3d"), **options)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


# ---------------------------------------------------------------- one function per primitive


def draw(ax, primitive: Primitive) -> list:
    """Add one primitive to the axes and return the matplotlib artists it created."""
    if isinstance(primitive, Face):
        return draw_face(ax, primitive)
    if isinstance(primitive, Path):
        return draw_path(ax, primitive)
    if isinstance(primitive, Marker):
        return draw_marker(ax, primitive)
    if isinstance(primitive, Label):
        return draw_label(ax, primitive)
    raise TypeError(f"Cannot draw {type(primitive).__name__}.")


def draw_face(ax, face: Face) -> list:
    style = face.style
    # A fully outlined face (e.g. of a box) uses the polygon's own edges. Otherwise the
    # outlined sides are drawn separately, so a cylinder shows no seams between side faces.
    edge_color = color_value(style.stroke) if face.fully_outlined else "none"
    polygon = Poly3DCollection(
        [face.points],
        facecolors=color_value(style.fill),
        edgecolors=edge_color,
        linewidths=style.line_width,
        linestyles=LINE_STYLES[style.dash],
        alpha=style.opacity,
        shade=style.fill != "none",
    )
    ax.add_collection3d(polygon)
    artists = [polygon]
    if face.outline and not face.fully_outlined:
        segments = [[face.points[i], face.points[j]] for i, j in face.outline]
        outline = Line3DCollection(
            segments,
            colors=color_value(style.stroke),
            linewidths=style.line_width,
            linestyles=LINE_STYLES[style.dash],
            alpha=style.opacity,
        )
        ax.add_collection3d(outline)
        artists.append(outline)
    return artists


def draw_path(ax, path: Path) -> list:
    style = path.style
    look = {"color": color_value(style.stroke), "alpha": style.opacity, "linestyle": LINE_STYLES[style.dash]}
    points = path.points
    if not style.arrow:
        return ax.plot(*points.T, linewidth=style.line_width, **look)

    # The line up to the last point, then an arrow along the last segment.
    *body, tip_start, tip_end = points
    artists = ax.plot(*np.vstack([*body, tip_start]).T, linewidth=style.line_width, **look) if body else []
    tip = np.array([tip_start, tip_end]).T
    arrow = Arrow3D(*tip, mutation_scale=15, arrowstyle="-|>", lw=style.line_width, **look)
    ax.add_artist(arrow)
    return [*artists, arrow]


def draw_marker(ax, marker: Marker) -> list:
    style = marker.style
    x, y, z = marker.position
    dot = ax.scatter(
        [x], [y], [z], s=style.marker_size, c=color_value(style.fill), alpha=style.opacity, depthshade=False
    )
    return [dot]


def draw_label(ax, label: Label) -> list:
    style = label.style
    return [
        ax.text(*label.position, label.text, color=color_value(style.text_color), fontsize=style.font_size)
    ]


# ---------------------------------------------------------------- view


def fit_view(ax, low: np.ndarray, high: np.ndarray, padding: float = 0.05) -> None:
    """Show everything between ``low`` and ``high`` with equal scale on all axes."""
    size = high - low
    largest = size.max() if size.max() > 0 else 1.0
    size = np.maximum(size, 0.05 * largest)  # flat scenes still get some depth
    middle = (low + high) / 2
    half = size / 2 * (1 + padding)
    ax.set_xlim(middle[0] - half[0], middle[0] + half[0])
    ax.set_ylim(middle[1] - half[1], middle[1] + half[1])
    ax.set_zlim(middle[2] - half[2], middle[2] + half[2])
    ax.set_box_aspect(size)
