"""
Draw a scene with matplotlib's 3D axes.

Every face becomes its own artist. They are drawn in the order of
:func:`minigeo.depth.painter_order`, which is recomputed whenever the view is rotated
with the mouse. Back faces of opaque solids are hidden.

Requires ``pip install "minigeo[mpl]"``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..camera import Camera
from ..depth import is_back_face, painter_order
from ..render import Face, Label, Marker, Path, Primitive, bounds
from ..style import color_value

LINE_STYLES = {"solid": "-", "dashed": "--", "dotted": ":"}


def render(
    scene,
    ax=None,
    *,
    show_axes: bool = True,
    elev: float | None = None,
    azim: float | None = None,
    vertical_axis: str = "z",
):
    """
    Draw ``scene`` into a 3D matplotlib axes and return the axes.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        ax: A 3D axes (``fig.add_subplot(projection="3d")``). If None, a new figure is made.
        show_axes: Show the x, y, z axes and grid; set False for a clean, TikZ-like drawing.
        elev, azim: Viewing angles in degrees, as in ``ax.view_init``. If not given, the
            axes keep their current angles (matplotlib's default is 30 and -60).
        vertical_axis: The axis that points up on the screen: ``"x"``, ``"y"`` or ``"z"``.
    """
    ax, light = _prepare_axes(ax, elev, azim, vertical_axis)
    primitives = scene.primitives()
    artists = [draw(ax, primitive, light) for primitive in primitives]
    _finish_axes(ax, primitives, show_axes)

    sorter = DepthSorter(ax, primitives, artists, vertical_axis)
    sorter.sort()
    sorter.connect()
    return ax


def animate(
    snapshots: list[list[Primitive]],
    ax=None,
    *,
    interval: float = 50,
    repeat: bool = True,
    show_axes: bool = True,
    elev: float | None = None,
    azim: float | None = None,
    vertical_axis: str = "z",
) -> MatplotlibAnimation:
    """
    Play a sequence of scene snapshots as a matplotlib animation.

    With an interactive backend (``%matplotlib widget`` from ipympl in Jupyter, or a desktop
    window) it plays live and the view can be rotated meanwhile. With the default inline
    backend, Jupyter shows it as an HTML player instead. Use
    :meth:`Scene.animate <minigeo.scene.Scene.animate>` rather than calling this directly.

    Args:
        snapshots: The primitives of each frame, see :meth:`Scene.snapshots <minigeo.scene.Scene.snapshots>`.
        interval: Time between frames in milliseconds.
        repeat: Start again after the last frame.
        Other arguments as in :func:`render`.
    """
    from matplotlib.animation import FuncAnimation

    created_here = ax is None
    ax, light = _prepare_axes(ax, elev, azim, vertical_axis)
    _finish_axes(ax, [p for snapshot in snapshots for p in snapshot], show_axes)  # one view for all frames
    sorter = DepthSorter(ax, [], [], vertical_axis)
    sorter.connect()

    def show(frame: int) -> list:
        for artist in (a for group in sorter.artists for a in group):
            artist.remove()
        sorter.primitives = snapshots[frame]
        sorter.artists = [draw(ax, primitive, light) for primitive in sorter.primitives]
        sorter.sort()
        return [a for group in sorter.artists for a in group]

    show(0)
    animation = FuncAnimation(ax.figure, show, frames=len(snapshots), interval=interval, repeat=repeat)
    if created_here and _inline_backend():
        import matplotlib.pyplot as plt

        plt.close(ax.figure)  # otherwise Jupyter also shows the first frame as a still image
    return MatplotlibAnimation(animation, len(snapshots))


@dataclass
class MatplotlibAnimation:
    """
    A matplotlib animation of a scene.

    In Jupyter it shows itself: live with ipympl (``%matplotlib widget``), otherwise as an
    HTML player. ``save("rotation.gif")`` writes it to a file.
    """

    animation: object  # matplotlib.animation.FuncAnimation
    frames: int

    @property
    def figure(self):
        return self.animation._fig

    def to_jshtml(self) -> str:
        """The animation as an HTML player with all frames embedded (works without a kernel)."""
        return self.animation.to_jshtml(default_mode="loop")

    def save(self, path: str, **options) -> None:
        """Save as a ``.gif`` (needs Pillow, which matplotlib installs) or ``.mp4`` (needs ffmpeg)."""
        if str(path).endswith(".gif"):
            options.setdefault("writer", "pillow")
        self.animation.save(path, **options)

    def _repr_html_(self) -> str | None:
        if _live_backend():
            return None  # the live figure shows itself
        return self.to_jshtml()

    def __repr__(self) -> str:
        return f"MatplotlibAnimation with {self.frames} frames"


def _backend_name() -> str:
    import matplotlib

    return matplotlib.get_backend().lower()


def _live_backend() -> bool:
    """ipympl and the classic notebook backend draw live in Jupyter."""
    name = _backend_name()
    return "ipympl" in name or "widget" in name or "nbagg" in name


def _inline_backend() -> bool:
    return "inline" in _backend_name()


def _prepare_axes(ax, elev, azim, vertical_axis):
    """Create or check the 3D axes, set the view and return the axes and a matching light."""
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    if getattr(ax, "name", None) != "3d":
        raise ValueError('The matplotlib backend needs 3D axes: fig.add_subplot(projection="3d").')

    ax.computed_zorder = False  # we decide the drawing order ourselves, see DepthSorter
    elev = ax.elev if elev is None else elev
    azim = ax.azim if azim is None else azim
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
    return ax, _light_source(Camera(elev, azim, vertical_axis))


def _finish_axes(ax, primitives: list[Primitive], show_axes: bool) -> None:
    fit_view(ax, *bounds(primitives))  # after view_init, see fit_view
    if show_axes:
        ax.set(xlabel="x", ylabel="y", zlabel="z")
    else:
        ax.set_axis_off()


class DepthSorter:
    """Keeps the artists of a scene in back-to-front order for the current view."""

    def __init__(self, ax, primitives: list[Primitive], artists: list[list], vertical_axis: str = "z"):
        self.ax = ax
        self.primitives = primitives
        self.artists = artists
        self.vertical_axis = vertical_axis

    def sort(self) -> None:
        """Set the drawing order and hide opaque back faces, for the axes' current angles."""
        toward_viewer = Camera(self.ax.elev, self.ax.azim, self.vertical_axis).toward_viewer
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


def to_figure(scene, *, size: float = 5, **options) -> Figure:
    """Render ``scene`` into a new figure without pyplot, e.g. for saving to a file."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(size, size))
    FigureCanvasAgg(figure)
    render(scene, figure.add_subplot(projection="3d"), **options)
    return figure


def to_png(scene, *, size: float = 5, dpi: int = 100, **options) -> bytes:
    """Render ``scene`` into PNG bytes, without pyplot (used for display in Jupyter)."""
    figure = to_figure(scene, size=size, **options)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


# ---------------------------------------------------------------- one function per primitive


def _light_source(camera: Camera) -> LightSource:
    """The same light as in the SVG backend, so shading looks alike whatever the vertical axis."""
    x, y, z = camera.light
    return LightSource(azdeg=90 - np.degrees(np.arctan2(y, x)), altdeg=np.degrees(np.arcsin(z)))


def draw(ax, primitive: Primitive, light: LightSource | None = None) -> list:
    """Add one primitive to the axes and return the matplotlib artists it created."""
    if isinstance(primitive, Face):
        return draw_face(ax, primitive, light)
    if isinstance(primitive, Path):
        return draw_path(ax, primitive)
    if isinstance(primitive, Marker):
        return draw_marker(ax, primitive)
    if isinstance(primitive, Label):
        return draw_label(ax, primitive)
    raise TypeError(f"Cannot draw {type(primitive).__name__}.")


def draw_face(ax, face: Face, light: LightSource | None = None) -> list:
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
        lightsource=light,
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
    arrow = _Arrow3D(*tip, mutation_scale=15, arrowstyle="-|>", lw=style.line_width, **look)
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
    placement = {"ha": "center", "va": "center"} if label.align == "center" else {}
    text = ax.text(
        *label.position,
        label.text,
        color=color_value(style.text_color),
        fontsize=style.font_size,
        **placement,
    )
    return [text]


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
    # call after ax.view_init: set_box_aspect accounts for the vertical axis set there
    ax.set_box_aspect(size)


class _Arrow3D(FancyArrowPatch):
    """A 2D arrow that follows two 3D points; from https://github.com/matplotlib/matplotlib/issues/22571."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
