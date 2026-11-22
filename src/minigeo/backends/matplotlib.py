"""
Draw a scene with matplotlib's 3D axes.

Every face becomes its own artist. They are drawn in the order of
:func:`minigeo.depth.painter_order`, which is recomputed whenever the view is rotated
with the mouse. Back faces of opaque solids are hidden.

:func:`measure` adds picking: with an interactive backend (``%matplotlib widget`` from
ipympl in Jupyter, or a desktop window), the mouse snaps to the nearest anchor, and two clicks
measure the distance between anchors.

Requires ``pip install "minigeo[mpl]"``.
"""

from __future__ import annotations

import io
import warnings
import weakref
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection, _generate_normals, _shade_colors

from ..camera import Camera
from ..depth import is_back_face, painter_order
from ..measure import Measurement, Spot, pickable_points
from ..render import Face, Label, Marker, Path, Primitive, bounds
from ..style import color_value

LINE_STYLES = {"solid": "-", "dashed": "--", "dotted": ":"}

#: The Measurer currently attached to each axes, so a new one can detach it (see Measurer.__init__).
_measurers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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


def measure(
    scene,
    ax=None,
    *,
    show_axes: bool = True,
    elev: float | None = None,
    azim: float | None = None,
    vertical_axis: str = "z",
    digits: int = 3,
    radius: float = 10,
) -> Measurer:
    """
    Draw ``scene`` and let the user pick anchors with the mouse and measure between them.

    Needs an interactive backend: ``%matplotlib widget`` (ipympl) in Jupyter, or a desktop
    window. Moving the mouse highlights the nearest anchor and shows its name and
    coordinates; a click picks it. The first pick (A) shows its coordinates, the second (B)
    the distance and the difference along x, y and z; a third pick starts again. Dragging
    still turns the view.

    Args:
        scene, ax, show_axes, elev, azim, vertical_axis: As for :func:`render`.
        digits: Decimals of the coordinates and distances shown.
        radius: How close (in pixels) the mouse must come to an anchor to snap to it.

    Example:
        >>> %matplotlib widget
        >>> tool = scene.measure()
        >>> tool.measurement.distance   # after two clicks
    """
    if _inline_backend():
        warnings.warn(
            "Picking needs an interactive matplotlib backend, e.g. %matplotlib widget (ipympl); "
            "the inline backend shows a still picture.",
            stacklevel=3,
        )
    ax = render(scene, ax, show_axes=show_axes, elev=elev, azim=azim, vertical_axis=vertical_axis)
    return Measurer(ax, pickable_points(scene), digits=digits, radius=radius)


class Measurer:
    """
    Picks anchors in a matplotlib 3D axes and measures between them (made by :func:`measure`).

    Picks can also be made from code with :meth:`pick`, e.g. to prepare a figure.

    Attributes:
        ax: The 3D axes.
        spots: The anchors that can be picked (see :func:`minigeo.measure.pickable_points`).
        measurement: The current picks and what follows from them
            (:class:`minigeo.measure.Measurement`).
    """

    #: Drawn above everything else of the scene.
    ON_TOP = 10**7

    def __init__(self, ax, spots: list[Spot], *, digits: int = 3, radius: float = 10):
        previous = _measurers.get(ax)
        if previous is not None:
            previous.disconnect()  # a Measurer already on this axes would double-pick otherwise
        self.ax = ax
        self.spots = spots
        self.measurement = Measurement(digits)
        self.radius = radius
        self._positions = np.array([spot.position for spot in spots]).reshape(-1, 3)
        self._hovered: Spot | None = None
        self._press: tuple[float, float] | None = None
        self._pick_artists: list = []
        self._hover_artists: list = []
        if spots:
            self.dots = ax.scatter(
                *self._positions.T, s=10, c="black", alpha=0.3, depthshade=False, zorder=self.ON_TOP
            )
        box = {"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85}
        text = {
            "transform": ax.transAxes,
            "ha": "left",
            "family": "monospace",
            "fontsize": 8,
            "zorder": self.ON_TOP + 2,
            "bbox": box,
        }
        self.info = ax.text2D(0.01, 0.99, self.measurement.text(), va="top", **text)
        self.hover_info = ax.text2D(0.01, 0.01, "", va="bottom", **text)
        self.hover_info.set_visible(False)
        canvas = ax.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
        ]
        _measurers[ax] = self

    def disconnect(self) -> None:
        """
        Stop reacting to mouse events and remove the drawn anchors, picks and info boxes.

        Called automatically when a new :class:`Measurer` is attached to the same axes
        (e.g. a second ``scene.measure(ax=ax)``), so the old one doesn't keep picking too.
        """
        canvas = self.ax.figure.canvas
        for cid in self._cids:
            canvas.mpl_disconnect(cid)
        self._cids = []
        for artist in [*self._pick_artists, *self._hover_artists, self.info, self.hover_info]:
            artist.remove()
        self._pick_artists = []
        self._hover_artists = []
        if hasattr(self, "dots"):
            self.dots.remove()
            del self.dots

    @property
    def figure(self):
        return self.ax.figure

    def pick(self, spot: Spot | str) -> Spot:
        """
        Pick a spot, or the spot with this name (e.g. ``"cube: top front left"``), as a click would.
        """
        if isinstance(spot, str):
            matches = [s for s in self.spots if s.name == spot]
            if not matches:
                similar = [s.name for s in self.spots if spot.split(":")[0] in s.name][:6]
                raise KeyError(f"No anchor named {spot!r}; for example: {', '.join(similar) or 'none'}.")
            spot = matches[0]
        self.measurement.pick(spot)
        self._show_picks()
        return spot

    def clear(self) -> None:
        """Forget the picks."""
        self.measurement.clear()
        self._show_picks()

    def spot_at(self, x: float, y: float) -> Spot | None:
        """The anchor nearest to a position on the screen (in pixels), if within ``radius``."""
        if not self.spots:
            return None
        distances = np.linalg.norm(self._on_screen(self._positions) - [x, y], axis=1)
        nearest = int(np.argmin(distances))
        return self.spots[nearest] if distances[nearest] <= self.radius else None

    # ------------------------------------------------------------ mouse

    def _on_press(self, event) -> None:
        if event.inaxes is self.ax and event.button == 1:
            self._press = (event.x, event.y)

    def _on_release(self, event) -> None:
        press, self._press = self._press, None
        if press is None or event.button != 1 or event.x is None:
            return  # not a click: the view was turned, or release was outside the canvas
        if np.hypot(event.x - press[0], event.y - press[1]) > 3:
            return  # not a click: the view was turned
        spot = self.spot_at(event.x, event.y)
        if spot is not None:
            self.pick(spot)

    def _on_motion(self, event) -> None:
        if self._press is not None:
            return  # turning the view
        spot = self.spot_at(event.x, event.y) if event.inaxes is self.ax else None
        if spot is not self._hovered:
            self._hovered = spot
            self._show_hover()

    # ------------------------------------------------------------ drawing

    def _on_screen(self, points: np.ndarray) -> np.ndarray:
        """Pixel positions of 3D points for the axes' current view."""
        xs, ys, _ = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], self.ax.get_proj())
        return self.ax.transData.transform(np.column_stack([xs, ys]))

    def _show_picks(self) -> None:
        for artist in self._pick_artists:
            artist.remove()
        self._pick_artists = []
        picks = self.measurement.picks
        if picks:
            points = np.array([spot.position for spot in picks])
            ax = self.ax
            self._pick_artists.append(
                ax.scatter(*points.T, s=45, c="red", depthshade=False, zorder=self.ON_TOP + 1)
            )
            for label, point in zip("AB", points, strict=False):
                self._pick_artists.append(
                    ax.text(*point, f" {label}", color="red", fontsize=10, zorder=self.ON_TOP + 1)
                )
            if len(picks) == 2:
                (line,) = ax.plot(*points.T, color="red", linewidth=1.5, zorder=self.ON_TOP + 1)
                self._pick_artists.append(line)
        self.info.set_text(self.measurement.text())
        self.figure.canvas.draw_idle()

    def _show_hover(self) -> None:
        for artist in self._hover_artists:
            artist.remove()
        self._hover_artists = []
        spot = self._hovered
        if spot is not None:
            ring = self.ax.scatter(
                *spot.position,
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                depthshade=False,
                zorder=self.ON_TOP + 1,
            )
            self._hover_artists.append(ring)
            x, y, z = spot.position
            digits = self.measurement.digits
            self.hover_info.set_text(f"{spot.name}  ({x:.{digits}f}, {y:.{digits}f}, {z:.{digits}f})")
        self.hover_info.set_visible(spot is not None)
        self.figure.canvas.draw_idle()

    def __repr__(self) -> str:
        return f"Measurer with {len(self.spots)} anchors; {self.measurement.text().splitlines()[0]}"


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

    The other arguments are as in :func:`render`.
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
        """Set the drawing order, hide opaque back faces and reshade faces, for the current angles."""
        camera = Camera(self.ax.elev, self.ax.azim, self.vertical_axis)
        toward_viewer = camera.toward_viewer
        light = _light_source(camera)
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
            if isinstance(primitive, Face) and primitive.style.fill != "none":
                # the light is fixed in world space from the view at render time, so after a
                # rotation it no longer comes from behind the viewer unless recomputed here
                normals = _generate_normals([primitive.points])
                facecolor = _shade_colors(color_value(primitive.style.fill), normals, light)
                self.artists[index][0].set_facecolor(facecolor)

    def connect(self) -> None:
        """Re-sort after the user rotates the view with the mouse."""
        canvas = self.ax.figure.canvas

        def on_release(event):
            # not gated on event.inaxes: a drag that ends outside the axes still rotated the
            # view, and would otherwise leave the zorder and shading stale
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
    with io.BytesIO() as buffer:
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
