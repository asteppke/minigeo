"""
A scene: geometries together with how they look, ready to be drawn by a backend.

The geometries stay independent of drawing. A scene only keeps references to them, so
moving a shape after adding it moves it in the next rendering, too.

Example:
    >>> scene = Scene()
    >>> table = scene.add(BaseBox([0, 0, 0], dimensions=[2, 1, 0.1]), fill="peru")
    >>> cup = scene.add(BaseCylinder(table.anchor("top"), dimensions=[0.1, 0.2]), label="cup")
    >>> scene.render()  # with matplotlib; later also "svg" and "plotly"
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TypeVar

import numpy as np

from .geometry import BaseGeometry, BaseShape
from .groups import GeometryGroup
from .render import Label, Path, Primitive, bounds, to_primitives, type_style
from .style import DEFAULT_STYLE, Style

Drawable = TypeVar("Drawable", BaseGeometry, GeometryGroup)

#: Backend names and the modules that implement them; each module has ``render(scene, **options)``.
BACKENDS = {
    "matplotlib": "minigeo.backends.matplotlib",
    "svg": "minigeo.backends.svg",
    "plotly": "minigeo.backends.plotly",
}


FaceKey = int | str  # a face index, or a face name such as "top" for shapes with FACE_NAMES


@dataclass
class SceneItem:
    """One geometry in a scene, with its own style and optional styles for single faces."""

    geometry: BaseGeometry | GeometryGroup
    style: Style
    face_styles: dict[FaceKey, Style] = field(default_factory=dict)


@dataclass
class SceneLabel:
    """A text placed at an anchor of a geometry (or at a fixed point)."""

    target: BaseGeometry | GeometryGroup | np.ndarray
    text: str
    at: str | float | np.ndarray
    style: Style

    @property
    def position(self) -> np.ndarray:
        if isinstance(self.target, BaseGeometry | GeometryGroup):
            return self.target.anchor(self.at)
        return np.asarray(self.target, dtype=float)


@dataclass
class SceneFrame:
    """
    The axes of a coordinate frame, drawn as three arrows with labels.

    For a shape these are its own axes x', y', z' at its center; otherwise the world axes
    x, y, z at a point. Both are looked up when the scene is drawn, so the frame follows a
    shape that moves or turns.
    """

    target: BaseShape | np.ndarray | None
    length: float | None
    labels: tuple[str, str, str]
    colors: tuple[str, str, str]
    style: Style
    on_top: bool

    def origin_axes_length(self) -> tuple[np.ndarray, np.ndarray, float]:
        if isinstance(self.target, BaseShape):
            shape = self.target
            # long enough to stick out of the shape along its longest side
            length = self.length or 0.75 * float(np.ptp(shape.local_vertices, axis=0).max())
            return shape.center, shape.axes, length
        origin = np.zeros(3) if self.target is None else np.asarray(self.target, dtype=float)
        return origin, np.eye(3), self.length or 1.0


class Scene:
    """
    A collection of geometries with styles, to be drawn by a backend.

    Styles are combined from general to specific: the library defaults, then defaults for
    the kind of geometry (axes are arrows), then the scene's ``style``, then each item's own,
    and finally the styles of single faces.
    """

    def __init__(self, style: Style | None = None):
        """
        Args:
            style: Defaults for everything in this scene, e.g. ``Style(opacity=0.8)``.
        """
        self.style = style if style is not None else Style()
        self.items: list[SceneItem] = []
        self.labels: list[SceneLabel] = []
        self.frames: list[SceneFrame] = []

    def add(
        self,
        geometry: Drawable,
        style: Style | None = None,
        *,
        label: str | None = None,
        label_at: str | float | np.ndarray = "center",
        face_styles: dict[FaceKey, Style] | None = None,
        frame: bool = False,
        **style_fields,
    ) -> Drawable:
        """
        Add a geometry and return it, so it can be created and added in one line.

        Args:
            geometry: A shape, line, point, axis or group.
            style: A :class:`Style` for this geometry.
            label: Text to show next to the geometry.
            label_at: The anchor where the label goes, e.g. ``"top"`` or ``"near end"``.
            face_styles: Styles for single faces of a shape, by index or by name
                (for boxes: bottom, top, front, back, left, right).
            frame: Also draw the shape's own axes x', y', z' (see :meth:`add_frame`).
            **style_fields: Shortcuts for single style fields, e.g. ``fill="red"``;
                they win over ``style``.

        Example:
            >>> box = scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="red", opacity=0.5)
            >>> dice = scene.add(BaseBox([2, 0, 0], dimensions=[1, 1, 1]), face_styles={"top": Style(fill="gold")})
        """
        item_style = (style if style is not None else Style()) | Style(**style_fields)
        if face_styles:
            _check_face_keys(geometry, face_styles)
        self.items.append(SceneItem(geometry, item_style, dict(face_styles or {})))
        if label is not None:
            self.labels.append(SceneLabel(geometry, label, label_at, item_style))
        if frame:
            self.add_frame(geometry)
        return geometry

    def add_frame(
        self,
        target: BaseShape | np.ndarray | None = None,
        *,
        length: float | None = None,
        labels: tuple[str, str, str] | None = None,
        colors: tuple[str, str, str] = ("red", "green", "blue"),
        on_top: bool = True,
        style: Style | None = None,
        **style_fields,
    ) -> None:
        """
        Draw the axes of a coordinate frame as three arrows: x red, y green, z blue.

        Args:
            target: A shape, for its own axes x', y', z' at its center; a point, for the
                world axes there; or None for the world axes at the origin.
            length: Length of the arrows. Default: for a shape, long enough to stick out of
                it; for the world axes, 1.
            labels: Names at the arrow tips; default ``x′ y′ z′`` for a shape, ``x y z`` otherwise.
            colors: Colors of the three arrows and their labels.
            on_top: Paint the arrows over all faces in the SVG and matplotlib backends, so
                they stay visible inside the shape. (Plotly always shows true depth.)
            style, **style_fields: Further style, e.g. ``line_width=2``.

        Example:
            >>> scene.add_frame(box)          # the box's own axes, turning with the box
            >>> scene.add_frame(length=0.5)   # the world axes at the origin
        """
        if target is not None and not isinstance(target, BaseShape):
            target = np.asarray(target, dtype=float)
            if target.shape != (3,):
                raise TypeError("A frame needs a shape, a point (x, y, z) or None for the origin.")
        if labels is None:
            labels = ("x′", "y′", "z′") if isinstance(target, BaseShape) else ("x", "y", "z")
        frame_style = (style if style is not None else Style()) | Style(**style_fields)
        self.frames.append(SceneFrame(target, length, tuple(labels), tuple(colors), frame_style, on_top))

    def label(
        self,
        target: BaseGeometry | GeometryGroup | np.ndarray,
        text: str,
        at: str | float | np.ndarray = "center",
        style: Style | None = None,
        **style_fields,
    ) -> None:
        """
        Add a text at an anchor of a geometry, or at a fixed point.

        The position is looked up when the scene is drawn, so the label follows the geometry
        when it moves. A geometry can have any number of labels.

        Example:
            >>> scene.label(cube, "(001)", at="top")
            >>> scene.label([0, 0, 2], "origin is below", text_color="gray")
        """
        label_style = (style if style is not None else Style()) | Style(**style_fields)
        self.labels.append(SceneLabel(target, text, at, label_style))

    def resolved_style(self, item: SceneItem | SceneLabel) -> Style:
        """The complete style an item or label is drawn with."""
        geometry = item.geometry if isinstance(item, SceneItem) else item.target
        return DEFAULT_STYLE | type_style(geometry) | self.style | item.style

    def primitives(self) -> list[Primitive]:
        """Everything in the scene as neutral primitives (faces, paths, markers, labels)."""
        result: list[Primitive] = []
        for item in self.items:
            primitives = to_primitives(item.geometry, self.resolved_style(item))
            if item.face_styles:
                primitives = _restyle_faces(item.geometry, primitives, item.face_styles)
            result.extend(primitives)
        for frame in self.frames:
            result.extend(self._frame_primitives(frame))
        for label in self.labels:
            result.append(Label(label.position, label.text, self.resolved_style(label)))
        return result

    def _frame_primitives(self, frame: SceneFrame) -> list[Primitive]:
        origin, axes, length = frame.origin_axes_length()
        primitives: list[Primitive] = []
        for axis, name, color in zip(axes, frame.labels, frame.colors, strict=True):
            look = Style(arrow=True, line_width=1.5, stroke=color, text_color=color)
            style = DEFAULT_STYLE | self.style | look | frame.style
            tip = origin + axis * length
            primitives.append(Path(np.array([origin, tip]), style, on_top=frame.on_top))
            primitives.append(Label(tip, name, style))
        return primitives

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The lowest and highest corner of the box around everything in the scene."""
        return bounds(self.primitives())

    def render(self, backend: str = "matplotlib", **options):
        """
        Draw the scene with a backend and return what the backend produces.

        Args:
            backend: ``"matplotlib"`` (returns the 3D axes), ``"svg"`` (returns an
                :class:`~minigeo.backends.svg.SvgDrawing`) or ``"plotly"`` (returns a
                plotly figure for interactive viewing).
            **options: Passed on to the backend, e.g. ``elev=20``, ``show_axes=False``.
        """
        return self._backend(backend).render(self, **options)

    def snapshots(self, update: Callable[[int], None], frames: int) -> list[list[Primitive]]:
        """
        Capture the scene as it changes: frame 0 as it is now, then ``update(i)`` before frame ``i``.

        ``update`` changes the geometry, e.g. ``lambda i: box.rotate("z", 10, about="center")``.
        Afterwards the geometry stays as it is in the last frame.
        """
        if frames < 1:
            raise ValueError("An animation needs at least one frame.")
        result = [self.primitives()]
        for i in range(1, frames):
            update(i)
            result.append(self.primitives())
        return result

    def animate(
        self,
        update: Callable[[int], None],
        frames: int,
        backend: str = "plotly",
        **options,
    ):
        """
        Animate the scene: draw it, then call ``update(i)`` before each further frame ``i``.

        Args:
            update: A function that changes the geometry for frame ``i`` (1, 2, ...), e.g.
                turning a box a little: ``lambda i: box.rotate("z'", 10, about="center")``.
                It may move geometry, but should not add, remove or restyle anything.
            frames: The number of frames, including the first one.
            backend: ``"plotly"`` (a figure with play button and slider, runs in the browser)
                or ``"matplotlib"`` (live with ipympl, otherwise an HTML player in Jupyter).
            **options: Passed on to the backend, e.g. ``interval=50`` (milliseconds between
                frames) or the view options ``elev``, ``azim`` and ``vertical_axis``.

        Example:
            >>> scene.animate(lambda i: box.rotate("z", 10, about="center"), frames=36)
        """
        animated = {"plotly", "matplotlib"}
        if backend not in animated:
            raise ValueError(f"Animations work with {' and '.join(sorted(animated))}, not {backend!r}.")
        module = self._backend(backend)
        return module.animate(self.snapshots(update, frames), **options)

    def _backend(self, backend: str):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; available: {', '.join(BACKENDS)}.")
        try:
            return importlib.import_module(BACKENDS[backend])
        except ImportError as error:
            extra = {"matplotlib": "mpl"}.get(backend, backend)
            raise ImportError(f'The {backend} backend needs: pip install "minigeo[{extra}]"') from error

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        """
        Show the scene when it is the last line of a Jupyter cell: as SVG if svg.py is
        installed, otherwise as a matplotlib image.
        """
        if importlib.util.find_spec("svg") is not None:
            return {"image/svg+xml": str(self.render("svg"))}
        if importlib.util.find_spec("matplotlib") is not None:
            from .backends.matplotlib import to_png

            return {"image/png": to_png(self)}
        return {"text/plain": repr(self)}

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"Scene with {len(self.items)} items"


def _face_index(geometry, key: FaceKey) -> int:
    if isinstance(key, str):
        names = getattr(geometry, "FACE_NAMES", ())
        if key not in names:
            raise ValueError(
                f"{type(geometry).__name__} has no face named {key!r}; names: {', '.join(names) or 'none'}."
            )
        return names.index(key)
    return key


def _check_face_keys(geometry, face_styles: dict[FaceKey, Style]) -> None:
    if not isinstance(geometry, BaseShape):
        raise TypeError("face_styles only work for shapes.")
    count = len(geometry.face_indices)
    for key in face_styles:
        if not 0 <= _face_index(geometry, key) < count:
            raise ValueError(f"{type(geometry).__name__} has no face {key!r}.")


def _restyle_faces(shape, primitives: list[Primitive], face_styles: dict[FaceKey, Style]) -> list[Primitive]:
    """Apply face styles; the faces of a shape come out of ``to_primitives`` in face order."""
    restyled = list(primitives)
    for key, face_style in face_styles.items():
        index = _face_index(shape, key)
        face = restyled[index]
        restyled[index] = replace(face, style=face.style | face_style)
    return restyled
