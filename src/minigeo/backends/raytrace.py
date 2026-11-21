"""
Draw a scene by ray tracing: lit faces, cast shadows and exact spheres and cylinders.

For every pixel, rays go from the viewer into the scene (several per pixel, against jagged
edges). Where a ray hits a surface, the surface is lit by a distant light from the upper
left, behind the viewer (Lambert shading), unless another surface is in the way: then it lies
in shadow. Transparent faces let the ray go on and are blended with what lies behind them.

Faces are split into triangles. Spheres and cylinders are hit exactly instead, so they are
perfectly round, also after scaling (as ellipsoids and elliptic cylinders).

Lines, points and labels are drawn on top of the rendered surfaces, but only where no surface
lies in front of them: the back edges of a box stay hidden. Paths marked ``on_top`` (such as
coordinate frames) are always drawn, and a label is only hidden by *other* objects than the
one it belongs to.

A *look* changes how surfaces are shaded, for diagrams rather than photos: cel shading in a
few bands, ambient occlusion that darkens corners and creases, outlines around objects and
along sharp edges, a reduced color palette, or gray tones for print. See :class:`Look` and
:data:`LOOKS`.

The camera is orthographic, as in the other backends, so the same ``elev``, ``azim`` and
``vertical_axis`` give the same picture. Line widths, text and margins are measured in pixels
of a 500-pixel-wide image: a wider image has more pixels, but looks the same.

Requires ``pip install "minigeo[ray]"`` (`Pillow <https://pypi.org/project/pillow/>`_, for text
and image files).
"""

from __future__ import annotations

import copy
import io
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path as FilePath

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, PngImagePlugin, features

from ..camera import Camera
from ..colors import to_rgb
from ..exact import ExactShape, exact_shape
from ..geometry import Geometry
from ..groups import Group
from ..render import Face, Label, Marker, Path, Primitive, to_primitives
from ..style import Style, color_value
from ..transform import Transform

#: Line widths, text sizes and margins are given in pixels of an image this wide.
REFERENCE_WIDTH = 500

#: How bright surfaces are without direct light (in shadow, or turned away from the light).
AMBIENT = 0.4

#: Dash and gap lengths, in multiples of the line width (as in the SVG backend).
DASH_PATTERNS = {"solid": None, "dashed": (6.0, 4.0), "dotted": (1.5, 3.0)}

#: At most this many transparent surfaces behind each other are blended.
MAX_LAYERS = 8

#: Points per circle for the outline of an exact cylinder.
CIRCLE_POINTS = 96

#: The size of an arrow tip, in multiples of the line width (as in the SVG backend).
ARROW_LENGTH = 6.0


SHADINGS = ("smooth", "cel", "flat")


@dataclass(frozen=True)
class Look:
    """
    How surfaces are shaded. Use one of the :data:`LOOKS` by name, or change single settings:
    ``scene.render("raytrace", look="diagram", bands=4)``.

    Attributes:
        shading: ``"smooth"`` (light falls off gradually), ``"cel"`` (a few flat bands, as in
            comics and technical illustrations) or ``"flat"`` (no light, only the colors).
        bands: Brightness steps for cel shading (at least 2).
        shadows: Whether surfaces cast shadows.
        ambient: Brightness without direct light, from 0 (black) to 1 (no shading).
        occlusion: Ambient occlusion, from 0 (off) to 1: darkens corners, creases and places
            close to other surfaces, where less light from the surroundings arrives.
        occlusion_radius: How far surfaces darken each other, as a fraction of the scene's size.
        occlusion_samples: Rays per point; more is smoother but slower. With the numba engine
            they go in random directions per point (rounded to a square number), with numpy
            in a fixed set of directions. None: 64 random rays with numba, 24 fixed
            directions with numpy (about equally fast, the random ones without rings).
        outlines: Draw lines around objects and along sharp edges (from the rendered picture,
            so they follow exact spheres and cylinders too).
        outline_color: Color of the outlines.
        outline_width: Width of the outlines, in pixels of a 500-pixel-wide image.
        crease_angle: Edges where the surface bends by more than this angle (degrees) are outlined.
        palette: Colors that surface colors are replaced with (each with the nearest one), for a
            restrained, consistent set of colors; None keeps the colors.
        grayscale: Turn the finished picture into gray tones, e.g. for printing.
    """

    shading: str = "smooth"
    bands: int = 3
    shadows: bool = True
    ambient: float = AMBIENT
    occlusion: float = 0.0
    occlusion_radius: float = 0.15
    occlusion_samples: int | None = None
    outlines: bool = False
    outline_color: str = "black"
    outline_width: float = 1.5
    crease_angle: float = 40.0
    palette: tuple[str, ...] | None = None
    grayscale: bool = False

    def __post_init__(self):
        if self.shading not in SHADINGS:
            raise ValueError(f"Unknown shading {self.shading!r}; use one of {', '.join(SHADINGS)}.")
        if self.bands < 2:
            raise ValueError("Cel shading needs at least 2 bands.")
        for name in ("ambient", "occlusion"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if (self.occlusion_samples is not None and self.occlusion_samples < 1) or self.occlusion_radius <= 0:
            raise ValueError("occlusion_samples and occlusion_radius must be positive.")
        if self.palette is not None:
            object.__setattr__(self, "palette", tuple(self.palette))
            if not self.palette:
                raise ValueError("A palette needs at least one color.")

    def with_(self, **changes) -> Look:
        """A copy with some settings changed."""
        known = {f.name for f in fields(self)}
        unknown = sorted(set(changes) - known)
        if unknown:
            raise TypeError(
                f"Unknown look settings: {', '.join(unknown)}; known: {', '.join(sorted(known))}."
            )
        return replace(self, **changes)


#: Ready-made looks, by name.
LOOKS: dict[str, Look] = {
    "natural": Look(),
    "soft": Look(shadows=False, ambient=0.55, occlusion=0.8),
    "cel": Look(shading="cel", outlines=True),
    "diagram": Look(shading="cel", shadows=False, ambient=0.55, occlusion=0.6, outlines=True),
    "print": Look(shading="cel", shadows=False, ambient=0.6, occlusion=0.6, outlines=True, grayscale=True),
}


@dataclass
class RaytracedImage:
    """
    The result of the ray tracing backend: an RGBA image.

    It shows itself in Jupyter and can be saved in any format Pillow knows, e.g. PNG or JPEG.

    Attributes:
        pixels: ``(height, width, 4)`` array of 8-bit red, green, blue and alpha values.
    """

    pixels: np.ndarray

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    def to_pil(self) -> Image.Image:
        """The image as a Pillow image."""
        return Image.fromarray(self.pixels)

    def to_png(self) -> bytes:
        """The image as PNG file contents."""
        buffer = io.BytesIO()
        self.to_pil().save(buffer, format="PNG")
        return buffer.getvalue()

    def _repr_png_(self) -> bytes:
        return self.to_png()

    def save(self, path: str | FilePath) -> None:
        """
        Write the image to a file; the extension chooses the format (``.png``, ``.jpg``, ...).

        Formats without transparency (such as JPEG) get a white background.
        """
        path = FilePath(path)
        image = self.to_pil()
        if path.suffix.lower() in {".jpg", ".jpeg", ".bmp"}:
            image = _on_white(image)
        image.save(path)


@dataclass
class RaytracedAnimation:
    """
    The result of a ray traced animation: a list of RGBA frames.

    It plays in Jupyter (as an animated PNG, which browsers show like any image) and can be
    saved as an animated PNG, GIF or WebP.

    Attributes:
        frames: One ``(height, width, 4)`` array of 8-bit RGBA values per frame.
        interval: Time between frames, in milliseconds.
        repeat: Whether the animation plays in a loop.
    """

    frames: list[np.ndarray]
    interval: float = 80
    repeat: bool = True

    def __len__(self) -> int:
        return len(self.frames)

    def to_png(self) -> bytes:
        """The animation as the contents of an animated PNG (APNG) file."""
        buffer = io.BytesIO()
        self._write(buffer, "PNG")
        return buffer.getvalue()

    def _repr_png_(self) -> bytes:
        return self.to_png()

    def save(self, path: str | FilePath) -> None:
        """
        Write the animation to a file; the extension chooses the format.

        * ``.png`` or ``.apng``: an animated PNG, sharp and with transparency
        * ``.gif``: a GIF, on a white background (GIF has no smooth transparency)
        * ``.webp``: an animated WebP, small and lossless (if Pillow supports WebP)
        """
        path = FilePath(path)
        formats = {".png": "PNG", ".apng": "PNG", ".gif": "GIF", ".webp": "WEBP"}
        suffix = path.suffix.lower()
        if suffix not in formats:
            raise ValueError(f"Cannot save an animation as {path.name}: use .png, .apng, .gif or .webp.")
        with open(path, "wb") as file:
            self._write(file, formats[suffix])

    def _write(self, file, format: str) -> None:
        images = [Image.fromarray(frame) for frame in self.frames]
        options = {"save_all": True, "append_images": images[1:], "duration": self.interval}
        if format == "PNG":
            # each frame replaces the previous one, also where it is transparent
            options |= {
                "disposal": PngImagePlugin.Disposal.OP_BACKGROUND,
                "blend": PngImagePlugin.Blend.OP_SOURCE,
            }
            options["loop"] = 0 if self.repeat else 1
        elif format == "GIF":
            images = [_on_white(image) for image in images]
            options |= {"append_images": images[1:], "disposal": 2}
            if self.repeat:
                options["loop"] = 0
        else:
            if not features.check("webp"):
                raise RuntimeError("This Pillow cannot write WebP; save as .png or .gif instead.")
            options |= {"lossless": True, "loop": 0 if self.repeat else 1}
        images[0].save(file, format=format, **options)


def _on_white(image: Image.Image) -> Image.Image:
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def render(
    scene,
    *,
    width: int = 500,
    elev: float = 30.0,
    azim: float = -60.0,
    vertical_axis: str = "z",
    margin: float = 20,
    samples: int = 3,
    background: str | None = None,
    look: str | Look = "natural",
    engine: str = "auto",
    **look_changes,
) -> RaytracedImage:
    """
    Draw ``scene`` by ray tracing.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        width: Width of the image in pixels; the height follows from the scene's shape.
        elev, azim, vertical_axis: Where the viewer looks from, as in the other backends.
        margin: Empty space around the drawing, in pixels of a 500-pixel-wide image.
        samples: Rays per pixel along each direction: 3 traces 3 x 3 rays per pixel for
            smooth edges; 1 is fastest.
        background: A background color, or None for a transparent background.
        look: How surfaces are shaded: a name from :data:`LOOKS` (``"natural"``, ``"soft"``,
            ``"cel"``, ``"diagram"``, ``"print"``) or a :class:`Look`.
        engine: How rays are cast: ``"numba"`` (compiled and parallel, much faster; needs
            ``pip install "minigeo[fast]"``), ``"numpy"``, or ``"auto"`` (numba if installed).
        **look_changes: Single settings of the look, e.g. ``shadows=False``, ``ambient=0.6``
            or ``bands=4`` (see :class:`Look`).

    Example:
        >>> image = scene.render("raytrace", width=800)
        >>> image.save("scene.png")
        >>> scene.render("raytrace", look="diagram", outline_width=2)
    """
    camera = Camera(elev, azim, vertical_axis)
    look = _resolve_look(look, look_changes)
    engine = _resolve_engine(engine)
    return RaytracedImage(_draw(scene, camera, None, width, margin, samples, background, look, engine))


def animate(
    scene,
    update: Callable[[int], None],
    frames: int,
    *,
    interval: float = 80,
    repeat: bool = True,
    width: int = 400,
    elev: float = 30.0,
    azim: float = -60.0,
    vertical_axis: str = "z",
    margin: float = 20,
    samples: int = 2,
    background: str | None = None,
    look: str | Look = "natural",
    engine: str = "auto",
    **look_changes,
) -> RaytracedAnimation:
    """
    Ray trace an animation: the scene as it is, then after ``update(i)`` for each further frame.

    All frames share one view, fitted around everything the scene covers in any frame, so the
    picture does not jump while things move. Every frame is traced, so the defaults are a
    little smaller and coarser than for :func:`render`; about half a second per frame.
    Usually called through ``scene.animate(update, frames, backend="raytrace")``.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to animate. Afterwards it stays as it is
            in the last frame.
        update: Changes the geometry for frame ``i`` (1, 2, ...).
        frames: The number of frames, including the first one.
        interval: Time between frames, in milliseconds.
        repeat: Whether the animation plays in a loop.
        width, elev, azim, vertical_axis, margin, samples, background, look, engine: As for
            :func:`render`; further keywords change single settings of the look, as there.

    Example:
        >>> movie = scene.animate(lambda i: box.rotate("z", 10, about="center"), frames=36,
        ...                       backend="raytrace", look="diagram")
        >>> movie.save("turning.png")   # an animated PNG
    """
    if frames < 1:
        raise ValueError("An animation needs at least one frame.")
    camera = Camera(elev, azim, vertical_axis)
    look = _resolve_look(look, look_changes)
    engine = _resolve_engine(engine)
    states = [copy.deepcopy(scene)]
    for i in range(1, frames):
        update(i)
        states.append(copy.deepcopy(scene))
    everywhere = np.vstack([_Contents.collect(state).points(camera) for state in states])
    pictures = [
        _draw(state, camera, everywhere, width, margin, samples, background, look, engine) for state in states
    ]
    return RaytracedAnimation(pictures, interval, repeat)


ENGINES = ("auto", "numpy", "numba")


def _resolve_engine(engine: str) -> str:
    """``"numba"`` or ``"numpy"``: numba for ``"auto"`` if it is installed."""
    from . import _fastcast

    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; use one of {', '.join(ENGINES)}.")
    if engine == "auto":
        return "numba" if _fastcast.available() else "numpy"
    if engine == "numba" and not _fastcast.available():
        raise ImportError('The numba engine needs numba: pip install "minigeo[fast]"')
    return engine


def _resolve_look(look: str | Look, changes: dict) -> Look:
    if isinstance(look, str):
        if look not in LOOKS:
            raise ValueError(f"Unknown look {look!r}; use one of {', '.join(LOOKS)} or a Look.")
        look = LOOKS[look]
    return look.with_(**changes)


def _draw(
    scene,
    camera: Camera,
    fit: np.ndarray | None,
    width,
    margin,
    samples,
    background,
    look: Look,
    engine="numpy",
) -> np.ndarray:
    """Trace one picture. ``fit`` are the points the view is fitted to; None: the scene's own."""
    if samples < 1:
        raise ValueError("samples must be at least 1.")
    contents = _Contents.collect(scene)
    points = contents.points(camera) if fit is None else fit
    view = _View.fitting(points, camera, width, margin * width / REFERENCE_WIDTH, samples)
    surfaces = _Surfaces(contents.surfaces, view.size, engine)
    if look.palette is not None:
        surfaces.colors = _nearest_colors(surfaces.colors, look.palette)
    lighting = _Lighting(surfaces, camera.light, look, view.size)

    layers = _trace_layers(surfaces, lighting, view)
    canvas = _Canvas(layers, view)
    if look.outlines:
        canvas.outline(layers, look)
    for line in contents.lines:
        canvas.stroke(line)
    for marker in contents.markers:
        canvas.dot(marker)
    image = canvas.downsampled(background)
    _draw_labels(image, contents.labels, canvas, view)
    pixels = np.asarray(image)
    return _gray(pixels) if look.grayscale else pixels


def _nearest_colors(colors: np.ndarray, palette: tuple[str, ...]) -> np.ndarray:
    """Each color replaced with the nearest color of the palette."""
    choices = np.array([_rgb(color) for color in palette])
    distances = np.linalg.norm(colors[..., None, :] - choices, axis=-1)
    return choices[np.argmin(distances, axis=-1)]


def _gray(pixels: np.ndarray) -> np.ndarray:
    """The image in gray tones (by luminance), keeping transparency."""
    luminance = pixels[..., :3].astype(float) @ np.array([0.2126, 0.7152, 0.0722])
    gray = np.round(luminance).astype(np.uint8)
    return np.dstack([gray, gray, gray, pixels[..., 3]])


# ---------------------------------------------------------------- what the scene contains


@dataclass
class _Surface:
    """Something rays can hit: a flat face, or an exact sphere or cylinder."""

    kind: str  # "face", "sphere" or "cylinder"
    owner: int  # index of the scene item, or -1
    piece: int = -1  # which geometry it belongs to (members of a group are different pieces)
    points: np.ndarray | None = None  # face corners
    style: Style | None = None  # face style
    exact: ExactShape | None = None  # sphere/cylinder, see minigeo.exact

    @property
    def to_world(self) -> np.ndarray | None:
        return None if self.exact is None else self.exact.to_world

    @property
    def part_styles(self) -> tuple[Style, ...]:
        return () if self.exact is None else self.exact.styles


@dataclass
class _Line:
    points: np.ndarray  # (k, 3)
    style: Style
    on_top: bool


@dataclass
class _Label:
    label: Label
    owners: frozenset[int]


@dataclass
class _Contents:
    """The scene sorted into surfaces, lines, points and labels, each knowing its scene item."""

    surfaces: list[_Surface]
    lines: list[_Line]
    markers: list[Marker]
    labels: list[_Label]
    extra_points: list[np.ndarray]
    _piece_count: int = 0

    @classmethod
    def collect(cls, scene) -> _Contents:
        contents = cls([], [], [], [], [])
        owners: dict[int, int] = {}  # id(geometry) -> scene item index
        outlines: set = set()  # outline segments already drawn, as rounded end points
        for index, item in enumerate(scene.items):
            style = scene.resolved_style(item)
            for geometry, primitives in _pieces(item.geometry, style, item.face_styles):
                owners[id(geometry)] = index
                contents._add(geometry, primitives, index, outlines)
            owners[id(item.geometry)] = index
        for frame in scene.frames:
            contents._add(None, scene._frame_primitives(frame), -1, outlines)
        for scene_label in scene.labels:
            align = getattr(scene_label.target, "label_align", "left")
            label = Label(scene_label.position, scene_label.text, scene.resolved_style(scene_label), align)
            contents.labels.append(_Label(label, frozenset(_owners_of(scene_label.target, owners))))
        return contents

    def _add(self, geometry, primitives: list[Primitive], owner: int, outlines: set) -> None:
        self._piece_count += 1
        exact = exact_shape(geometry, primitives)
        if exact is not None:
            self.surfaces.append(_Surface(exact.kind, owner, self._piece_count, exact=exact))
            for z in (-0.5, 0.5) if exact.kind == "cylinder" else ():
                self.lines.append(_Line(exact.rim(z, CIRCLE_POINTS), exact.styles[0], False))
            return
        for primitive in primitives:
            if isinstance(primitive, Face):
                self._add_face(primitive, owner, outlines)
            elif isinstance(primitive, Path):
                self.lines.append(
                    _Line(np.asarray(primitive.points, float), primitive.style, primitive.on_top)
                )
            elif isinstance(primitive, Marker):
                self.markers.append(primitive)
            elif isinstance(primitive, Label):
                self.labels.append(_Label(primitive, frozenset({owner})))

    def _add_face(self, face: Face, owner: int, outlines: set) -> None:
        style = face.style
        if style.fill != "none" and style.opacity > 0:
            points = np.asarray(face.points, float)
            self.surfaces.append(_Surface("face", owner, self._piece_count, points=points, style=style))
        else:
            self.extra_points.append(face.points)
        if style.stroke == "none":
            return
        for i, j in face.outline:  # each edge once, also where two faces share it
            a, b = face.points[i], face.points[j]
            key = (*sorted([tuple(np.round(a, 9)), tuple(np.round(b, 9))]), style.stroke, style.line_width)
            if key not in outlines:
                outlines.add(key)
                self.lines.append(_Line(np.array([a, b], dtype=float), style, False))

    def points(self, camera: Camera) -> np.ndarray:
        """All points that should fit into the picture, seen with ``camera``."""
        points = [s.points for s in self.surfaces if s.points is not None] + self.extra_points
        points += [_outline_points(s.exact, camera) for s in self.surfaces if s.exact is not None]
        points += [line.points for line in self.lines]
        points += [m.position[np.newaxis] for m in self.markers]
        points += [entry.label.position[np.newaxis] for entry in self.labels]
        return np.vstack(points) if points else np.zeros((1, 3))


def _pieces(geometry, style: Style, face_styles):
    """The geometries of a scene item with their primitives; groups are split into members."""
    if isinstance(geometry, Group):
        for member in geometry.geometries:
            yield from _pieces(member, style, None)
        return
    from ..scene import _restyle_faces  # here, to avoid a circular import

    primitives = to_primitives(geometry, style)
    if face_styles:
        primitives = _restyle_faces(geometry, primitives, face_styles)
    yield geometry, primitives


def _owners_of(target, owners: dict[int, int]) -> set[int]:
    if isinstance(target, Group):
        found = {owners[id(target)]} if id(target) in owners else set()
        for member in target.geometries:
            found |= _owners_of(member, owners)
        return found
    if isinstance(target, Geometry) and id(target) in owners:
        return {owners[id(target)]}
    return set()


def _outline_points(exact: ExactShape, camera: Camera) -> np.ndarray:
    """
    Points at the exact outline of a sphere or cylinder: its farthest points to the right,
    left, top, bottom, front and back of the screen.
    """
    points = []
    for direction in (camera.right, camera.up, camera.toward_viewer):
        reach = exact.reach(direction)
        points += [exact.center + reach * direction, exact.center - reach * direction]
    return np.array(points)


def _rgb(color: str) -> np.ndarray:
    return np.array(to_rgb(color), dtype=float)


# ---------------------------------------------------------------- the view


@dataclass
class _View:
    """Where each (sub)pixel's ray starts, and where 3D points end up in the image."""

    camera: Camera
    scale: float  # pixels per world unit, in the final image
    low: np.ndarray  # lowest screen corner, world units
    high: np.ndarray
    margin: float  # pixels
    width: int
    height: int
    samples: int
    far: float  # rays start at this distance along toward_viewer
    size: float  # size of the scene, for small offsets
    zoom: float  # pixel sizes are given for a REFERENCE_WIDTH image; multiply by this

    @classmethod
    def fitting(cls, points: np.ndarray, camera: Camera, width: int, margin: float, samples: int) -> _View:
        xy, _ = camera.project(points)
        low, high = xy.min(axis=0), xy.max(axis=0)
        extent = np.maximum(high - low, 1e-9)
        scale = (width - 2 * margin) / max(extent[0], 0.1 * extent[1])
        height = round(extent[1] * scale + 2 * margin)
        far = float((points @ camera.toward_viewer).max()) + 1.0
        size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0
        return cls(
            camera,
            scale,
            low,
            high,
            margin,
            int(width),
            max(height, 1),
            samples,
            far,
            size,
            width / REFERENCE_WIDTH,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """The size of the supersampled image: (rows, columns)."""
        return self.height * self.samples, self.width * self.samples

    def ray_origins(self) -> np.ndarray:
        """The start of every supersampled ray, row by row, on a plane in front of the scene."""
        rows, columns = self.shape
        x = self.low[0] + ((np.arange(columns) + 0.5) / self.samples - self.margin) / self.scale
        y = self.high[1] - ((np.arange(rows) + 0.5) / self.samples - self.margin) / self.scale
        camera = self.camera
        grid_x, grid_y = np.meshgrid(x, y)
        return (
            grid_x.reshape(-1, 1) * camera.right
            + grid_y.reshape(-1, 1) * camera.up
            + self.far * camera.toward_viewer
        )

    def subpixels(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Supersampled pixel coordinates (column, row) and ray depths of 3D points."""
        xy, _ = self.camera.project(points)
        column = ((xy[:, 0] - self.low[0]) * self.scale + self.margin) * self.samples
        row = ((self.high[1] - xy[:, 1]) * self.scale + self.margin) * self.samples
        depth = self.far - np.asarray(points, float) @ self.camera.toward_viewer
        return np.column_stack([column, row]), depth

    @property
    def world_per_subpixel(self) -> float:
        return 1 / (self.scale * self.samples)


# ---------------------------------------------------------------- rays and surfaces


@dataclass
class _Hits:
    distance: np.ndarray  # inf where nothing was hit
    surface: np.ndarray  # index into _Surfaces.surfaces
    part: np.ndarray  # cylinder: 0 side, 1 bottom, 2 top; triangles: triangle index


def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``direction`` and to each other."""
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


class _Surfaces:
    """
    All surfaces, prepared for casting many parallel rays at once.

    Rays are parallel both from the orthographic camera and toward the distant light. Seen
    along the ray direction, a ray is a point and a surface an outline: only the rays inside
    the outline's bounding box need to be tested, and sorting the rays makes them easy to find.
    """

    def __init__(self, surfaces: list[_Surface], size: float, engine: str = "numpy"):
        self.surfaces = surfaces
        self.engine = engine
        self._accelerator = None
        triangles, owners = [], []
        for index, surface in enumerate(surfaces):
            if surface.kind == "face":
                corners = surface.points
                for k in range(1, len(corners) - 1):  # a fan of triangles
                    triangles.append([corners[0], corners[k], corners[k + 1]])
                    owners.append(index)
        self.triangles = np.array(triangles, dtype=float).reshape(-1, 3, 3)
        self.triangle_surface = np.array(owners, dtype=int)
        edges1 = self.triangles[:, 1] - self.triangles[:, 0]
        edges2 = self.triangles[:, 2] - self.triangles[:, 0]
        normals = np.cross(edges1, edges2)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        self.triangle_normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        self.exact = [i for i, s in enumerate(surfaces) if s.kind != "face"]
        self.from_world = {i: np.linalg.inv(surfaces[i].to_world) for i in self.exact}
        self.epsilon = 1e-7 * size

        # colors and opacities, per surface and part
        self.colors = np.zeros((len(surfaces), 3, 3))
        self.opacities = np.ones((len(surfaces), 3))
        self.owners = np.array([s.owner for s in surfaces] or [0], dtype=int)
        self.pieces = np.array([s.piece for s in surfaces] or [0], dtype=int)
        for index, surface in enumerate(surfaces):
            styles = surface.part_styles or (surface.style,)
            for part in range(3):
                style = styles[min(part, len(styles) - 1)]
                self.colors[index, part] = _rgb(style.fill)
                self.opacities[index, part] = style.opacity

    def cast(self, origins: np.ndarray, direction: np.ndarray, start: np.ndarray, stop=None) -> _Hits:
        """
        The nearest hit of each ray ``origin + t * direction`` with ``start < t < stop``.

        ``direction`` is one direction for all rays; the numba engine also takes one per ray
        (``(n, 3)``). ``stop`` (one value or one per ray) ends the rays early.
        """
        count = len(origins)
        hits = _Hits(np.full(count, np.inf), np.full(count, -1), np.zeros(count, dtype=int))
        if count == 0 or not self.surfaces:
            return hits
        if self.engine == "numba":
            stop = None if stop is None else np.broadcast_to(np.asarray(stop, dtype=float), (count,))
            return self._cast_compiled(origins, direction, start, stop)
        if np.ndim(direction) != 1:
            raise ValueError("The numpy engine casts parallel rays only: one direction for all.")
        if stop is not None:
            hits = self.cast(origins, direction, start)
            beyond = hits.distance >= stop
            hits.distance[beyond], hits.surface[beyond], hits.part[beyond] = np.inf, -1, 0
            return hits
        across, up = _basis(direction)
        ray_u, ray_v, ray_w = origins @ across, origins @ up, origins @ direction
        order = np.argsort(ray_u, kind="stable")
        sorted_u = ray_u[order]

        def candidates(u, v):
            """The rays whose (u, v) lies in the bounding box of the given outline points."""
            pad = 1e-9 * (1 + np.abs(u).max())
            low, high = np.searchsorted(sorted_u, [u.min() - pad, u.max() + pad])
            rays = order[low:high]
            return rays[(ray_v[rays] >= v.min() - pad) & (ray_v[rays] <= v.max() + pad)]

        corner_u, corner_v, corner_w = (
            self.triangles @ across,
            self.triangles @ up,
            self.triangles @ direction,
        )
        for k in range(len(self.triangles)):
            u, v, w = corner_u[k], corner_v[k], corner_w[k]
            e1u, e1v, e2u, e2v = u[1] - u[0], v[1] - v[0], u[2] - u[0], v[2] - v[0]
            area = e1u * e2v - e2u * e1v
            if abs(area) < 1e-14 * (e1u**2 + e1v**2 + e2u**2 + e2v**2) + 1e-300:
                continue  # seen edge-on
            rays = candidates(u, v)
            if rays.size == 0:
                continue
            pu, pv = ray_u[rays] - u[0], ray_v[rays] - v[0]
            beta = (pu * e2v - e2u * pv) / area
            gamma = (e1u * pv - pu * e1v) / area
            tolerance = 1e-9
            inside = (beta >= -tolerance) & (gamma >= -tolerance) & (beta + gamma <= 1 + tolerance)
            distance = w[0] + beta * (w[1] - w[0]) + gamma * (w[2] - w[0]) - ray_w[rays]
            better = inside & (distance > start[rays]) & (distance < hits.distance[rays])
            rays = rays[better]
            hits.distance[rays] = distance[better]
            hits.surface[rays] = self.triangle_surface[k]
            hits.part[rays] = k

        for index in self.exact:
            surface = self.surfaces[index]
            corners = surface.exact.box_corners()
            rays = candidates(corners @ across, corners @ up)
            if rays.size == 0:
                continue
            inverse = self.from_world[index]
            unit_origins = origins[rays] @ inverse[:3, :3].T + inverse[:3, 3]
            unit_direction = inverse[:3, :3] @ direction
            intersect = _hit_sphere if surface.kind == "sphere" else _hit_cylinder
            distance, part = intersect(unit_origins, unit_direction, start[rays])
            better = distance < hits.distance[rays]
            rays = rays[better]
            hits.distance[rays] = distance[better]
            hits.surface[rays] = index
            hits.part[rays] = part[better]
        return hits

    def _cast_compiled(self, origins, direction, start, stop=None) -> _Hits:
        """The same as :meth:`cast`, walking a bounding volume hierarchy in compiled code (numba)."""
        if self._accelerator is None:
            from . import _fastcast

            exact = [self.surfaces[i].exact for i in self.exact]
            self._accelerator = _fastcast.Accelerator(
                self.triangles,
                [shape.kind for shape in exact],
                [self.from_world[i] for i in self.exact],
                [shape.box_corners() for shape in exact],
            )
        distance, primitive, part = self._accelerator.cast(origins, direction, start, stop)
        triangles = len(self.triangles)
        found = primitive >= 0
        is_triangle = found & (primitive < triangles)
        is_exact = found & ~is_triangle
        surface = np.full(len(origins), -1)
        surface[is_triangle] = self.triangle_surface[primitive[is_triangle]]
        surface[is_exact] = np.asarray(self.exact, dtype=int)[primitive[is_exact] - triangles]
        part = np.where(is_triangle, primitive, part)  # triangles: their index, as in cast
        return _Hits(distance, surface, part)

    def normals(self, hits: _Hits, points: np.ndarray) -> np.ndarray:
        """Unit normals at the hit points (pointing out of solids)."""
        normals = np.zeros_like(points)
        is_face = np.array([self.surfaces[s].kind == "face" for s in range(len(self.surfaces))])
        flat = is_face[hits.surface]
        normals[flat] = self.triangle_normals[hits.part[flat]]
        for index in self.exact:
            mask = hits.surface == index
            if not mask.any():
                continue
            inverse = self.from_world[index]
            unit = points[mask] @ inverse[:3, :3].T + inverse[:3, 3]
            if self.surfaces[index].kind == "sphere":
                unit_normal = unit
            else:
                part = hits.part[mask]
                unit_normal = np.column_stack([unit[:, 0], unit[:, 1], np.zeros(len(unit))])
                unit_normal[part == 1] = [0, 0, -1]
                unit_normal[part == 2] = [0, 0, 1]
            normal = unit_normal @ inverse[:3, :3]  # normals transform with the inverse transpose
            normals[mask] = normal / np.linalg.norm(normal, axis=1, keepdims=True)
        return normals

    def part_index(self, hits: _Hits) -> np.ndarray:
        """Which style of the hit surface applies: the cylinder part, else 0."""
        is_exact = np.zeros(len(self.surfaces), dtype=bool)
        is_exact[self.exact] = True
        return np.where(is_exact[hits.surface], hits.part, 0)


def _hit_sphere(origins, direction, start):
    """Nearest hits with the unit sphere, beyond ``start``."""
    a = direction @ direction
    b = 2 * origins @ direction
    c = np.einsum("ij,ij->i", origins, origins) - 1
    discriminant = b * b - 4 * a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    near, far = (-b - root) / (2 * a), (-b + root) / (2 * a)
    distance = np.where(near > start, near, np.where(far > start, far, np.inf))
    distance[discriminant < 0] = np.inf
    return distance, np.zeros(len(origins), dtype=int)


def _hit_cylinder(origins, direction, start):
    """Nearest hits with the unit cylinder (radius 1, z from -0.5 to 0.5), beyond ``start``."""
    count = len(origins)
    candidates = np.full((count, 4), np.inf)
    ox, oy, oz = origins.T
    dx, dy, dz = direction
    a = dx * dx + dy * dy
    if a > 1e-15:  # the side
        b = 2 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - 1
        discriminant = b * b - 4 * a * c
        root = np.sqrt(np.maximum(discriminant, 0))
        for column, sign in ((0, -1), (1, 1)):
            t = (-b + sign * root) / (2 * a)
            valid = (discriminant >= 0) & (np.abs(oz + t * dz) <= 0.5)
            candidates[valid, column] = t[valid]
    if abs(dz) > 1e-15:  # the bottom and top
        for column, z in ((2, -0.5), (3, 0.5)):
            t = (z - oz) / dz
            valid = (ox + t * dx) ** 2 + (oy + t * dy) ** 2 <= 1
            candidates[valid, column] = t[valid]
    candidates[candidates <= start[:, None]] = np.inf
    best = np.argmin(candidates, axis=1)
    distance = candidates[np.arange(count), best]
    part = np.array([0, 0, 1, 2])[best]
    return distance, part


# ---------------------------------------------------------------- light and layers


class _Lighting:
    """
    The brightness of surfaces: light from a distant source (Lambert's law, with shadows) plus
    light from the surroundings (ambient, reduced by occlusion), shaded as the look says.
    """

    def __init__(self, surfaces: _Surfaces, light: np.ndarray, look: Look, size: float):
        self.surfaces = surfaces
        self.light = np.asarray(light, dtype=float)
        self.look = look
        self.ambient = look.ambient
        self.shadows = look.shadows
        self.radius = look.occlusion_radius * size

    def brightness(
        self, points: np.ndarray, normals: np.ndarray, openness: np.ndarray | None = None
    ) -> np.ndarray:
        """
        From 0 (black) to 1 (full color). ``openness`` is 1 where nothing blocks the
        surroundings and lower in corners (see :meth:`openness`); None means 1 everywhere.
        """
        look = self.look
        ambient = self.ambient if openness is None else self.ambient * (1 - look.occlusion * (1 - openness))
        if look.shading == "flat":
            return ambient / self.ambient if self.ambient > 0 else np.ones(len(points))
        direct = np.clip(normals @ self.light, 0, 1)
        if self.shadows:
            lit = direct > 0
            direct[lit] *= self.light_reaching(points[lit] + normals[lit] * 1e3 * self.surfaces.epsilon)
        if look.shading == "cel":
            direct = np.minimum(np.floor(direct * look.bands), look.bands - 1) / (look.bands - 1)
        return ambient + (1 - self.ambient) * direct

    def openness(self, points: np.ndarray, normals: np.ndarray, turn: int = 0) -> np.ndarray:
        """
        How open each point is to its surroundings (ambient occlusion): 1 if no surface lies
        within ``occlusion_radius`` above it, down to 0 if it is enclosed.

        With the numba engine, every point casts its own random rays (see
        :meth:`random_openness`). The numpy engine can only cast parallel rays, so it uses a
        fixed set of directions spread over the sphere, the same for all points; each counts
        with the cosine of its angle to the normal. ``turn`` chooses one of several turned
        copies of the set (or a different random seed), so that neighbouring pixels sample
        different directions.
        """
        if self.surfaces.engine == "numba":
            return self.random_openness(points, normals, seed=turn)
        origins = points + normals * 1e3 * self.surfaces.epsilon
        blocked = np.zeros(len(points))
        total = np.zeros(len(points))
        samples = self.look.occlusion_samples or FIXED_OCCLUSION_SAMPLES
        for direction in _sphere_directions(2 * samples, turn):
            weight = normals @ direction
            above = np.nonzero(weight > 0)[0]
            if above.size == 0:
                continue
            hits = self.surfaces.cast(origins[above], direction, np.zeros(above.size))
            near = hits.distance < self.radius
            opacity = np.zeros(above.size)
            opacity[near] = self.surfaces.opacities[
                hits.surface[near], self.surfaces.part_index(_subset(hits, near))
            ]
            blocked[above] += weight[above] * opacity
            total[above] += weight[above]
        return 1 - np.divide(blocked, total, out=np.zeros_like(blocked), where=total > 0)

    def random_openness(self, points: np.ndarray, normals: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        Ambient occlusion with random rays, which the numba engine can cast in any direction.

        Each point casts ``occlusion_samples`` rays into the half space above its surface,
        more of them near the normal (cosine weighted, so they count equally). The directions
        are stratified (one per cell of a grid) and shifted randomly per point, so that
        neighbouring points sample different directions: the result is fine noise, which the
        smoothing over surfaces removes, instead of the rings a fixed set of directions leaves.
        The random numbers come from a fixed ``seed``, so the same scene gives the same picture.
        """
        count = len(points)
        side = max(int(round(np.sqrt(self.look.occlusion_samples or RANDOM_OCCLUSION_SAMPLES))), 1)
        rng = np.random.default_rng(12345 + seed)
        shift = rng.random((count, 2))  # one random shift of the grid per point
        tangent, bitangent = _tangents(normals)
        origins = points + normals * 1e3 * self.surfaces.epsilon
        stop = np.full(count, self.radius)
        blocked = np.zeros(count)
        for k in range(side * side):
            cell = np.array(divmod(k, side), dtype=float)
            u = np.mod((cell + rng.random((count, 2))) / side + shift, 1.0)
            radius, angle = np.sqrt(u[:, 0]), 2 * np.pi * u[:, 1]
            directions = (
                (radius * np.cos(angle))[:, None] * tangent
                + (radius * np.sin(angle))[:, None] * bitangent
                + np.sqrt(1 - u[:, 0])[:, None] * normals
            )
            hits = self.surfaces.cast(origins, directions, np.zeros(count), stop)
            near = np.isfinite(hits.distance)
            blocked[near] += self.surfaces.opacities[
                hits.surface[near], self.surfaces.part_index(_subset(hits, near))
            ]
        return 1 - blocked / (side * side)

    def light_reaching(self, points: np.ndarray) -> np.ndarray:
        """How much light reaches each point: 1 unblocked, less behind transparent faces, 0 in shadow."""
        through = np.ones(len(points))
        active = np.arange(len(points))
        start = np.zeros(len(points))
        for _ in range(MAX_LAYERS):
            hits = self.surfaces.cast(points[active], self.light, start[active])
            found = np.isfinite(hits.distance)
            active, hits = active[found], _subset(hits, found)
            if active.size == 0:
                break
            opacity = self.surfaces.opacities[hits.surface, self.surfaces.part_index(hits)]
            through[active] *= 1 - opacity
            start[active] = hits.distance + 1e3 * self.surfaces.epsilon
            active = active[through[active] > 0.01]
        return through


def _subset(hits: _Hits, mask: np.ndarray) -> _Hits:
    return _Hits(hits.distance[mask], hits.surface[mask], hits.part[mask])


def _tangents(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors per normal, perpendicular to it and to each other."""
    helper = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    tangent = np.cross(helper, normals)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent, np.cross(normals, tangent)


def _sphere_directions(count: int, turn: int = 0) -> np.ndarray:
    """
    ``count`` unit vectors spread evenly over the sphere (a Fibonacci lattice); different
    ``turn`` numbers give the same lattice turned in a fixed, irregular way.
    """
    k = np.arange(count) + 0.5
    z = 1 - 2 * k / count
    angle = np.pi * (1 + 5**0.5) * k
    radius = np.sqrt(1 - z * z)
    directions = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])
    if turn:
        axis = np.array([np.sin(1.3 * turn), np.cos(2.1 * turn), 0.7])
        directions = directions @ Transform().rotate(axis, 47.0 * turn).matrix[:3, :3].T
    return directions


#: Rays per point for ambient occlusion when ``occlusion_samples`` is None: a fixed set of
#: directions with the numpy engine, random rays with numba (which needs more of them for the
#: same smoothness, but casts them faster).
FIXED_OCCLUSION_SAMPLES = 24
RANDOM_OCCLUSION_SAMPLES = 64

#: How many turned direction sets neighbouring pixels use for ambient occlusion (2 x 2 pattern).
OCCLUSION_TURNS = 4

#: How far ambient occlusion is smoothed, in pixels of a 500-pixel-wide image.
OCCLUSION_BLUR = 3.0


@dataclass
class _Layers:
    """The traced surfaces of the supersampled image."""

    color: np.ndarray  # (n, 3) premultiplied with coverage
    coverage: np.ndarray  # (n,) 0 where only background shows, 1 where opaque
    depth: np.ndarray  # (n,) depth of the first opaque surface (inf if none)
    owner: np.ndarray  # (n,) scene item of that surface (-1 if none)
    first_piece: np.ndarray  # (n,) geometry of the first surface hit, opaque or not (-1 if none)
    first_normal: np.ndarray  # (n, 3) its normal, facing the viewer


def _trace_layers(surfaces: _Surfaces, lighting: _Lighting, view: _View) -> _Layers:
    origins = view.ray_origins()
    direction = -view.camera.toward_viewer
    count = len(origins)
    color = np.zeros((count, 3))
    through = np.ones(count)
    depth = np.full(count, np.inf)
    owner = np.full(count, -1)
    first_piece = np.full(count, -1)
    first_normal = np.zeros((count, 3))
    start = np.zeros(count)
    active = np.arange(count)
    for layer in range(MAX_LAYERS):
        hits = surfaces.cast(origins[active], direction, start[active])
        found = np.isfinite(hits.distance)
        active, hits = active[found], _subset(hits, found)
        if active.size == 0:
            break
        points = origins[active] + hits.distance[:, None] * direction
        normals = surfaces.normals(hits, points)
        normals[normals @ direction > 0] *= -1  # thin faces are seen from both sides
        if layer == 0:
            first_piece[active] = surfaces.pieces[hits.surface]
            first_normal[active] = normals
        part = surfaces.part_index(hits)
        base = surfaces.colors[hits.surface, part]
        opacity = surfaces.opacities[hits.surface, part]
        openness = None
        if lighting.look.occlusion:
            openness = _openness(lighting, view, active, hits, points, normals, smooth=layer == 0)
        shade = lighting.brightness(points, normals, openness)
        color[active] += (through[active] * opacity * shade)[:, None] * base

        first_opaque = (opacity >= 0.999) & np.isinf(depth[active])
        depth[active[first_opaque]] = hits.distance[first_opaque]
        owner[active[first_opaque]] = surfaces.owners[hits.surface[first_opaque]]

        through[active] *= 1 - opacity
        start[active] = hits.distance + 10 * surfaces.epsilon
        active = active[through[active] > 0.004]
    return _Layers(color, 1 - through, depth, owner, first_piece, first_normal)


def _openness(lighting: _Lighting, view: _View, rays: np.ndarray, hits: _Hits, points, normals, smooth: bool):
    """
    Ambient occlusion for the hit points of a layer of rays.

    It changes slowly over a surface, so it is computed once per pixel (for its middle
    subpixel), with direction sets that differ between neighbouring pixels, and then smoothed
    over each surface (``smooth``, for the first layer). Subpixels that hit the same surface
    as their pixel's middle share its value; the others get their own.
    """
    samples, columns = view.samples, view.shape[1]
    row, column = np.divmod(rays, columns)
    pixel_row, pixel_column = row // samples, column // samples
    middle = (pixel_row * samples + samples // 2) * columns + pixel_column * samples + samples // 2
    where = np.minimum(np.searchsorted(rays, middle), len(rays) - 1)  # rays are sorted
    is_middle = middle == rays
    shared = (rays[where] == middle) & (hits.surface[where] == hits.surface) & ~is_middle

    openness = np.ones(len(rays))
    turn = (pixel_row % 2) * 2 + pixel_column % 2
    for k in range(OCCLUSION_TURNS):
        mine = is_middle & (turn == k)
        if mine.any():
            openness[mine] = lighting.openness(points[mine], normals[mine], k)
    if smooth and is_middle.any():
        openness[is_middle] = _smooth_over_surfaces(
            openness[is_middle],
            pixel_row[is_middle],
            pixel_column[is_middle],
            lighting.surfaces.pieces[hits.surface[is_middle]],
            normals[is_middle],
            (view.height, view.width),
            OCCLUSION_BLUR * view.zoom,
        )
    alone = ~is_middle & ~shared
    if alone.any():
        openness[alone] = lighting.openness(points[alone], normals[alone])
    openness[shared] = openness[where[shared]]
    return openness


def _smooth_over_surfaces(values, rows, columns, pieces, normals, shape, radius: float) -> np.ndarray:
    """
    Average each pixel's value with its neighbours within ``radius`` pixels that show the same
    geometry facing the same way, so the smoothing stops at outlines and sharp edges.
    """
    reach = max(int(round(radius)), 1)
    grid = np.full(shape, np.nan)
    piece = np.full(shape, -1)
    normal = np.zeros((*shape, 3))
    grid[rows, columns], piece[rows, columns], normal[rows, columns] = values, pieces, normals
    total = np.zeros(len(values))
    weight = np.zeros(len(values))
    height, width = shape
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > reach * reach:
                continue
            r, c = rows + dy, columns + dx
            inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            r, c = np.clip(r, 0, height - 1), np.clip(c, 0, width - 1)
            alike = inside & (piece[r, c] == pieces) & (np.einsum("ij,ij->i", normal[r, c], normals) > 0.95)
            total[alike] += grid[r, c][alike]
            weight[alike] += 1
    return total / np.maximum(weight, 1)


# ---------------------------------------------------------------- lines, points and labels


class _Canvas:
    """The supersampled image, onto which lines and points are drawn with a depth test."""

    def __init__(self, layers: _Layers, view: _View):
        rows, columns = view.shape
        self.view = view
        self.color = layers.color.reshape(rows, columns, 3)
        self.coverage = layers.coverage.reshape(rows, columns)
        self.depth = layers.depth.reshape(rows, columns)
        self.owner = layers.owner.reshape(rows, columns)

    def outline(self, layers: _Layers, look: Look) -> None:
        """
        Draw outlines where one geometry meets another or the background, and where the
        surface bends by more than the crease angle.
        """
        rows, columns = self.view.shape
        piece = layers.first_piece.reshape(rows, columns)
        normal = layers.first_normal.reshape(rows, columns, 3)
        limit = np.cos(np.radians(look.crease_angle))
        edges = np.zeros((rows, columns), dtype=bool)
        for axis in (0, 1):
            here = [slice(None), slice(None)]
            there = [slice(None), slice(None)]
            here[axis], there[axis] = slice(None, -1), slice(1, None)
            a, b = piece[tuple(here)], piece[tuple(there)]
            bend = np.einsum("ijk,ijk->ij", normal[tuple(here)], normal[tuple(there)])
            edge = (a != b) | ((a >= 0) & (bend < limit))
            edges[tuple(here)] |= edge
            edges[tuple(there)] |= edge
        radius = max(look.outline_width * self.view.zoom * self.view.samples / 2 - 1, 0)
        edges = _dilate(edges, radius + 0.5)  # the edges already cover the pixels on both sides
        rgb = _rgb(look.outline_color)
        self.color[edges] = rgb
        self.coverage[edges] = 1

    def bias(self, width: float) -> float:
        """How far in front of a surface a line may be and still count as on it."""
        return self.view.world_per_subpixel * (3 * width + 6) + 1e-6 * self.view.size

    def stroke(self, line: _Line) -> None:
        style = line.style
        if style.stroke == "none" or style.opacity <= 0 or len(line.points) < 2:
            return
        view = self.view
        pixels, depths = view.subpixels(line.points)
        width = max(style.line_width * view.zoom * view.samples, 0.5)
        tip = None
        if style.arrow:
            tip, pixels = _arrow_tip(pixels, width)
        dashes = DASH_PATTERNS[style.dash]
        pattern = None if dashes is None else (dashes[0] * width, (dashes[0] + dashes[1]) * width)
        bias = self.bias(width)

        windows = []
        travelled = 0.0
        for k in range(len(pixels) - 1):
            a, b = pixels[k], pixels[k + 1]
            window = self._window(np.vstack([a, b]), width / 2 + 1)
            if window is None:
                travelled += float(np.linalg.norm(b - a))
                continue
            rows, columns, x, y = window
            segment = b - a
            length_squared = float(segment @ segment)
            along = (
                np.zeros_like(x)
                if length_squared == 0
                else np.clip(((x - a[0]) * segment[0] + (y - a[1]) * segment[1]) / length_squared, 0, 1)
            )
            closest_x, closest_y = a[0] + along * segment[0], a[1] + along * segment[1]
            mask = (x - closest_x) ** 2 + (y - closest_y) ** 2 <= (width / 2) ** 2
            if pattern is not None:
                position = travelled + along * np.sqrt(length_squared)
                mask &= np.mod(position, pattern[1]) < pattern[0]
            if not line.on_top:
                depth = depths[k] + along * (depths[k + 1] - depths[k])
                mask &= depth <= self.depth[rows, columns] + bias
            windows.append((rows, columns, mask))
            travelled += float(np.sqrt(length_squared))
        if tip is not None:
            windows.append(self._triangle(tip, depths[-1], bias, line.on_top))
        self._paint(windows, _rgb(style.stroke), style.opacity)

    def dot(self, marker: Marker) -> None:
        style = marker.style
        if style.fill == "none":
            return
        view = self.view
        pixels, depths = view.subpixels(marker.position[np.newaxis])
        radius = np.sqrt(style.marker_size) / 2 * view.zoom * view.samples
        window = self._window(pixels, radius + 1)
        if window is None:
            return
        rows, columns, x, y = window
        mask = (x - pixels[0, 0]) ** 2 + (y - pixels[0, 1]) ** 2 <= radius**2
        mask &= depths[0] <= self.depth[rows, columns] + self.bias(radius)
        self._paint([(rows, columns, mask)], _rgb(style.fill), style.opacity)

    def _window(self, pixels: np.ndarray, pad: float):
        """The subpixels near some points: row and column slices and the subpixel centers."""
        height, width = self.depth.shape
        low = np.floor(pixels.min(axis=0) - pad).astype(int)
        high = np.ceil(pixels.max(axis=0) + pad).astype(int)
        c0, r0 = max(low[0], 0), max(low[1], 0)
        c1, r1 = min(high[0], width), min(high[1], height)
        if c0 >= c1 or r0 >= r1:
            return None
        y, x = np.mgrid[r0:r1, c0:c1] + 0.5
        return slice(r0, r1), slice(c0, c1), x, y

    def _triangle(self, corners: np.ndarray, depth: float, bias: float, on_top: bool):
        window = self._window(corners, 1)
        if window is None:
            return None
        rows, columns, x, y = window
        (ax, ay), (bx, by), (cx, cy) = corners
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if area == 0:
            return None
        beta = ((x - ax) * (cy - ay) - (cx - ax) * (y - ay)) / area
        gamma = ((bx - ax) * (y - ay) - (x - ax) * (by - ay)) / area
        mask = (beta >= 0) & (gamma >= 0) & (beta + gamma <= 1)
        if not on_top:
            mask &= depth <= self.depth[rows, columns] + bias
        return rows, columns, mask

    def _paint(self, windows, rgb: np.ndarray, opacity: float) -> None:
        """Blend a color over the union of the windows' masks, once per subpixel."""
        windows = [w for w in windows if w is not None]
        if not windows:
            return
        height, width = self.depth.shape
        rows = slice(min(w[0].start for w in windows), max(w[0].stop for w in windows))
        columns = slice(min(w[1].start for w in windows), max(w[1].stop for w in windows))
        union = np.zeros((rows.stop - rows.start, columns.stop - columns.start), dtype=bool)
        for r, c, mask in windows:
            union[
                r.start - rows.start : r.stop - rows.start, c.start - columns.start : c.stop - columns.start
            ] |= mask
        alpha = opacity * union
        self.color[rows, columns] = (
            self.color[rows, columns] * (1 - alpha[..., None]) + alpha[..., None] * rgb
        )
        self.coverage[rows, columns] = self.coverage[rows, columns] * (1 - alpha) + alpha

    def downsampled(self, background: str | None) -> Image.Image:
        """Average the subpixels into pixels and return the image."""
        view = self.view
        s = view.samples
        color = self.color.reshape(view.height, s, view.width, s, 3).mean(axis=(1, 3))
        coverage = self.coverage.reshape(view.height, s, view.width, s).mean(axis=(1, 3))
        if background is not None:
            color = color + (1 - coverage)[..., None] * _rgb(background)
            coverage = np.ones_like(coverage)
        else:
            color = np.divide(
                color, coverage[..., None], out=np.zeros_like(color), where=coverage[..., None] > 0
            )
        rgba = np.dstack([np.clip(color, 0, 1), coverage])
        return Image.fromarray(np.round(rgba * 255).astype(np.uint8))

    def hidden(self, point: np.ndarray, owners: frozenset[int]) -> bool:
        """Whether a surface of another scene item lies in front of the point."""
        pixels, depths = self.view.subpixels(point[np.newaxis])
        column, row = np.floor(pixels[0]).astype(int)
        height, width = self.depth.shape
        if not (0 <= row < height and 0 <= column < width):
            return False
        in_front = self.depth[row, column] + self.bias(2.0) < depths[0]
        return bool(in_front and self.owner[row, column] not in owners)


def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Widen the marked pixels by ``radius`` pixels in every direction."""
    reach = int(np.floor(radius))
    if reach < 1:
        return mask
    grown = mask.copy()
    rows, columns = mask.shape
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if (dy or dx) and dx * dx + dy * dy <= radius * radius:
                grown[max(dy, 0) : rows + min(dy, 0), max(dx, 0) : columns + min(dx, 0)] |= mask[
                    max(-dy, 0) : rows + min(-dy, 0), max(-dx, 0) : columns + min(-dx, 0)
                ]
    return grown


def _arrow_tip(pixels: np.ndarray, width: float) -> tuple[np.ndarray | None, np.ndarray]:
    """The corners of an arrow tip at the end of the line, and the line shortened to meet it."""
    end, before = pixels[-1], pixels[-2]
    direction = end - before
    length = float(np.linalg.norm(direction))
    if length == 0:
        return None, pixels
    direction /= length
    size = ARROW_LENGTH * width
    base = end - direction * size
    across = np.array([-direction[1], direction[0]]) * size / 2
    shortened = pixels.copy()
    shortened[-1] = end - direction * min(size * 0.7, length)
    return np.array([end, base + across, base - across]), shortened


def _font(size: float):
    """A font with many symbols (DejaVu Sans from matplotlib if available), else Pillow's own."""
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # matplotlib missing, or font not found
        return ImageFont.load_default(size)


def _draw_labels(image: Image.Image, labels: list[_Label], canvas: _Canvas, view: _View) -> None:
    if not labels:
        return
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fonts: dict[float, ImageFont.ImageFont] = {}
    for entry in labels:
        label = entry.label
        if not label.text or canvas.hidden(np.asarray(label.position, float), entry.owners):
            continue
        style = label.style
        size = style.font_size * view.zoom
        font = fonts.setdefault(size, _font(size))
        pixels, _ = view.subpixels(np.asarray(label.position, float)[np.newaxis])
        x, y = pixels[0] / view.samples
        color = ImageColor.getrgb(color_value(style.text_color))
        if label.align == "center":
            draw.text((x, y + 0.35 * size), label.text, fill=color, font=font, anchor="ms")
        else:
            draw.text((x + 3 * view.zoom, y - 3 * view.zoom), label.text, fill=color, font=font, anchor="ls")
    image.alpha_composite(layer)
