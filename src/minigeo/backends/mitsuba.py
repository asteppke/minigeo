"""
Draw a scene with `Mitsuba 3 <https://www.mitsuba-renderer.org/>`_, a physically based
renderer: soft light from the surroundings, light bouncing between surfaces, and
see-through faces.

The scene is handed to Mitsuba like this:

* Spheres and cylinders become Mitsuba's own exact shapes, so they are perfectly round. A
  cylinder gets two discs as caps (Mitsuba's cylinder is open). Stretched spheres and
  cylinders, which Mitsuba's shapes cannot be, become fine triangle meshes instead.
* All other faces become triangle meshes, one per scene item and style.
* A fill becomes a matte (diffuse) surface, seen the same from both sides; an opacity below
  1 makes it see-through, as in the other backends.
* Colors are converted from screen colors (sRGB) to the linear light values Mitsuba works
  with, and back for the picture.
* The camera is orthographic and fitted exactly as in the ray tracer, so the same ``elev``,
  ``azim`` and ``width`` give the same picture.

Mitsuba renders only surfaces. Lines, points and labels are drawn afterwards by the ray
tracer's code, hidden where a surface lies in front of them, using the depth image that
Mitsuba renders as well.

Mitsuba chooses its *variant* (CPU or GPU, RGB or spectral) once for the whole Python
process. This backend uses the variant already chosen, if any, and otherwise the CPU variant
``"scalar_rgb"``, which works everywhere; pass ``variant="cuda_ad_rgb"`` for an NVIDIA GPU or
``"llvm_ad_rgb"`` for a faster, vectorized CPU variant (needs LLVM).

Requires ``pip install "minigeo[mitsuba]"``. It is not part of ``minigeo[all]``.
"""

from __future__ import annotations

import numpy as np

from ..camera import Camera
from ..colors import to_rgb
from ..geometry import Cylinder, Sphere
from ..geometry.mesh import polygon_triangles, triangulate
from ..style import Style
from .raytrace import REFERENCE_WIDTH, RaytracedImage, _Canvas, _Contents, _finish, _Layers, _View

#: The variant used when none was chosen before.
DEFAULT_VARIANT = "scalar_rgb"

#: Light setups: ``"sun"`` is a distant light from the upper left behind the viewer (as in
#: the ray tracer) plus soft light from the surroundings; ``"sky"`` only the surroundings,
#: which gives soft shading with darkened corners and no cast shadows.
LIGHTS = ("sun", "sky")

#: Share of light from the surroundings in the ``"sun"`` setup (as the ray tracer's ``ambient``).
AMBIENT = 0.4

#: Segments of the triangle meshes for stretched spheres and cylinders.
FINE_SEGMENTS = 64


def render(
    scene,
    *,
    width: int = 500,
    elev: float = 30.0,
    azim: float = -60.0,
    vertical_axis: str = "z",
    margin: float = 20,
    samples: int = 2,
    spp: int = 64,
    light: str = "sun",
    background: str | None = None,
    variant: str | None = None,
    max_depth: int = 8,
    seed: int = 0,
) -> RaytracedImage:
    """
    Draw ``scene`` with Mitsuba 3.

    Args:
        scene: The :class:`~minigeo.scene.Scene` to draw.
        width: Width of the image in pixels; the height follows from the scene's shape.
        elev, azim, vertical_axis: Where the viewer looks from, as in the other backends.
        margin: Empty space around the drawing, in pixels of a 500-pixel-wide image.
        samples: Subpixels along each direction, for smooth lines (as in the ray tracer).
        spp: Light paths per pixel: more gives less noise and takes longer.
        light: ``"sun"`` or ``"sky"`` (see :data:`LIGHTS`).
        background: A background color, or None for a transparent background.
        variant: The Mitsuba variant, e.g. ``"cuda_ad_rgb"``; default: the one already chosen,
            else ``"scalar_rgb"``.
        max_depth: How often light may bounce between surfaces (more: brighter corners).
        seed: Seed for the random light paths; the same seed gives the same picture.

    Returns:
        A :class:`~minigeo.backends.raytrace.RaytracedImage`, as from the ray tracer.

    Example:
        >>> scene.render("mitsuba", spp=256).save("table.png")
        >>> scene.save("table.png", backend="mitsuba", light="sky")
    """
    if samples < 1 or spp < 1:
        raise ValueError("samples and spp must be at least 1.")
    mi = _mitsuba(variant)
    camera = Camera(elev, azim, vertical_axis)
    contents = _Contents.collect(scene)
    view = _View.fitting(contents.points(camera), camera, width, margin * width / REFERENCE_WIDTH, samples)
    shapes = _shapes(contents)

    per_subpixel = max(1, round(spp / samples**2))
    picture = scene_dict(mi, view, shapes, light, per_subpixel, max_depth, seed)
    rgba = np.asarray(mi.render(mi.load_dict(picture), seed=seed), dtype=float)
    depth, owner = _depth_and_owner(mi, view, shapes, seed)

    coverage = np.clip(rgba[..., 3], 0, 1)
    color = _premultiplied_srgb(rgba[..., :3], coverage)
    rows, columns = view.shape
    count = rows * columns
    layers = _Layers(
        color.reshape(count, 3),
        coverage.reshape(count),
        depth.reshape(count),
        owner.reshape(count),
        np.full(count, -1),
        np.zeros((count, 3)),
    )
    return RaytracedImage(_finish(_Canvas(layers, view), contents, background))


def _mitsuba(variant: str | None):
    """Import Mitsuba and make sure a variant is chosen."""
    try:
        import mitsuba as mi
    except ImportError as error:  # pragma: no cover - depends on the installation
        raise ImportError('The mitsuba backend needs Mitsuba 3: pip install "minigeo[mitsuba]"') from error
    if variant is not None:
        mi.set_variant(variant)
    elif mi.variant() is None:
        mi.set_variant(DEFAULT_VARIANT)
    return mi


# ---------------------------------------------------------------- colors


def linear(rgb) -> list[float]:
    """Screen colors (sRGB, 0 to 1) as linear light values, as Mitsuba expects them."""
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).tolist()


def srgb(values: np.ndarray) -> np.ndarray:
    """Linear light values as screen colors (sRGB, 0 to 1): the inverse of :func:`linear`."""
    values = np.clip(values, 0, 1)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


def _premultiplied_srgb(rgb: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Mitsuba's colors are linear and weighted by coverage; convert the color, keep the weight."""
    covered = coverage[..., None]
    plain = np.divide(rgb, covered, out=np.zeros_like(rgb), where=covered > 1e-6)
    return srgb(plain) * covered


# ---------------------------------------------------------------- shapes


class _Shape:
    """
    One Mitsuba shape: an exact one (a dict without material) or a triangle mesh, with
    its style and the scene item it belongs to.
    """

    count = 0  # for names: Mitsuba tells shapes apart by name

    def __init__(self, style: Style, owner: int, exact: dict | None = None, points=None, triangles=None):
        _Shape.count += 1
        self.name = f"shape_{_Shape.count:06d}"
        self.style, self.owner = style, owner
        self.exact, self.points, self.triangles = exact, points, triangles

    @property
    def opaque(self) -> bool:
        return self.style.opacity >= 0.999

    def build(self, mi, bsdf: dict):
        """The shape for ``mi.load_dict``, with the given material."""
        if self.exact is not None:
            shape = {**self.exact, "bsdf": bsdf}
            if "to_world" in shape:
                shape["to_world"] = mi.ScalarTransform4f(shape["to_world"].tolist())
            return shape
        properties = mi.Properties()
        properties["bsdf"] = mi.load_dict(bsdf)
        mesh = mi.Mesh(self.name, len(self.points), len(self.triangles), props=properties)
        parameters = mi.traverse(mesh)
        positions, indices = parameters["vertex_positions"], parameters["faces"]
        parameters["vertex_positions"] = type(positions)(np.asarray(self.points, dtype=np.float32).ravel())
        parameters["faces"] = type(indices)(np.asarray(self.triangles, dtype=np.uint32).ravel())
        parameters.update()
        return mesh


def _matte(rgb) -> dict:
    """A matte surface, the same from both sides; ``rgb`` in linear light values."""
    return {"type": "twosided", "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": rgb}}}


def _bsdf(style: Style) -> dict:
    """A matte surface in the fill color, see-through if not opaque."""
    matte = _matte(linear(to_rgb(style.fill)))
    if style.opacity >= 0.999:
        return matte
    return {"type": "mask", "opacity": float(style.opacity), "material": matte}


def _shapes(contents: _Contents) -> list[_Shape]:
    shapes: list[_Shape] = []
    faces: dict[tuple, list[np.ndarray]] = {}  # (owner, fill, opacity) -> triangles as corner points
    styles: dict[tuple, Style] = {}
    for surface in contents.surfaces:
        if surface.kind == "face":
            key = (surface.owner, surface.style.fill, surface.style.opacity)
            styles[key] = surface.style
            points = surface.points
            faces.setdefault(key, []).append(points[polygon_triangles(points)])
        elif surface.kind == "sphere":
            shapes += _sphere(surface.to_world, surface.part_styles[0], surface.owner)
        else:
            shapes += _cylinder(surface.to_world, surface.part_styles, surface.owner)
    for key, triangles in faces.items():
        corners = np.concatenate(triangles).reshape(-1, 3)  # flat shading: no shared corners
        shapes.append(
            _Shape(styles[key], key[0], points=corners, triangles=np.arange(len(corners)).reshape(-1, 3))
        )
    return shapes


def _fine_mesh(unit_shape, to_world: np.ndarray, style: Style, owner: int) -> list[_Shape]:
    """A stretched sphere or cylinder as a fine triangle mesh (Mitsuba's shapes cannot be stretched)."""
    points = unit_shape.local_vertices @ to_world[:3, :3].T + to_world[:3, 3]
    _, triangles = triangulate(unit_shape.face_indices, unit_shape.local_vertices)
    return [_Shape(style, owner, points=points, triangles=triangles)]


def _sphere(to_world: np.ndarray, style: Style, owner: int) -> list[_Shape]:
    linear_part = to_world[:3, :3]
    radius = float(np.linalg.norm(linear_part[:, 0]))
    if not np.allclose(linear_part.T @ linear_part, radius**2 * np.eye(3), atol=1e-9 * max(radius, 1) ** 2):
        return _fine_mesh(Sphere([0, 0, 0], 1, segments=FINE_SEGMENTS), to_world, style, owner)
    return [_Shape(style, owner, {"type": "sphere", "center": to_world[:3, 3].tolist(), "radius": radius})]


def _cylinder(to_world: np.ndarray, styles: tuple[Style, ...], owner: int) -> list[_Shape]:
    """The open side as Mitsuba's cylinder, and the caps as two discs."""
    across1, across2, along = to_world[:3, 0], to_world[:3, 1], to_world[:3, 2]
    radius = float(np.linalg.norm(across1))
    scale = max(radius, float(np.linalg.norm(along)), 1e-12)
    dots = np.array([across1 @ across2, across1 @ along, across2 @ along]) / scale**2
    if abs(np.linalg.norm(across2) - radius) > 1e-9 * scale or np.abs(dots).max() > 1e-9:
        unit = Cylinder([0, 0, 0], 1, 1, segments=FINE_SEGMENTS)
        return _fine_mesh(unit, to_world, styles[0], owner)  # all in the side's style

    center = to_world[:3, 3]
    side = {
        "type": "cylinder",
        "p0": (center - along / 2).tolist(),
        "p1": (center + along / 2).tolist(),
        "radius": radius,
    }
    shapes = [_Shape(styles[0], owner, side)]
    x = across1 / radius
    for sign, style in ((-1, styles[1]), (1, styles[2])):
        z = sign * along / np.linalg.norm(along)  # the disc's normal points out of the cylinder
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack([x, np.cross(z, x), z]) * radius
        matrix[:3, 3] = center + sign * along / 2
        shapes.append(_Shape(style, owner, {"type": "disk", "to_world": matrix}))
    return shapes


# ---------------------------------------------------------------- scene, camera and light


def scene_dict(
    mi, view: _View, shapes: list[_Shape], light: str, spp: int, max_depth: int, seed: int
) -> dict:
    """The Mitsuba scene as a dictionary for ``mi.load_dict``."""
    if light not in LIGHTS:
        raise ValueError(f"Unknown light {light!r}; use one of {', '.join(LIGHTS)}.")
    picture = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": max_depth, "hide_emitters": True},
        "sensor": _sensor(mi, view, spp, seed),
    }
    if light == "sun":
        picture["sun"] = {
            "type": "directional",
            "direction": (-view.camera.light).tolist(),  # the way the light travels
            "irradiance": {"type": "rgb", "value": [np.pi * (1 - AMBIENT)] * 3},
        }
        picture["surroundings"] = {"type": "constant", "radiance": {"type": "rgb", "value": [AMBIENT] * 3}}
    else:
        picture["surroundings"] = {"type": "constant", "radiance": {"type": "rgb", "value": [1.0] * 3}}
    for shape in shapes:
        picture[shape.name] = shape.build(mi, _bsdf(shape.style))
    return picture


def _sensor(mi, view: _View, spp: int, seed: int) -> dict:
    """An orthographic camera that sees exactly what the ray tracer's view sees."""
    camera = view.camera
    rows, columns = view.shape
    # the middle of the image, on the screen, in world units
    middle_x = view.low[0] + (view.width / 2 - view.margin) / view.scale
    middle_y = view.high[1] - (view.height / 2 - view.margin) / view.scale
    target = middle_x * camera.right + middle_y * camera.up
    origin = target + view.far * camera.toward_viewer  # so that depths match the ray tracer's
    half_width = view.width / 2 / view.scale  # Mitsuba scales the height by the film's shape
    to_world = mi.ScalarTransform4f().look_at(
        origin=origin.tolist(), target=target.tolist(), up=camera.up.tolist()
    ) @ mi.ScalarTransform4f().scale([half_width, half_width, 1])
    return {
        "type": "orthographic",
        "to_world": to_world,
        "far_clip": float(2 * view.far + 4 * view.size),
        "film": {
            "type": "hdrfilm",
            "width": columns,
            "height": rows,
            "rfilter": {"type": "box"},
            "pixel_format": "rgba",
        },
        "sampler": {"type": "independent", "sample_count": spp, "seed": seed},
    }


def _depth_and_owner(mi, view: _View, shapes: list[_Shape], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Per subpixel: the depth of the first opaque surface (inf if none) and its scene item
    (-1 if none), rendered by Mitsuba with one ray per subpixel.

    For this, every opaque shape gets a color of its own, which says which shape a subpixel
    shows: red counts the shapes, green the thousands.
    """
    rows, columns = view.shape
    opaque = [shape for shape in shapes if shape.opaque]
    if not opaque:
        return np.full((rows, columns), np.inf), np.full((rows, columns), -1)
    picture = {
        "type": "scene",
        "integrator": {"type": "aov", "aovs": "depth:depth,color:albedo"},
        "sensor": _sensor(mi, view, 1, seed),
    }
    for number, shape in enumerate(opaque, start=1):
        code = [(number % 1000) / 1000, (number // 1000) / 1000, 0.0]
        picture[shape.name] = shape.build(mi, _matte(code))
    values = np.asarray(mi.render(mi.load_dict(picture), seed=seed), dtype=float)
    depth, red, green = values[..., 0], values[..., 1], values[..., 2]  # the channels as listed in "aovs"
    number = np.round(red * 1000).astype(int) + 1000 * np.round(green * 1000).astype(int)
    owners = np.array([-1] + [shape.owner for shape in opaque])
    hit = (depth > 0) & (number > 0) & (number < len(owners))
    owner = np.where(hit, owners[np.where(hit, number, 0)], -1)
    return np.where(hit, depth, np.inf), owner
