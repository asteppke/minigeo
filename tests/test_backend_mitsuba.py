"""The Mitsuba 3 backend. Skipped where Mitsuba is not installed (it is not in minigeo[all])."""

import numpy as np
import pytest

from minigeo import Box, Cylinder, Line, Scene, Sphere, Transform
from minigeo.backends.mitsuba import linear, srgb

pytest.importorskip("PIL")


def test_colors_go_to_linear_light_and_back():
    values = np.linspace(0, 1, 11)
    assert np.allclose(srgb(np.array(linear(values))), values)
    assert linear([1.0, 0.5, 0.0]) == pytest.approx([1.0, 0.214, 0.0], abs=1e-3)


mi = pytest.importorskip("mitsuba")


def small_scene():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="C0", label="box")
    scene.add(Sphere([1.5, 0, 0], radius=0.5), fill="C1")
    scene.add(Cylinder([-1.5, 0, -0.5], radius=0.4, height=1), fill="C2")
    return scene


def coverage(image):
    return image.pixels[..., 3] > 127


def test_same_picture_as_the_ray_tracer():
    scene = small_scene()
    mitsuba = scene.render("mitsuba", width=120, spp=4)
    traced = scene.render("raytrace", width=120)
    assert mitsuba.pixels.shape == traced.pixels.shape
    assert np.mean(coverage(mitsuba) == coverage(traced)) > 0.98  # the same outlines
    assert mitsuba.pixels[0, 0, 3] == 0  # transparent background


def test_background_and_save(tmp_path):
    scene = small_scene()
    image = scene.render("mitsuba", width=60, spp=2, background="white")
    assert np.all(image.pixels[0, 0] == [255, 255, 255, 255])
    scene.save(tmp_path / "scene.png", backend="mitsuba", width=60, spp=2)
    assert (tmp_path / "scene.png").stat().st_size > 0


def test_lines_are_hidden_behind_surfaces():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="white")
    scene.add(Line([-2, 2, 0.2], [2, 2, 0.2]), stroke="red", line_width=3)  # behind the box
    scene.add(Line([-2, -2, -0.2], [2, -2, -0.2]), stroke="blue", line_width=3)  # in front
    pixels = scene.render("mitsuba", width=500, spp=1, elev=0, azim=-90).pixels
    red = (pixels[..., 0] > 150) & (pixels[..., 1] < 100) & (pixels[..., 2] < 100)
    blue = (pixels[..., 2] > 150) & (pixels[..., 0] < 100)
    columns = pixels.shape[1]
    box = slice(int(0.42 * columns), int(0.58 * columns))  # the box covers the middle fifth or so
    assert red[:, : box.start].any() and not red[:, box].any()
    assert blue[:, box].any()


def test_labels_are_not_hidden_by_their_own_shape():
    from minigeo.backends.mitsuba import _depth_and_owner, _mitsuba, _shapes
    from minigeo.backends.raytrace import _Contents, _View
    from minigeo.camera import Camera

    contents = _Contents.collect(small_scene())
    camera = Camera()
    view = _View.fitting(contents.points(camera), camera, 80, 3, 1)
    depth, owner = _depth_and_owner(_mitsuba(None), view, _shapes(contents), 0)
    assert set(np.unique(owner)) == {-1, 0, 1, 2}  # background and the three items
    assert np.all(np.isinf(depth[owner == -1])) and np.all(np.isfinite(depth[owner >= 0]))


def test_exact_shapes_and_meshes():
    from minigeo.backends.mitsuba import _cylinder, _sphere
    from minigeo.exact import unit_map
    from minigeo.style import DEFAULT_STYLE

    ball = Sphere([0, 0, 0], radius=2)
    (exact,) = _sphere(unit_map(ball)[1], DEFAULT_STYLE, 0)
    assert exact.exact["type"] == "sphere" and exact.exact["radius"] == pytest.approx(2)
    Transform().scale([1, 1, 2]) @ ball
    (stretched,) = _sphere(unit_map(ball)[1], DEFAULT_STYLE, 0)
    assert stretched.exact is None and len(stretched.triangles) > 1000  # a mesh

    can = Cylinder([0, 0, 0], radius=0.5, height=3)
    can.rotate("x", 30)
    side, bottom, top = _cylinder(unit_map(can)[1], (DEFAULT_STYLE,) * 3, 0)
    assert side.exact["type"] == "cylinder" and bottom.exact["type"] == top.exact["type"] == "disk"
    assert side.exact["radius"] == pytest.approx(0.5)
    assert np.linalg.norm(np.subtract(side.exact["p1"], side.exact["p0"])) == pytest.approx(3)
    Transform().scale([2, 1, 1]) @ can
    assert len(_cylinder(unit_map(can)[1], (DEFAULT_STYLE,) * 3, 0)) == 1  # an elliptic mesh


def test_see_through_faces_and_lights():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="red", opacity=0.3)
    see_through = scene.render("mitsuba", width=60, spp=8)
    alpha = see_through.pixels[..., 3]
    assert 0 < np.median(alpha[alpha > 0]) < 250
    sky = scene.render("mitsuba", width=60, spp=8, light="sky")
    assert not np.array_equal(sky.pixels, see_through.pixels)
    with pytest.raises(ValueError, match="Unknown light"):
        scene.render("mitsuba", width=60, light="studio")
