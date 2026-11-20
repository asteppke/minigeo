"""The ray tracing backend: exact shapes, light, shadows, transparency, lines and labels."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from minigeo import Box, Cone, Cylinder, Group, Line, Plane, Point, Rectangle, Scene, Sphere, Transform
from minigeo.backends import raytrace
from minigeo.backends.raytrace import RaytracedImage, render

TOP_DOWN = {"elev": 90, "azim": -90}  # x to the right, y up on the screen


def coverage(image: RaytracedImage) -> np.ndarray:
    return image.pixels[..., 3] / 255


def pixel(image: RaytracedImage, view, point) -> np.ndarray:
    """The RGBA value where a 3D point ends up (via the backend's own view)."""
    xy, _ = view.subpixels(np.atleast_2d(point))
    column, row = (xy[0] / view.samples).astype(int)
    return image.pixels[row, column]


def view_of(scene, **options):
    contents = raytrace._Contents.collect(scene)
    camera = raytrace.Camera(options.get("elev", 30), options.get("azim", -60), "z")
    width = options.get("width", 500)
    return raytrace._View.fitting(
        contents.points(camera), camera, width, 20 * width / 500, options.get("samples", 3)
    )


def test_image_size_and_transparent_background():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)))
    image = render(scene, width=300)
    assert image.pixels.dtype == np.uint8
    assert image.width == 300
    assert image.pixels.shape == (image.height, 300, 4)
    assert image.pixels[0, 0, 3] == 0  # a corner of the margin is transparent
    assert coverage(image).max() == 1


def test_background_color():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)))
    image = render(scene, width=100, background="white")
    assert (image.pixels[0, 0] == [255, 255, 255, 255]).all()


def test_sphere_is_exactly_round():
    """Seen from above, the area of a coarse sphere matches a circle, not a 6-sided polygon."""
    scene = Scene()
    scene.add(Sphere([0, 0, 0], radius=1, segments=6))
    image = render(scene, width=400, margin=0, **TOP_DOWN, samples=4)
    area = coverage(image).sum() / (400 / 2) ** 2  # in units of radius squared
    assert area == pytest.approx(np.pi, rel=0.01)


def test_cylinder_is_exactly_round():
    scene = Scene()
    scene.add(Cylinder([0, 0, 0], radius=1, height=1, segments=5))
    image = render(scene, width=400, margin=0, **TOP_DOWN, samples=4)
    area = coverage(image).sum() / (400 / 2) ** 2
    assert area == pytest.approx(np.pi, rel=0.01)


def test_stretched_sphere_is_an_ellipsoid():
    scene = Scene()
    ellipsoid = scene.add(Sphere([0, 0, 0], radius=1, segments=6))
    Transform().scale([1, 0.5, 1]) @ ellipsoid
    image = render(scene, width=400, margin=0, **TOP_DOWN, samples=4)
    assert image.height == 200
    area = coverage(image).sum() / 200**2  # pixels per unit: 200
    assert area == pytest.approx(np.pi * 0.5, rel=0.01)


def test_mirrored_cylinder_is_still_exact():
    scene = Scene()
    cylinder = scene.add(Cylinder([0, 0, 0], radius=1, height=1, segments=5))
    Transform().scale([-1, 1, 1]) @ cylinder
    contents = raytrace._Contents.collect(scene)
    assert contents.surfaces[0].kind == "cylinder"


def test_cones_and_differently_styled_sides_use_faces():
    from minigeo import Style

    scene = Scene()
    scene.add(Cone([0, 0, 0], radius=1, height=1))
    scene.add(Cylinder([3, 0, 0], radius=1, height=1, segments=4))
    contents = raytrace._Contents.collect(scene)
    assert {s.kind for s in contents.surfaces} == {"face", "cylinder"}

    styled = Scene()
    styled.add(Cylinder([0, 0, 0], radius=1, height=1, segments=4), face_styles={0: Style(fill="red")})
    assert {s.kind for s in raytrace._Contents.collect(styled).surfaces} == {"face"}


def test_cylinder_caps_keep_their_face_styles():
    from minigeo import Style

    scene = Scene()
    top = 16 + 1  # 16 side faces, then the bottom, then the top
    scene.add(Cylinder([0, 0, 0], radius=1, height=1), fill="blue", face_styles={top: Style(fill="red")})
    image = render(scene, width=200, **TOP_DOWN, shadows=False, ambient=1)
    middle = image.pixels[image.height // 2, image.width // 2]
    assert middle[0] > 200 and middle[2] < 50  # the red top


def test_lambert_shading_darkens_faces_turned_from_the_light():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="white", stroke="none")
    view = view_of(scene)
    image = render(scene, shadows=False)
    top = pixel(image, view, [0, 0, 0.5])
    front = pixel(image, view, [0, -0.5, 0])
    right = pixel(image, view, [0.5, 0, 0])
    assert len({int(top[0]), int(front[0]), int(right[0])}) == 3  # three different brightnesses
    assert min(top[0], front[0], right[0]) >= round(255 * raytrace.AMBIENT) - 1


def test_ambient_one_means_no_shading():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="white", stroke="none")
    image = render(scene, ambient=1)
    colors = image.pixels[coverage(image) == 1][:, :3]
    assert colors.min() == 255


def test_shadows_fall_on_the_floor():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(4, 4, 0.1), anchor="top"), fill="white", stroke="none")
    scene.add(Box([0, 0, 0.5], size=(0.5, 0.5, 0.5)), fill="white", stroke="none")
    light = raytrace.Camera(30, -60).light
    beside = [1.5, -1.5, 0]
    behind = np.array([0, 0, 0.5]) - light * (
        0.5 / light[2]
    )  # where the light through the box's middle lands
    view = view_of(scene)
    with_shadows = render(scene)
    without = render(scene, shadows=False)
    assert pixel(with_shadows, view, behind)[0] < pixel(without, view, behind)[0]
    assert pixel(with_shadows, view, beside)[0] == pixel(without, view, beside)[0]


def test_transparent_faces_blend_with_what_is_behind():
    scene = Scene()
    scene.add(Box([0, 0, -1], size=(1, 1, 0.2)), fill="red", stroke="none")
    scene.add(Rectangle([0, 0, 0], size=(2, 2)), fill="blue", opacity=0.5, stroke="none")
    image = render(scene, **TOP_DOWN, shadows=False, ambient=1)
    middle = image.pixels[image.height // 2, image.width // 2]
    assert middle[0] == pytest.approx(128, abs=2) and middle[2] == pytest.approx(128, abs=2)
    edge = image.pixels[image.height // 2, 25]  # only the half-transparent rectangle
    assert edge[3] == pytest.approx(128, abs=2)


def test_translucent_faces_cast_lighter_shadows():
    def floor_brightness(opacity):
        scene = Scene()
        scene.add(Box([0, 0, 0], size=(4, 4, 0.1), anchor="top"), fill="white", stroke="none")
        scene.add(Rectangle([0, 0, 1], size=(4, 4)), fill="white", opacity=opacity, stroke="none")
        surfaces = raytrace._Surfaces(raytrace._Contents.collect(scene).surfaces, 5)
        lighting = raytrace._Lighting(surfaces, raytrace.Camera().light, 0.4, True)
        return lighting.light_reaching(np.array([[0.0, 0.0, 0.001]]))[0]

    assert floor_brightness(1.0) == 0
    assert floor_brightness(0.25) == pytest.approx(0.75)


def test_open_faces_are_lit_from_both_sides():
    scene = Scene()
    plane = scene.add(Plane([0, 0, 0], [0, 0, -1]), fill="white", opacity=1.0, stroke="none")
    image = render(scene, **TOP_DOWN, shadows=False)
    assert plane.normal[2] < 0
    middle = image.pixels[image.height // 2, image.width // 2]
    assert middle[0] > 200


def test_back_edges_are_hidden_and_front_edges_drawn():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="white", stroke="red", line_width=3)
    view = view_of(scene)
    image = render(scene, shadows=False, ambient=1)
    front_edge = pixel(image, view, [0.5, -0.5, 0])  # vertical edge facing the viewer
    hidden_edge = pixel(image, view, [-0.5, 0.5, -0.25])  # back edge, behind the box
    assert front_edge[0] > 200 and front_edge[1] < 80
    assert hidden_edge[0] > 200 and hidden_edge[1] > 200  # white face, no red


def test_lines_behind_solids_are_hidden_unless_on_top():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="white", stroke="none")
    scene.add(Line([0, 0, -2], [0, 0, 2]), stroke="red", line_width=3)  # through the box
    view = view_of(scene)
    image = render(scene, shadows=False, ambient=1)
    assert pixel(image, view, [0, 0, 0])[1] > 200  # hidden inside the box
    assert pixel(image, view, [0, 0, 1.5])[1] < 80  # visible above it

    scene.add_frame([0, 0, 0], length=1.5, labels=("", "", ""))  # on top by default
    view = view_of(scene)
    image = render(scene, shadows=False, ambient=1)
    z_arrow = pixel(image, view, [0, 0, 0.3])  # inside the box, but drawn on top
    assert z_arrow[2] > 200 and z_arrow[0] < 80


def test_dashed_lines_have_gaps():
    def drawn(dash):
        scene = Scene()
        scene.add(Line([0, 0, 0], [1, 0, 0]), dash=dash, line_width=2)
        return coverage(render(scene, **TOP_DOWN)).sum()

    assert 0.4 * drawn("solid") < drawn("dashed") < 0.8 * drawn("solid")
    assert drawn("dotted") < drawn("dashed")


def test_arrow_tips():
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 0, 0]), line_width=2)
    plain = coverage(render(scene, **TOP_DOWN)).sum()
    scene.items[0] = scene.items[0].__class__(
        scene.items[0].geometry, scene.items[0].style.with_(arrow=True), None
    )
    arrow = coverage(render(scene, **TOP_DOWN)).sum()
    assert arrow > plain


def test_points_are_dots():
    scene = Scene()
    scene.add(Point([0, 0, 0]), fill="red")
    scene.add(Point([1, 0, 0]), fill="red")
    image = render(scene, **TOP_DOWN)
    assert coverage(image).sum() > 20
    assert image.pixels[coverage(image) > 0.99][:, 0].min() > 200


def test_labels_are_drawn_and_hidden_by_other_objects():
    def label_ink(add_wall):
        scene = Scene()
        scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="white", stroke="none", label="label", label_at="top")
        if add_wall:  # a wall between the viewer and the box
            scene.add(Box([0, 0, 2], size=(3, 3, 0.1)), fill="white", stroke="none")
        image = render(scene, **TOP_DOWN, shadows=False, ambient=1, width=300, background="white")
        rgb = image.pixels[..., :3].astype(int)
        return int((rgb.sum(axis=2) < 200).sum())  # dark text pixels

    assert label_ink(False) > 10  # its own box does not hide the label
    assert label_ink(True) == 0  # the wall does


def test_labels_of_groups_belong_to_their_members():
    scene = Scene()
    group = Group([Box([0, 0, 0], size=(1, 1, 1)), Box([0.5, 0, 0], size=(1, 1, 1))])
    scene.add(group, fill="white", stroke="none", label="pair", label_at="center")  # inside both boxes
    image = render(scene, **TOP_DOWN, shadows=False, ambient=1, background="white")
    assert (image.pixels[..., :3].astype(int).sum(axis=2) < 600).sum() > 10


def test_dimension_labels_are_centered_text():
    from minigeo import Dimension

    scene = Scene()
    scene.add(Dimension([0, 0, 0], [1, 0, 0], offset=[0, -0.2, 0]))
    image = render(scene, **TOP_DOWN, width=300)
    assert coverage(image).sum() > 50


def test_line_widths_scale_with_the_image():
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 0, 0]), line_width=2)
    small = render(scene, **TOP_DOWN, width=250)
    large = render(scene, **TOP_DOWN, width=500)
    ratio = coverage(large).sum() / coverage(small).sum()
    assert ratio == pytest.approx(4, rel=0.15)


def test_samples_must_be_positive():
    with pytest.raises(ValueError, match="samples"):
        render(Scene(), samples=0)


def test_single_sample_has_hard_edges():
    scene = Scene()
    scene.add(Sphere([0, 0, 0], radius=1), stroke="none")
    image = render(scene, samples=1, width=100)
    assert set(np.unique(image.pixels[..., 3])) <= {0, 255}


def test_png_and_saving(tmp_path):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)))
    image = render(scene, width=100)
    assert image.to_png().startswith(b"\x89PNG")
    assert image._repr_png_() == image.to_png()
    image.save(tmp_path / "box.png")
    image.save(tmp_path / "box.jpg")
    from PIL import Image

    assert Image.open(tmp_path / "box.png").mode == "RGBA"
    assert Image.open(tmp_path / "box.jpg").mode == "RGB"


def test_scene_render_and_save(tmp_path):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(1, 1, 1)))
    assert isinstance(scene.render("raytrace", width=100), RaytracedImage)
    scene.save(tmp_path / "box.png", backend="raytrace", width=120)
    from PIL import Image

    assert Image.open(tmp_path / "box.png").size[0] == 120
    with pytest.raises(ValueError, match="raytrace"):
        scene.save(tmp_path / "box.png", backend="svg")


def test_empty_scene_and_lines_only():
    assert render(Scene(), width=50).width == 50
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 1, 1]))
    assert coverage(render(scene, width=100)).sum() > 0


def test_flat_and_edge_on_faces():
    scene = Scene()
    scene.add(Rectangle([0, 0, 0], size=(1, 1)), fill="none", stroke="red")  # outline only
    scene.add(Rectangle([2, 0, 0], size=(1, 1)).copy(), fill="C1")
    Transform().rotate("x", 90, about=[2, 0, 0]) @ scene.items[1].geometry
    image = render(scene, **TOP_DOWN, width=200)  # the second one is seen edge-on
    assert coverage(image).sum() > 0
